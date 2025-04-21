import lightning as L
from diffusers.pipelines import FluxPipeline
import torch
import torch.nn as nn
from peft import LoraConfig, get_peft_model_state_dict
from transformers import AutoTokenizer, AutoModel
import prodigyopt

from ..flux.transformer import tranformer_forward
from ..flux.condition import Condition
from ..flux.pipeline_tools import encode_images, prepare_text_input


class OminiModel(L.LightningModule):
    def __init__(
        self,
        flux_pipe_id: str,
        lora_path: str = None,
        lora_config: dict = None,
        device: str = "cuda",
        dtype: torch.dtype = torch.bfloat16,
        model_config: dict = {},
        optimizer_config: dict = None,
        gradient_checkpointing: bool = False,
        connector_config: dict = {},
        mllm_config: dict = {},
    ):
        # Initialize the LightningModule
        super().__init__()
        self.model_config = model_config
        self.optimizer_config = optimizer_config

        # Load the Flux pipeline
        self.flux_pipe: FluxPipeline = (
            FluxPipeline.from_pretrained(flux_pipe_id).to(dtype=dtype).to(device)
        )
        self.transformer = self.flux_pipe.transformer
        self.transformer.gradient_checkpointing = gradient_checkpointing
        self.transformer.train()

        # Freeze the Flux pipeline
        self.flux_pipe.text_encoder.requires_grad_(False).eval()
        self.flux_pipe.text_encoder_2.requires_grad_(False).eval()
        self.flux_pipe.vae.requires_grad_(False).eval()

        # Initialize LoRA layers
        self.lora_layers = self.init_lora(lora_path, lora_config)
        self.learnable_query = nn.Parameter(torch.randn(256, 3584))
        self.learnable_query.requires_grad = True
        # Initialize connector
        self.connector = Connector(**connector_config)
        self.connector.requires_grad = True
        self.connector.train()
        self.mllm = UnderstandingModel(**mllm_config)
        self.mllm.requires_grad_(False).eval()

        self.to(device).to(dtype)

    def init_lora(self, lora_path: str, lora_config: dict):
        assert lora_path or lora_config
        if lora_path:
            # TODO: Implement this
            raise NotImplementedError
        else:
            self.transformer.add_adapter(LoraConfig(**lora_config))
            # TODO: Check if this is correct (p.requires_grad)
            lora_layers = filter(
                lambda p: p.requires_grad, self.transformer.parameters()
            )
        return list(lora_layers)

    def save_lora(self, path: str):
        FluxPipeline.save_lora_weights(
            save_directory=path,
            transformer_lora_layers=get_peft_model_state_dict(self.transformer),
            safe_serialization=True,
        )

    def configure_optimizers(self):
        # Freeze the transformer
        self.transformer.requires_grad_(False)
        opt_config = self.optimizer_config

        # Set the trainable parameters
        self.trainable_params = [self.learnable_query, self.connector]
        self.trainable_params.extend(self.lora_layers)

        # Unfreeze trainable parameters
        for p in self.trainable_params:
            p.requires_grad_(True)

        # Initialize the optimizer
        if opt_config["type"] == "AdamW":
            optimizer = torch.optim.AdamW(self.trainable_params, **opt_config["params"])
        elif opt_config["type"] == "Prodigy":
            optimizer = prodigyopt.Prodigy(
                self.trainable_params,
                **opt_config["params"],
            )
        elif opt_config["type"] == "SGD":
            optimizer = torch.optim.SGD(self.trainable_params, **opt_config["params"])
        else:
            raise NotImplementedError

        return optimizer

    def training_step(self, batch, batch_idx):
        step_loss = self.step(batch)
        self.log_loss = (
            step_loss.item()
            if not hasattr(self, "log_loss")
            else self.log_loss * 0.95 + step_loss.item() * 0.05
        )
        return step_loss

    def step(self, batch):
        imgs = batch["image"]
        conditions = batch["condition"]
        condition_types = batch["condition_type"]
        prompts = batch["description"]
        position_delta = batch["position_delta"][0]
        position_scale = float(batch.get("position_scale", [1.0])[0])
        text_query = self.mllm(
            image=conditions,
            prompt=prompts,
            learnable_query=self.learnable_query,
        )
        prompt_embeds = self.connector(text_query)
        # Prepare text input
        prompt_embeds, pooled_prompt_embeds, text_ids = prepare_text_input(
            self.flux_pipe, prompts, prompt_embeds
        )
        # Prepare inputs
        with torch.no_grad():
            # Prepare image input
            x_0, img_ids = encode_images(self.flux_pipe, imgs)


            # Prepare t and x_t
            t = torch.sigmoid(torch.randn((imgs.shape[0],), device=self.device))
            x_1 = torch.randn_like(x_0).to(self.device)
            t_ = t.unsqueeze(1).unsqueeze(1)
            x_t = ((1 - t_) * x_0 + t_ * x_1).to(self.dtype)

            # Prepare conditions
            condition_latents, condition_ids = encode_images(self.flux_pipe, conditions)

            # Add position delta
            condition_ids[:, 1] += position_delta[0]
            condition_ids[:, 2] += position_delta[1]

            if position_scale != 1.0:
                scale_bias = (position_scale - 1.0) / 2
                condition_ids[:, 1] *= position_scale
                condition_ids[:, 2] *= position_scale
                condition_ids[:, 1] += scale_bias
                condition_ids[:, 2] += scale_bias

            # Prepare condition type
            condition_type_ids = torch.tensor(
                [
                    Condition.get_type_id(condition_type)
                    for condition_type in condition_types
                ]
            ).to(self.device)
            condition_type_ids = (
                torch.ones_like(condition_ids[:, 0]) * condition_type_ids[0]
            ).unsqueeze(1)

            # Prepare guidance
            guidance = (
                torch.ones_like(t).to(self.device)
                if self.transformer.config.guidance_embeds
                else None
            )

        # Forward pass
        transformer_out = tranformer_forward(
            self.transformer,
            # Model config
            model_config=self.model_config,
            # Inputs of the condition (new feature)
            condition_latents=condition_latents,
            condition_ids=condition_ids,
            condition_type_ids=condition_type_ids,
            # Inputs to the original transformer
            hidden_states=x_t,
            timestep=t,
            guidance=guidance,
            pooled_projections=pooled_prompt_embeds,
            encoder_hidden_states=prompt_embeds,
            txt_ids=text_ids,
            img_ids=img_ids,
            joint_attention_kwargs=None,
            return_dict=False,
        )
        pred = transformer_out[0]

        # Compute loss
        loss = torch.nn.functional.mse_loss(pred, (x_1 - x_0), reduction="mean")
        self.last_t = t.mean().item()
        return loss

class Connector(nn.Module):
    def __init__(self, connector_name:str="Qwen/Qwen2.5-0.5B", embedding_dim:int=4096):
        super().__init__()
        self.connector = AutoModel.from_pretrained(connector_name)
        self.tokenizer = AutoTokenizer.from_pretrained(connector_name)
        self.connector.config.is_causal = False

        self.projector = nn.Linear(self.connector.config.hidden_size, embedding_dim)

    def forward(self, x):
        x = self.connector(x)
        x = self.projector(x)
        return x
        

class UnderstandingModel(nn.Module):
    def __init__(
        self, 
        mllm:str = "Qwen/Qwen2.5-VL-7B-Instruct",
        num_queries:int=256,
        hidden_size:int=3584,
    ):
        super().__init__()
        self.mllm = AutoModel.from_pretrained(mllm)
        self.tokenizer = AutoTokenizer.from_pretrained(mllm)
        self.num_queries=num_queries
        self.hidden_size=hidden_size
        
        
    def forward(self, image, prompt, learnable_query):
        image_features = self.mllm.encode_image(image)
        prompt_features = self.mllm.encode_text(prompt)
        query_features = learnable_query.unsqueeze(0).repeat(prompt_features.shape[0], 1, 1)
        query_token_number = query_features.shape[1]
        concat_features = torch.cat([image_features, prompt_features, query_features], dim=1)
        output_query = self.mllm(inputs_embeds=concat_features)['hidden_states'][:, -query_token_number:]
        return output_query