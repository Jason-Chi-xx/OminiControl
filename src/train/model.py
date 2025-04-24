import lightning as L
from diffusers.pipelines import FluxPipeline
import copy
import torch
import torch.nn as nn
from peft import LoraConfig, get_peft_model_state_dict
from transformers import Qwen2_5_VLForConditionalGeneration, AutoProcessor, CLIPVisionModel, CLIPModel, Qwen2Model, Qwen2Config, Cache
import prodigyopt
import torch.nn.functional as F
import torchvision.transforms as T
from qwen_vl_utils import process_vision_info
from ..flux.transformer import tranformer_forward
from ..flux.condition import Condition
from ..flux.pipeline_tools import encode_images, prepare_text_input
from typing import Union, Optional, List, Tuple, Dict, Any
from qwen_model_utils import Qwen2RMSNorm, Qwen2DecoderLayer

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
        query_config: dict = {},
        image_encoder_config: dict = {},
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
        # self.Query = Query(**query_config).to(device).to(dtype=dtype)
        # self.Query.train()
        # Initialize connector
        # self.connector = Connector(**connector_config).to(device).to(dtype=dtype)
        self.connector = QwenConnector().to(device).to(dtype=dtype)
        self.connector.requires_grad = True
        self.connector.train()
        self.mllm = MetaUnderstandingModel(**mllm_config, device=device).to(device).to(dtype=dtype)
        # self.mllm.requires_grad_(False).eval()
        self.clip_model = CLIPModel.from_pretrained(image_encoder_config['name'])
        self.image_encoder = self.clip_model.vision_model.to(device).to(dtype=dtype)
        self.image_encoder.requires_grad_(False).eval()
        self.clip_vision_projection = self.clip_model.visual_projection.to(device).to(dtype=dtype)
        self.clip_vision_projection.requires_grad = True
        self.clip_vision_projection.train()
        self.clip_processor = AutoProcessor.from_pretrained(image_encoder_config['name'])

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
        # self.trainable_params = list(filter(lambda p: p.requires_grad, self.Query.parameters()))
        self.trainable_params = [self.mllm.learnable_query]
        self.trainable_params.extend(list(filter(lambda p: p.requires_grad, self.connector.parameters())))
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
        prompts = batch["description"]
        imagebase64 = batch["imagebase64"]
        image_pils = [T.ToPILImage()(img) for img in imgs]

        with torch.no_grad():
            text_query = self.mllm(
                image=imagebase64,
                prompt=prompts,
            )
            image_pils = self.clip_processor(images=image_pils,return_tensors="pt").to(self.device)
            image_embeds = self.image_encoder(**image_pils)
            pooled_image_embeds = image_embeds.pooler_output
            pooled_image_embeds = pooled_image_embeds.to(dtype=self.dtype, device=self.device)
            pooled_prompt_embeds = self.clip_vision_projection(pooled_image_embeds)

        prompt_embeds = self.connector(text_query)

        # condition_query = self.Query(text_query)
        prompt_embeds, pooled_prompt_embeds, text_ids = prepare_text_input(
            self.flux_pipe, prompts, prompt_embeds, pooled_prompt_embeds
        )
        # Prepare text input
        # Prepare inputs
        with torch.no_grad():
            # Prepare image input
            x_0, img_ids = encode_images(self.flux_pipe, imgs)


            # Prepare t and x_t
            t = torch.sigmoid(torch.randn((imgs.shape[0],), device=self.device))
            x_1 = torch.randn_like(x_0).to(self.device)
            t_ = t.unsqueeze(1).unsqueeze(1)
            x_t = ((1 - t_) * x_0 + t_ * x_1).to(self.dtype)

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
            condition_latents=None,
            condition_ids=None,
            condition_type_ids=None,
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

class SimpleConnector(nn.Module):
    def __init__(self, input_dim:int=3584, output_dim:int=4096):
        super().__init__()
        self.input_dim = input_dim
        self.output_dim = output_dim
        self.proj_out = nn.Sequential(
            nn.Linear(input_dim, input_dim),
            nn.LayerNorm(input_dim),
            nn.GELU(),
            nn.Linear(input_dim, output_dim),
        )

    def forward(self, x):
        return self.proj_out(x)
        

class Connector(nn.Module):
    def __init__(self, input_dim:int=3584, output_dim:int=4096, output_dim_2:int=768, num_layers:int=2, n_heads:int=16, dropout:float=0.1):
        super().__init__()
        self.input_dim = input_dim
        self.output_dim = output_dim
        self.num_layers = num_layers
        self.n_heads = n_heads
        self.dropout = dropout
        encoder_layer = nn.TransformerEncoderLayer(d_model=input_dim, nhead=n_heads, dim_feedforward=2* input_dim, dropout=dropout, batch_first=True)
        self.encoder = nn.TransformerEncoder(encoder_layer, num_layers=num_layers)
        self.proj_out = nn.Linear(input_dim, output_dim)
        # self.proj_out_2 = nn.Linear(input_dim, output_dim_2)
        
    def forward(self, x):
        x = self.encoder(x)
        x_1 = self.proj_out(x)
        # Pool the sequence dimension and project to output_dim_2
        # x_2 = self.proj_out_2(x.mean(dim=1))  # Shape: (batch_size, output_dim_2)
        return x_1

class Qwen2ModelNoCausal(Qwen2Model):

    def _update_causal_mask(
        self,
        attention_mask: Union[torch.Tensor, "BlockMask"],
        input_tensor: torch.Tensor,
        cache_position: torch.Tensor,
        past_key_values: Cache,
        output_attentions: bool = False,
    ):
        return None



class QwenConnector(nn.Module):
    def __init__(self, input_dim:int=3584, output_dim:int=4096, device:str="cuda", dtype:torch.dtype=torch.bfloat16):
        super().__init__()
        self.input_dim = input_dim
        self.output_dim = output_dim
        self.config = Qwen2Config.from_json_file("/hy-tmp/Qwen2-7B/config.json")
        self.layers = nn.ModuleList([
            Qwen2DecoderLayer(self.config, layer_idx) for layer_idx in range(6)
        ])
        self.norm = Qwen2RMSNorm(self.input_dim, eps=self.config.rms_norm_eps)
        self.proj_out = nn.Linear(self.input_dim, self.output_dim)

    def forward(self, x):
        x = self.transformer(inputs_embeds=x)
        return self.proj_out(x)


class UnderstandingModel(nn.Module):
    def __init__(
        self, 
        mllm:str = "Qwen/Qwen2.5-VL-7B-Instruct",
        num_queries:int=256,
        hidden_size:int=3584,
        device:str="cuda",
    ):
        super().__init__()
        self.device = device
        self.mllm = Qwen2_5_VLForConditionalGeneration.from_pretrained(
            mllm,
            torch_dtype=torch.bfloat16,
            attn_implementation="flash_attention_2",
        ).to(device)
        self.processor = AutoProcessor.from_pretrained(mllm)
        self.num_queries=num_queries
        self.hidden_size=hidden_size
        
    def forward(self, image, prompt):
        if isinstance(image, str):
            img = image
        elif isinstance(image, list):
            img = image[0]
        messages = [
            {
                "role": "user",
                "content": [
                    {
                        "type": "image",
                        "image": f"data:image;base64,{img}",
                    },
                    {
                        "type": "text", 
                        "text": prompt
                    },
                ],
            }
        ]
        text = self.processor.apply_chat_template(
            messages, tokenize=False, add_generation_prompt=True
        )
        image_inputs, _ = process_vision_info(messages)
        inputs = self.processor(
            text=[text],
            images=image_inputs,
            padding=True,
            return_tensors="pt",
        )
        inputs = inputs.to(self.device)
        outputs = self.mllm.forward(**inputs, output_hidden_states=True)
        hidden_states = outputs['hidden_states'][-1] # [batch_size, 1, 3584]
        return hidden_states
    
class HierarchicalQuery(nn.Module):
    def __init__(self, num_queries:int=256, hidden_size:int=3584, num_levels:int=3, dropout:float=0.1):
        super().__init__()
        self.num_queries = num_queries
        self.hidden_size = hidden_size
        self.num_levels = num_levels
        
        # 为每个层次创建可学习的查询参数
        self.level_queries = nn.ParameterList([
            nn.Parameter(torch.randn(num_queries, hidden_size))
            for _ in range(num_levels)
        ])
        
        # 层间转换投影
        self.level_projections = nn.ModuleList([
            nn.Sequential(
                nn.Linear(hidden_size, hidden_size),
                nn.LayerNorm(hidden_size),
                nn.GELU(),
                nn.Dropout(dropout)
            )
            for _ in range(num_levels - 1)
        ])
        
        # 输出投影
        self.output_projection = nn.Linear(hidden_size, hidden_size)
        
    def forward(self, hidden_states):
        batch_size = hidden_states.shape[0]
        current_features = hidden_states
        
        level_outputs = []
        
        # 逐层进行注意力计算
        for level in range(self.num_levels):
            # 获取当前层的查询参数
            level_query = self.level_queries[level].unsqueeze(0).expand(batch_size, -1, -1)
            
            # 计算当前层的注意力
            attended_features = F.scaled_dot_product_attention(
                level_query,
                current_features,
                current_features,
                scale=1.0 / (self.hidden_size ** 0.5)
            )
            
            level_outputs.append(attended_features)
            
            # 如果不是最后一层，进行转换以准备下一层
            if level < self.num_levels - 1:
                # 将当前层的输出作为下一层的上下文输入
                current_features = self.level_projections[level](attended_features)
        
        # 整合所有层次的输出
        # 方法1: 只使用最后一层输出
        final_output = level_outputs[-1]
        
        # 方法2: 结合所有层次的输出（可选）
        # all_outputs = torch.cat(level_outputs, dim=1)  # 沿着序列维度拼接
        # final_output = self.output_projection(all_outputs)
        
        return final_output
    
class Query(nn.Module):
    def __init__(self, num_queries:int=256, hidden_size:int=3584):
        super().__init__()
        self.num_queries = num_queries
        self.hidden_size = hidden_size
        
        
        self.query = nn.Parameter(torch.randn(num_queries, hidden_size))
        self.query.requires_grad = True
        
    def forward(self, hidden_states):
        # 使用self.query作为查询，hidden_states作为键和值进行注意力计算
        batch_size = hidden_states.shape[0]
        
        # 扩展查询维度以匹配批次大小
        # [num_queries, hidden_size] -> [batch_size, num_queries, hidden_size]
        query = self.query.unsqueeze(0).expand(batch_size, -1, -1)
        
        # 使用PyTorch的scaled_dot_product_attention函数
        # query: [batch_size, num_queries, hidden_size]
        # key=value=hidden_states: [batch_size, seq_len, hidden_size]
        attended_features = F.scaled_dot_product_attention(
            query, 
            hidden_states, 
            hidden_states,
            scale=1.0 / (self.hidden_size ** 0.5)
        )
        
        return attended_features
    
class MetaUnderstandingModel(nn.Module):
    def __init__(
        self, 
        mllm:str = "Qwen/Qwen2.5-VL-7B-Instruct",
        num_queries:int=256,
        hidden_size:int=3584,
        device:str="cuda",
    ):
        super().__init__()
        self.device = device
        self.mllm = Qwen2_5_VLForConditionalGeneration.from_pretrained(
            mllm,
            torch_dtype=torch.bfloat16,
            # attn_implementation="flash_attention_2",
        ).to(device)
        for param in self.mllm.parameters():
            param.requires_grad = False
        self.mllm.eval()
        self.processor = AutoProcessor.from_pretrained(mllm)
        self.num_queries=num_queries
        self.hidden_size=hidden_size
        self.learnable_query = nn.Parameter(torch.randn(num_queries, hidden_size))
        self.learnable_query.requires_grad = True
        
    def forward(self, image, prompt):
        if isinstance(image, str):
            img = image
        elif isinstance(image, list):
            img = image[0]
        messages = [
            {
                "role": "user",
                "content": [
                    {
                        "type": "image",
                        "image": f"data:image;base64,{img}",
                    },
                    {
                        "type": "text", 
                        "text": prompt
                    },
                ],
            }
        ]
        text = self.processor.apply_chat_template(
            messages, tokenize=False, add_generation_prompt=True
        )
        image_inputs, _ = process_vision_info(messages)
        inputs = self.processor(
            text=[text],
            images=image_inputs,
            padding=True,
            return_tensors="pt",
        )
        inputs = inputs.to(self.device)
        inputs_embeds = self.mllm.get_input_embeddings()(inputs.input_ids)
        pixel_values = inputs['pixel_values']
        if pixel_values is not None:
            pixel_values = pixel_values.type(self.mllm.dtype)
            image_embeds = self.mllm.visual(pixel_values, grid_thw=inputs.image_grid_thw)
            n_image_tokens = (inputs.input_ids == self.mllm.config.image_token_id).sum().item()
            n_image_features = image_embeds.shape[0]
            if n_image_tokens != n_image_features:
                raise ValueError(
                    f"Image features and image tokens do not match: tokens: {n_image_tokens}, features {n_image_features}"
                )

            mask = inputs.input_ids == self.mllm.config.image_token_id
            mask_unsqueezed = mask.unsqueeze(-1)
            mask_expanded = mask_unsqueezed.expand_as(inputs_embeds)
            image_mask = mask_expanded.to(inputs_embeds.device)

            image_embeds = image_embeds.to(inputs_embeds.device, inputs_embeds.dtype)
            inputs_embeds = inputs_embeds.masked_scatter(image_mask, image_embeds)
        batch_size = inputs_embeds.shape[0]
        # inputs_embeds = torch.cat([inputs_embeds, self.learnable_query.unsqueeze(0).expand(batch_size, -1, -1)], dim=1)
        outputs = self.mllm.model.forward(inputs_embeds=inputs_embeds)
        hidden_states = outputs['last_hidden_state'] # [batch_size, seq_len, 3584]
        hidden_states = hidden_states[:, -self.num_queries:, :]
        return hidden_states