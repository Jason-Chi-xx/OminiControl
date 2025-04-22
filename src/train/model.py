import lightning as L
from diffusers.pipelines import FluxPipeline
import torch
import torch.nn as nn
from peft import LoraConfig, get_peft_model_state_dict
from transformers import AutoTokenizer, AutoModel, AutoProcessor
import prodigyopt
import torch.nn.functional as F
from qwen_vl_utils import process_vision_info
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
        query_config: dict = {},
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
        self.Query = Query(**query_config)
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
        self.trainable_params = list(filter(lambda p: p.requires_grad, self.Query.parameters()))
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
        conditions = batch["condition"]
        prompts = batch["description"]
        with torch.no_grad():
            text_query = self.mllm(
                image=conditions,
                prompt=prompts,
            )
        condition_query = self.Query(text_query)
        prompt_embeds = self.connector(condition_query)
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

class Connector(nn.Module):
    def __init__(self, input_dim:int=3584, output_dim:int=4096, num_layers:int=2, n_heads:int=16, dropout:float=0.1):
        super().__init__()
        self.input_dim = input_dim
        self.output_dim = output_dim
        self.num_layers = num_layers
        self.n_heads = n_heads
        self.dropout = dropout
        encoder_layer = nn.TransformerEncoderLayer(d_model=input_dim, nhead=n_heads, dim_feedforward=2* input_dim, dropout=dropout, batch_first=True)
        self.encoder = nn.TransformerEncoder(encoder_layer, num_layers=num_layers)
        self.proj_out = nn.Linear(input_dim, output_dim)
        
    def forward(self, x):
        x = self.encoder(x)
        x = self.projector(x)
        return x

class UnderstandingModel(nn.Module):
    def __init__(
        self, 
        mllm:str = "Qwen/Qwen2.5-VL-7B-Instruct",
        num_queries:int=256,
        hidden_size:int=3584,
        device:str="cuda",
    ):
        super().__init__()
        self.mllm = AutoModel.from_pretrained(mllm)
        self.processor = AutoProcessor.from_pretrained(mllm)
        self.tokenizer = AutoTokenizer.from_pretrained(mllm)
        self.num_queries=num_queries
        self.hidden_size=hidden_size
        self.device = device
        
    def forward(self, image, prompt):
        messages = [
            {
                "role": "user",
                "content": [
                    {
                        "type": "image",
                        "image": image,
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
    
    