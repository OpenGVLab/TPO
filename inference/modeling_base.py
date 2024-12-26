import io
import logging
import torch
import torch.utils.checkpoint
from torch import nn
from torch.nn import MSELoss
from transformers.modeling_outputs import (
    CausalLMOutputWithPast,
)
from typing import List, Optional, Tuple, Union
from transformers import LlamaForCausalLM

from torch.cuda.amp import autocast as autocast

from .modeling_vit import build_vit
from .modeling_qformer  import build_qformer
from .model_config import VideoChatEConfig
logger = logging.getLogger(__name__)

from transformers import LlamaTokenizer,AutoTokenizer,AutoModel,AutoModelForCausalLM,AutoProcessor
from transformers import AutoConfig, PreTrainedModel

import os
import sys


try:
    from third_party.sam2.build_sam import build_sam2_video_predictor
    from third_party.cgdetr.cg_detr.model import build_cgdetr_model
except:
    print("can not import sam2 and cg-detr, install them first.")

DEFAULT_IMG_TOKEN = "[IMG]"
DEFAULT_IMG_END_TOKEN = "[/IMG]"

DEFAULT_IMAGE_TOKEN = "<image>"
DEFAULT_VIDEO_TOKEN = "[VIDEO]"

IMG_TOKEN = "[<IMG_PLH>]"
VID_TOKEN = "[<VID_PLH>]"

BOX_START = '<box_begin>'
# BOX_END = '<box_end>'
ATBOXES_PLACEHOLDER = '<box_begin><boxes>'
# ATBOXES_PLACEHOLDER = '<box_begin>'
BOXES_PLACEHOLDER = '<boxes>'
EXPR_PLACEHOLDER = '<expr>'
QUESTION_PLACEHOLDER = '<question>'
TIME_START = '<time_begin>'
# TIME_END = '<time_end>'
TIME_PLACEHOLDER = '<temp>'
ATTEMP_PLACEHOLDER = TIME_START + TIME_PLACEHOLDER
# ATTEMP_PLACEHOLDER = TIME_START
TRACK_START='<track_begin>'
TRACK_PLACEHOLDER = '<tracking>'
TRACK_START_BOX = '<track_box>'
ATTRACK_PLACEHOLDER = TRACK_START + TRACK_PLACEHOLDER
need_template_list = ['REC', 'flickr', 'tracking', 'tracking2', 'tracking3', 'tracking4'] 

load_image_list = ['image', 'REC', 'flickr']
load_video_list = ['video', 'TVG', 'tracking', 'tracking2','tracking3', 'tracking4', 'TVG+HL']
special_tokens = [BOX_START, TIME_START, TIME_PLACEHOLDER, BOXES_PLACEHOLDER, TRACK_START, TRACK_PLACEHOLDER, TRACK_START_BOX]

def disabled_train(self, mode=True):
    """Overwrite model.train with this function to make sure train/eval mode
    does not change anymore."""
    return self


def freeze_module(module):
    for _, param in module.named_parameters():
        param.requires_grad = False
    module = module.eval()
    module.train = disabled_train
    return module


class LLMConfig(AutoConfig):
    model_type = "20b"


class BaseMLLM(PreTrainedModel):
    config_class = VideoChatEConfig
    def __init__(self, config,_tokenizer=None):
        # super().__init__(config)
        self.model_config = config.model_config
        self.tokenizer = _tokenizer
        
        config.cg_opt = None
        config.model_config = None
        config.model_tokenizer = None
        super().__init__(config)
        self.build_vision_encoder()
        self.build_llm()
        self.build_bridge()
        self.build_loss()
        
        self.load_pretrained_weights()
        try:
            if config.build_decoder:
                self.cg_opt = config.cg_opt
                self.build_bbox_decoder()
                self.build_sam()
                self.build_CGDETR()
        except:
            print("please install cgdetr and sam2 first")
        logger.info(f'Length of tokenizer and resize embedding: {len(self.tokenizer)}')

    
    def build_vision_encoder(self):
        if 'internvideo2' in self.model_config.vision_encoder.name.lower():
            encoder_name = self.model_config.vision_encoder.name
            logger.info(f"Build vision_encoder: {encoder_name}")
            if encoder_name == 'internvideo2-1B':
                self.vision_encoder = pretrain_internvideo2_giant_patch14_224_clean(self.model_config)

            else:
                raise ValueError(f"Not implemented: {encoder_name}")
        elif 'vit' in self.model_config.vision_encoder.name.lower():
            self.vision_encoder = build_vit(self.model_config)
        else:
            raise NotImplementedError(self.model_config.vision_encoder.name)

        if self.model_config.vision_encoder.vit_add_ln:
            self.vision_layernorm = nn.LayerNorm(self.model_config.vision_encoder.encoder_embed_dim, eps=1e-12)
        else:
            self.vision_layernorm = nn.Identity()

        self.freeze_vision_encoder = self.model_config.get("freeze_vision_encoder", False)

        if self.freeze_vision_encoder:
            logger.info("freeze vision encoder")
            freeze_module(self.vision_encoder)
            freeze_module(self.vision_layernorm)

    def build_CGDETR(self):
        self.cg_model, self.cg_criterion = build_cgdetr_model()
    
    def build_bridge(self):
        # ViT to LM: 1792 -> 6656 NOTE 768 is qformer dim
        self.project_up = nn.Linear(768, self.lm.config.hidden_size) # whether bias is needed?
        # LM to ViT: 6656 -> 1792
        self.project_down = nn.Linear(self.lm.config.hidden_size, 768)
        
        if 'qformer' in self.model_config.bridge.name.lower():
            from transformers import BertTokenizer
            self.qformer_tokenizer = BertTokenizer.from_pretrained("bert-base-uncased", truncation_side="left")
            self.qformer_tokenizer.add_special_tokens({"bos_token": "[DEC]"})
            self.qformer_tokenizer.padding_side = "left"
            if self.model_config.bridge.name == 'qformer':
                self.qformer, self.query_tokens = build_qformer(
                        self.model_config.bridge.num_query_token, self.model_config.vision_encoder.encoder_embed_dim,
                        qformer_hidden_dropout_prob=self.model_config.bridge.qformer_hidden_dropout_prob,
                        qformer_attention_probs_dropout_prob=self.model_config.bridge.qformer_attention_probs_dropout_prob,
                        qformer_drop_path_rate=self.model_config.bridge.qformer_drop_path_rate,
                )
            elif self.model_config.bridge.name == 'causal_qformer':
                self.qformer, self.query_tokens = build_causal_qformer(
                        self.model_config.bridge.num_query_token, self.model_config.vision_encoder.encoder_embed_dim,
                        qformer_hidden_dropout_prob=self.model_config.bridge.qformer_hidden_dropout_prob,
                        qformer_attention_probs_dropout_prob=self.model_config.bridge.qformer_attention_probs_dropout_prob
                )
            self.qformer.resize_token_embeddings(len(self.qformer_tokenizer))
            self.qformer.cls = None
            self.extra_num_query_token = self.model_config.bridge.extra_num_query_token
            if self.model_config.bridge.extra_num_query_token > 0:
                logger.info(f"Add extra {self.model_config.bridge.extra_num_query_token} tokens in QFormer")
                self.extra_query_tokens = nn.Parameter(
                    torch.zeros(1, self.model_config.bridge.extra_num_query_token, self.query_tokens.shape[-1])
                )
            
            self.freeze_bridge = self.model_config.get("freeze_bridge", False)
            if self.freeze_bridge:
                logger.info("freeze bridge")
                freeze_module(self.qformer)
                self.query_tokens.requires_grad = False

    def build_llm(self):
        self.lm_name = self.model_config.llm.name
        if self.model_config.llm.name == "vicuna1.5_7b":
            self.lm = LlamaForCausalLM.from_pretrained(self.model_config.llm.pretrained_llm_path)
            self.lm.gradient_checkpointing = self.model_config.llm.get("use_llama_gradient_checkpointing", True)
        elif self.model_config.llm.name == 'mistral_7b':
            from transformers import AutoModelForCausalLM

            config = AutoConfig.from_pretrained(
                self.model_config.llm.pretrained_llm_path,
                torch_dtype=torch.bfloat16,
                # attn_implementation="flash_attention_2",
            )
            self.lm = AutoModelForCausalLM.from_config(config)
        elif self.model_config.llm.name == 'internlm_20b':
            from transformers import AutoModelForCausalLM
            self.lm = AutoModelForCausalLM.from_pretrained(
                self.model_config.llm.pretrained_llm_path,
                torch_dtype=torch.bfloat16,
                trust_remote_code=True,
            )
            self.lm.gradient_checkpointing = True
            self.lm._set_gradient_checkpointing()
        else:
            raise NotImplementedError(self.model_config.llm.name)

        num_new_tokens = len(special_tokens)
        self.lm.resize_token_embeddings(len(self.tokenizer))

        input_embeddings = self.lm.get_input_embeddings().weight.data
        output_embeddings = self.lm.get_output_embeddings().weight.data

        input_embeddings_avg = input_embeddings[:-num_new_tokens].mean(
            dim=0, keepdim=True)
        output_embeddings_avg = output_embeddings[:-num_new_tokens].mean(
            dim=0, keepdim=True)

        input_embeddings[-num_new_tokens:] = input_embeddings_avg
        output_embeddings[-num_new_tokens:] = output_embeddings_avg

        self.model_config.token_at_ids = self.tokenizer.convert_tokens_to_ids([BOX_START])[0]
        self.freeze_llm = self.model_config.get("freeze_llm", True)
        logger.info(f'freeze_llm: {self.freeze_llm}')
        if self.freeze_llm:
            logger.info("freeze llm")
            freeze_module(self.lm)
    
        if self.model_config.llm.use_lora:
            self.use_lora = True
            from peft import get_peft_model, LoraConfig, TaskType
            logger.info("Use lora")
            if self.model_config.llm.name == 'internlm_20b':
                peft_config = LoraConfig(
                    task_type=TaskType.CAUSAL_LM, inference_mode=False, 
                    r=self.model_config.llm.lora_r, lora_alpha=self.model_config.llm.lora_alpha, lora_dropout=self.model_config.llm.lora_dropout,
                    target_modules=['wqkv', 'wo', 'w1', 'w2', 'w3', 'output']
                )
            else:
                peft_config = LoraConfig(
                    task_type=TaskType.CAUSAL_LM, inference_mode=False, 
                    r=self.model_config.llm.lora_r, lora_alpha=self.model_config.llm.lora_alpha, lora_dropout=self.model_config.llm.lora_dropout,
                    target_modules=["q_proj", "k_proj", "v_proj", "o_proj",
                                    "gate_proj", "up_proj", "down_proj", "lm_head"]
                )
                
            self.lm = get_peft_model(self.lm, peft_config)
            self.lm.enable_input_require_grads()
            self.lm.print_trainable_parameters()

            if self.model_config.get("freeze_lora", False):
                logger.info("freeze lora")
                freeze_module(self.lm)
                self.lm.print_trainable_parameters()

        else:
            self.use_lora = False

    def add_lora(self):
        if self.model_config.llm.use_lora:
            self.use_lora = True
            from peft import get_peft_model, LoraConfig, TaskType
            logger.info("Use lora")
            if self.model_config.llm.name == 'internlm_20b':
                peft_config = LoraConfig(
                    task_type=TaskType.CAUSAL_LM, inference_mode=False, 
                    r=self.model_config.llm.lora_r, lora_alpha=self.model_config.llm.lora_alpha, lora_dropout=self.model_config.llm.lora_dropout,
                    target_modules=['wqkv', 'wo', 'w1', 'w2', 'w3', 'output']
                )
            else:
                peft_config = LoraConfig(
                    task_type=TaskType.CAUSAL_LM, inference_mode=False, 
                    r=self.model_config.llm.lora_r, lora_alpha=self.model_config.llm.lora_alpha, lora_dropout=self.model_config.llm.lora_dropout,
                    target_modules=["q_proj", "k_proj", "v_proj", "o_proj",
                                    "gate_proj", "up_proj", "down_proj", "lm_head"]
                )
                
            self.lm = get_peft_model(self.lm, peft_config)
            self.lm.enable_input_require_grads()
            self.lm.print_trainable_parameters()

            if self.model_config.get("freeze_lora", False):
                logger.info("freeze lora")
                freeze_module(self.lm)
                self.lm.print_trainable_parameters()

        else:
            self.use_lora = False

    def add_tokens(self):
        num_new_tokens = len(special_tokens)
        self.lm.resize_token_embeddings(len(self.tokenizer))

        input_embeddings = self.lm.get_input_embeddings().weight.data
        output_embeddings = self.lm.get_output_embeddings().weight.data

        input_embeddings_avg = input_embeddings[:-num_new_tokens].mean(
            dim=0, keepdim=True)
        output_embeddings_avg = output_embeddings[:-num_new_tokens].mean(
            dim=0, keepdim=True)
        print(self.lm.get_input_embeddings().weight.data.shape)
        input_embeddings[-num_new_tokens:] = input_embeddings_avg
        output_embeddings[-num_new_tokens:] = output_embeddings_avg

        self.model_config.token_at_ids = self.tokenizer.convert_tokens_to_ids([BOX_START])[0]

    def build_loss(self):
        self.use_vision_regression_loss = self.model_config.loss.get("use_vision_regression_loss", False)
        self.use_bbox_loss = self.model_config.loss.get("add_bbox_loss", False)
        self.use_mask_loss = self.model_config.loss.get("use_mask_loss", False)
        self.use_temporal_loss = self.model_config.loss.get('use_temporal_loss', False)
        if self.use_vision_regression_loss:
            self.image_loss_fct = MSELoss()
        
        
    def load_pretrained_weights(self):
        if self.model_config.pretrained_paths.get('pretrained_vit_qformer_path', None):
            if 'safetensor' in self.model_config.pretrained_paths.pretrained_vit_qformer_path:
                from safetensors import safe_open
                from safetensors.torch import save_file
                state_dict = {}
                with safe_open(self.model_config.pretrained_paths.pretrained_vit_qformer_path, framework="pt", device="cpu") as f:
                    for key in f.keys():
                        state_dict[key] = f.get_tensor(key)
            else:
                state_dict = torch.load(self.model_config.pretrained_paths.pretrained_vit_qformer_path, map_location="cpu")
                if "model" in state_dict.keys():
                    state_dict = state_dict["model"]
                elif "module" in state_dict.keys(): 
                    state_dict = state_dict["module"] # for deepspeed
            self.check_temp_emb(state_dict)
            msg = self.load_state_dict(state_dict, strict=False)
            print('Loading vit: ', msg)
            logger.info(f"Load ViT and QFormer from {self.model_config.pretrained_paths.pretrained_vit_qformer_path}: {msg}")

        if self.model_config.pretrained_paths.get('pretrained_videochat2', None):
            state_dict = torch.load(self.model_config.pretrained_paths.pretrained_videochat2, map_location="cpu")
            
            new_state_dict = {}
            for k in state_dict.keys():
                if 'bert.embeddings' not in k:
                    new_state_dict[k] = state_dict[k]
            state_dict = new_state_dict
            # self.check_temp_emb(state_dict)
            msg = self.load_state_dict(state_dict, strict=False)
            print('Loading videochat2: ', msg)
        

    def check_temp_emb(self, state_dict):
        old_num_frames = self.model_config.vision_encoder.get('origin_num_frames', None)
        new_num_frames = self.model_config.vision_encoder.num_frames
        if old_num_frames is not None and old_num_frames != new_num_frames:
            logger.info(f"interpolate_pos_embed_internvideo2 to {new_num_frames} (origin_num_frames={old_num_frames})!!!")
            a = len(state_dict)
            interpolate_pos_embed_internvideo2_new(state_dict, self.vision_encoder, orig_t_size=4)
            assert a == len(state_dict), state_dict.keys()

    def build_bbox_decoder(self):
        self.loc_encoder = nn.Sequential(
            nn.Linear(4, self.model_config.llm.hidden_size // 2, dtype=torch.bfloat16),
            nn.ReLU(),
            nn.Linear(self.model_config.llm.hidden_size // 2, self.model_config.llm.hidden_size, dtype=torch.bfloat16),
        )

        self.loc_decoder = nn.Sequential(
            nn.Linear(self.model_config.llm.hidden_size, self.model_config.llm.hidden_size // 2, dtype=torch.bfloat16),
            nn.ReLU(),
            nn.Linear(self.model_config.llm.hidden_size // 2, 4, dtype=torch.bfloat16)
        )
        self._initialize_bbox_weights()

    def _initialize_bbox_weights(self):
        return

    def build_sam(self):
        sam2_checkpoint = "/cpfs01/user/heyinan/checkpoints/sam2_hiera_large.pt"
        model_cfg = "sam2_hiera_l.yaml"
        predictor = build_sam2_video_predictor(model_cfg, sam2_checkpoint, device=self.lm.device)
        
        self.sam = predictor
        freeze_module(self.sam)
        

    @property
    def dtype(self):
        return self.lm.dtype


    @property
    def device(self):
        return self.lm.device
