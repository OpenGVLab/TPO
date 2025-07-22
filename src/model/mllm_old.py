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
# from transformers import LlamaForCausalLM
from .llm.llama_xformer import LlamaForCausalLM

from petrel_client.client import Client
from torch.cuda.amp import autocast as autocast

from ..share_utils.constants import *
from ..share_utils.func_utils import freeze_module

from .vision_encoder import pretrain_internvideo2_giant_patch14_224_clean, build_vit
from .vision_encoder.pos_embed import interpolate_pos_embed_internvideo2_new
from .bridge import build_qformer, build_causal_qformer

from .base_model import BaseMLLM

logger = logging.getLogger(__name__)


class MultiModalLLM_PT_old(BaseMLLM):
    
    def __init__(
        self,
        config
    ):
        super().__init__(config=config)

    def forward(
        self,
        input_ids: torch.LongTensor = None,
        attention_mask: Optional[torch.Tensor] = None,
        labels: Optional[torch.LongTensor] = None,
        image: Optional[torch.Tensor] = None,
        video: Optional[torch.Tensor] = None,
        instruction = None,
        video_idx = None,
        image_idx = None,
    ):  
        # print('Model Forwarding')

        if self.use_vision_regression_loss:
            text_embeds, visual, visual_idx = self.pad_text_embeds(input_ids=input_ids, image=image,video=video, return_visual=True, video_idx=video_idx, image_idx=image_idx, instruction = instruction)
        else:
            text_embeds = self.pad_text_embeds(input_ids=input_ids, image=image, video=video, return_visual=False, video_idx=video_idx, image_idx=image_idx,  instruction = instruction)
        
        outputs = self.lm(
            inputs_embeds=text_embeds,
            attention_mask=attention_mask,
            labels=labels,
            output_hidden_states=True,
            return_dict=True,
        )

        if not self.use_vision_regression_loss:
            return outputs

        raise NotImplementedError
        loss_text = outputs.loss
        recover_image = outputs.hidden_states[-1][visual_idx]
        recover_image.view(-1, recover_image.shape[-1])
        project_up_image = self.project_down(recover_image)
        visual = visual / visual.norm(dim=-1, keepdim=True)
        loss_visual = self.image_loss_fct(visual, project_up_image)
        # print(loss_text, loss_visual)

        return loss_visual + loss_text, None
    
    def pad_text_embeds(
        self,
        input_ids: torch.LongTensor = None,
        image: Optional[torch.Tensor] = None,
        video: Optional[torch.Tensor] = None,
        image_idx = None,
        video_idx = None,
        return_visual: bool = False,
        instruction = None,
    ):
        # text_embeds
        text_embeds = self.lm.get_input_embeddings()(input_ids).clone()

        visual = None
        visual_idx = None
        # print('\033[31mtext_embeds.shape\033[0m', text_embeds.shape)
        if image is not None:
            # IMAGE = self.tokenizer.convert_tokens_to_ids([DEFAULT_IMAGE_TOKEN])[0]
            B, T, C, H, W = image.shape
            image = image.permute(0, 2, 1, 3, 4)
            # print('--- Encoding Image Here ---- ')
            prompt_image_embeds = self.encode_vision(image, instruction=instruction)
            # print('Prompt Image Embeds shape:', prompt_image_embeds.shape) # 116 * 64 * 1408
            # _, _, c = prompt_image_embeds.shape
            # prompt_image_embeds = prompt_image_embeds.view(-1, c)
            visual = prompt_image_embeds
            prompt_image_embeds = self.project_up(prompt_image_embeds)
            prompt_image_embeds = prompt_image_embeds.view(-1, prompt_image_embeds.shape[-1])
            # image_idx = (input_ids == IMAGE)
            visual_idx = image_idx
            # print('image:', torch.sum(torch.flatten(image_idx)))
            # new_text_embed = text_embeds.clone().detach()
            text_embeds[image_idx == 1] = prompt_image_embeds.to(text_embeds.device)
        elif video is not None:
            # VIDEO = self.tokenizer.convert_tokens_to_ids([DEFAULT_VIDEO_TOKEN])[0]
            
            if len(video.shape) == 5:
                B, T, C, H, W = video.shape
                N = 1
            else:
                B, N, T, C, H, W = video.shape
            video = video.reshape(B*N, T, C, H, W).permute(0, 2, 1, 3, 4)
            # print('--- Encoding Video Here ---- ')
            prompt_video_embeds = self.encode_vision(video, instruction=instruction)
            # print('Prompt Image Embeds shape:', prompt_video_embeds.shape) # 116 * 64 * 1408
            # _, _, c = prompt_video_embeds.shape
            # prompt_video_embeds = prompt_video_embeds.view(-1, c)
            visual = prompt_video_embeds
            prompt_video_embeds = self.project_up(prompt_video_embeds)
            prompt_video_embeds = prompt_video_embeds.view(-1, prompt_video_embeds.shape[-1])
            # video_idx = (input_ids == VIDEO)
            visual_idx = video_idx
            
            # print(text_embeds.shape, video_idx.shape, prompt_video_embeds.shape)
            # print(text_embeds[video_idx == 1].shape)
            # print(video_idx)
            # new_text_embed = text_embeds.clone().detach()
            text_embeds[video_idx == 1] = prompt_video_embeds.to(text_embeds.device).to(text_embeds.dtype)
        else:
            logger.warn(f"don't get visual input, input_ids: {input_ids}")
            
        if return_visual:
            return text_embeds, visual, visual_idx
        
        return text_embeds


    def encode_vision(
        self,
        image,
        instruction
    ):
        device = image.device
        B = image.shape[0]
        T = image.shape[2]
        use_image = True if T == 1 else False
        image_embeds = self.vision_encoder(image, use_image=use_image)
        C = image_embeds.shape[-1]
        image_embeds = image_embeds.reshape(B, -1, C)
        image_embeds = self.vision_layernorm(image_embeds).to(device)  # [B, T*L, C]
        
        image_atts = torch.ones(image_embeds.size()[:-1], dtype=torch.long).to(device)
        if self.extra_num_query_token > 0:
            query_tokens = torch.cat([self.query_tokens, self.extra_query_tokens], dim=1)
        query_tokens = query_tokens.expand(image_embeds.shape[0], -1, -1)
        if False:
            text_Qformer = self.qformer_tokenizer(
                instruction,
                padding='longest',
                truncation=True,
                max_length=512,
                return_tensors="pt",
            ).to(image_embeds.device)
            query_atts = torch.ones(query_tokens.size()[:-1], dtype=torch.long).to(image_embeds.device)
            Qformer_atts = torch.cat([query_atts, text_Qformer.attention_mask], dim=1)
            query_output = self.qformer.bert(
                text_Qformer.input_ids,
                attention_mask=Qformer_atts,
                query_embeds=query_tokens,
                encoder_hidden_states=image_embeds,
                encoder_attention_mask=image_atts,
                return_dict=True,
            )
        else:
            query_output = self.qformer.bert(
                query_embeds=query_tokens,
                encoder_hidden_states=image_embeds,
                encoder_attention_mask=image_atts,
                return_dict=True,
            )
        
        return query_output.last_hidden_state[:, :query_tokens.size(1), :]


    def generate_caption(
        self,
        input_ids,
        attention_mask,
        image_idx = None,
        video_idx = None,
        image: Optional[torch.Tensor] = None,
        video: Optional[torch.Tensor] = None,
        num_beams=1,
        max_new_tokens=200,
        do_sample=True,
        top_p=0.9,
        top_k=None,
        temperature=1.0,
        length_penalty=1,
        repetition_penalty=1.0,
    ):
        text_embeds = self.pad_text_embeds(input_ids=input_ids, image=image, video=video, image_idx=image_idx, video_idx=video_idx)
        outputs = self.lm.generate(
            inputs_embeds=text_embeds,
            attention_mask=attention_mask,
            num_beams=num_beams,
            max_new_tokens=max_new_tokens,
            do_sample=do_sample,
            min_length=1,
            top_p=top_p,
            top_k=top_k,
            temperature=temperature,
            length_penalty=length_penalty,
            repetition_penalty=repetition_penalty,
        )

        return outputs
