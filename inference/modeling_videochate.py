import io
import logging
import json
import torch
import torch.utils.checkpoint
from torch import nn
from torch.nn import MSELoss
from transformers.modeling_outputs import (
    CausalLMOutputWithPast,
)
from typing import List, Optional, Tuple, Union
from transformers import LlamaForCausalLM
from transformers.modeling_outputs import (
    CausalLMOutputWithPast,
)

from torch.cuda.amp import autocast as autocast
import torch.nn.functional as F

import numpy as np
from .modeling_vit import  build_vit, MLP, PostProcess

from .modeling_qformer import build_qformer
from .modeling_base import BaseMLLM

from transformers.modeling_outputs import BaseModelOutputWithPast, CausalLMOutputWithPast
logger = logging.getLogger(__name__)

import pycocotools.mask as mask_util

from .modeling_base import VID_TOKEN, IMG_TOKEN

class MultiModalLLM_PT(BaseMLLM):
    def __init__(
        self,
        config,
        _tokenizer=None
    ):
        super().__init__(config=config, _tokenizer=_tokenizer)
        self.use_clip = False
        self.num_frames = 16
        self.num_clips = 1
        self.token_merge_len = 4

        self.per_clip_frames = self.num_frames // self.num_clips
        
        print(self.config)
        self.merge_proj = nn.Linear(
            self.qformer.config.hidden_size*self.token_merge_len, self.config.hidden_size
        )

        if config.build_decoder:
            self.track_embed = MLP(self.config.hidden_size, self.config.hidden_size, 3 * 256, 2, dropout=0)
            self.track_embed_decode2 = MLP(4096, 4096, 4, 2, dropout=0)
            self.temporal_embed = MLP(self.config.hidden_size, self.config.hidden_size, 2, 2, dropout=0.3)
            self.action_embed = MLP(self.config.hidden_size, self.config.hidden_size, 1, 2, dropout=0.3)
            self.postprocess = PostProcess()
            self.track_token = nn.Parameter(torch.randn((1, 1, 4096)))
            self.temporal_token = nn.Parameter(torch.randn((1, 1, 4096)))
            self.box_token = nn.Parameter(torch.randn((1, 1, 4096)))


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
        output_boxes = None, # REC
        input_boxes = None, # tracking inputs
        text_input = None,
        video_info = None, 
        temporal_labels = None,
        gt_masks = None,
        sam_images = None,
        size_hw = None,
        path = None,
        mask_path = None,
        tvg_inputs = None,
        tvg_targets = None,
    ):  
        if text_input is not None:
            time_instructions = self.get_clip_time_instruct(text_input)
        else:
            time_instructions = None
        text_embeds = self.pad_text_embeds(input_ids=input_ids, image=image, video=video, return_visual=False,
                                        video_idx=video_idx, image_idx=image_idx,  instruction = instruction, 
                                        output_boxes = output_boxes, input_boxes=input_boxes, time_instructions = time_instructions)
        outputs = self.lm(
            inputs_embeds=text_embeds,
            attention_mask=attention_mask,
            labels=labels,
            output_hidden_states=True,
            return_dict=True,
        )
        loss = outputs.loss
        logger.info(f'llm loss:{loss}')

        if output_boxes is not None and self.use_bbox_loss:
            last_hidden_states = outputs.hidden_states[-1]
            pred_locs = []
            for idx in range(last_hidden_states.shape[0]):
                loc_positions = ( (input_ids[idx].flatten() == self.tokenizer.box_token) ).nonzero().flatten()
                selected_hidden_states = last_hidden_states[idx][loc_positions]
                pred_locs.append(self.loc_decoder(selected_hidden_states))
            box_loss = self.box_loss(pred_locs, output_boxes)
            logger.info(f'box loss:{box_loss}')
            loss += box_loss
        
        if (gt_masks is not None or input_boxes is not None) and self.use_mask_loss:
            last_hidden_states = outputs.hidden_states[-1]
            pred_masks = []
            sam_losses = []
            box_losses = []
            for idx in range(last_hidden_states.shape[0]):
                loc_positions = ( (input_ids[idx].flatten() == self.tokenizer.track_token) ).nonzero().flatten()
                selected_hidden_states = last_hidden_states[idx][loc_positions]
                embed_sam_boxes = self.track_embed(selected_hidden_states).reshape(1, 3, 256)
                inference_state = self.sam.init_state_images(sam_images, size_hw[idx][0], size_hw[idx][1])
                
                if input_boxes is not None:
                    gt_embeds = self.sam.get_prompt_embeding(inference_state, None, None, False, input_boxes[idx], device = text_embeds.device) 
                else:
                    input_boxes = self.find_boundaries_torch(gt_masks.squeeze(0)[:,:,:1].squeeze(2).cpu()).to(text_embeds.device)
                    gt_embeds = self.sam.get_prompt_embeding(inference_state, None, None, False, input_boxes, device = text_embeds.device) 
                pred_locs = [self.track_embed_decode2((selected_hidden_states))[0]]
                target_boxes = [input_boxes[idx]]

                src_boxes = pred_locs
                loss_bbox = self.box_loss2(src_boxes, target_boxes)

                loss_bbox = self.masked_loss(loss_bbox, 0)
                box_losses.append(loss_bbox)
                sam_losses.append( F.l1_loss(embed_sam_boxes, gt_embeds))
            
            logger.info(f'refering sam loss:{sam_losses}')
            sam_losses = torch.stack(sam_losses)
            box_losses = torch.stack(box_losses)
            loss += torch.mean(sam_losses)
            loss += torch.mean(box_losses)
        
        if tvg_inputs is not None and self.use_temporal_loss:
            last_hidden_states = outputs.hidden_states[-1]                                               # [bsz,1024, 4096]
            last_hidden_states = last_hidden_states.view(-1, last_hidden_states.size(-1))                # [bsz*1024, 4096]
            loc_positions = (input_ids.flatten()==self.tokenizer.temp_token).nonzero().flatten()         # [bsz]
            prompt_token = last_hidden_states[loc_positions]
            prompt_token = prompt_token.view(input_ids.shape[0], -1 ,prompt_token.shape[-1])   # [bsz, 1, 4096]

        
            cg_outputs = self.cg_model(**tvg_inputs, targets=tvg_targets, prompt_token=prompt_token)
            loss_dict = self.cg_criterion(cg_outputs, tvg_targets)
            weight_dict = self.cg_criterion.weight_dict
            tvg_loss = 0.05*sum(loss_dict[k] * weight_dict[k] for k in loss_dict.keys() if k in weight_dict)
            logger.info(f'tvg_loss:{tvg_loss}')
            loss += tvg_loss


        logger.info(f'all loss:{loss}')
        return CausalLMOutputWithPast(
            loss=loss,
            logits=outputs.logits,
            past_key_values=outputs.past_key_values,
            hidden_states=outputs.hidden_states,
            attentions=outputs.attentions,
        )

    def pad_text_embeds(
        self,
        input_ids: torch.LongTensor = None,
        image: Optional[torch.Tensor] = None,
        video: Optional[torch.Tensor] = None,
        image_idx = None,
        video_idx = None,
        return_visual: bool = False,
        instruction = None,
        output_boxes = None, # boxes for REC
        input_boxes = None, # boxes for tracking
        time_instructions = None,
    ):
        text_embeds = self.lm.get_input_embeddings()(input_ids.long()).detach()
        if input_boxes is not None:
            input_boxes = input_boxes[0].to(dtype=text_embeds.dtype)
            
            boxes_emb = self.loc_encoder(input_boxes)
            boxes_emb = boxes_emb.view(-1, 4096)
            
            text_embeds[input_ids == torch.full_like(input_ids, self.tokenizer.track_box_token)] = text_embeds[input_ids == torch.full_like(input_ids, self.tokenizer.track_box_token)] * 0 + boxes_emb.to(text_embeds.device)
            logger.info(f'embedings:{text_embeds[input_ids == torch.full_like(input_ids, self.tokenizer.track_box_token)].shape}')
        visual = None
        visual_idx = None
     
        if image is not None:

            B, T, C, H, W = image.shape
            image = image.permute(0, 2, 1, 3, 4)

            instruction = None
            
            prompt_image_embeds = self.encode_vision(image, instruction)

            visual = prompt_image_embeds

            prompt_image_embeds = self.project_up(prompt_image_embeds) # 768 -> 4096
            prompt_image_embeds = prompt_image_embeds.view(-1, prompt_image_embeds.shape[-1])

            visual_idx = image_idx

            prompt_image_embeds = prompt_image_embeds.to(dtype=text_embeds.dtype)

            text_embeds[image_idx == 1] = torch.zeros_like(text_embeds[image_idx == 1]) + prompt_image_embeds.to(text_embeds.device)


        elif video is not None:
            if len(video.shape) == 5:
                B, T, C, H, W = video.shape
                N = 1
                if self.use_clip:
                    video = video.reshape(B*self.num_clips, T//self.num_clips, C, H, W)  # [16, 8, 3, 224, 224]
            else:
                B, N, T, C, H, W = video.shape

            video = video.permute(0,2,1,3,4)      # 


            prompt_video_embeds = self.encode_vision(video, instruction=time_instructions)  # [2, 96, 768]
            if self.use_clip:
                prompt_video_embeds = prompt_video_embeds.reshape(B,-1,prompt_video_embeds.shape[-1])     # [2,8*96,768]
                batch_size, img_len, token_dim = prompt_video_embeds.shape
                prompt_video_embeds = prompt_video_embeds.view(batch_size, img_len // self.token_merge_len, self.token_merge_len * token_dim)  # [B, 768//4, 4*768] = [2, 192, 3072]
                prompt_video_embeds = self.merge_proj(prompt_video_embeds)   # [2, 192, 4096]
                prompt_video_embeds = prompt_video_embeds.view(-1, prompt_video_embeds.shape[-1]) # [2*192, 4096]
        
            else:
                prompt_video_embeds = self.project_up(prompt_video_embeds) # [2, 96, 4096]

            prompt_video_embeds = prompt_video_embeds.view(-1, prompt_video_embeds.shape[-1]) 
            visual_idx = video_idx
            
         
            text_embeds[video_idx == 1] = torch.zeros_like(text_embeds[video_idx == 1]) + prompt_video_embeds.to(text_embeds.device).to(text_embeds.dtype)
        
        else:
            logger.warn(f"don't get visual input, input_ids: {input_ids}")    
        

        for idx, text_embed in enumerate(text_embeds): 
            if text_embeds[idx][input_ids[idx].flatten() == self.tokenizer.box_token].shape[0] != 0:
                text_embeds[idx][input_ids[idx].flatten() == self.tokenizer.box_token] = torch.zeros_like(text_embeds[idx][input_ids[idx] == self.tokenizer.box_token]) + torch.cat([self.box_token.squeeze(0)] * (text_embeds[idx][input_ids[idx] == self.tokenizer.box_token]).shape[0]).to(text_embeds.dtype)
            if text_embeds[idx][input_ids[idx].flatten() == self.tokenizer.temp_token].shape[0] != 0:            
                text_embeds[idx][input_ids[idx].flatten() == self.tokenizer.temp_token] = torch.zeros_like(text_embeds[idx][input_ids[idx] == self.tokenizer.temp_token]) + self.temporal_token
            if text_embeds[idx][input_ids[idx].flatten() == self.tokenizer.track_token].shape[0] != 0:            
                text_embeds[idx][input_ids[idx].flatten() == self.tokenizer.track_token] = torch.zeros_like(text_embeds[idx][input_ids[idx] == self.tokenizer.track_token]) + self.track_token
             
        if return_visual:
            return text_embeds, visual, visual_idx
        
        return text_embeds

 
    
    def temporal_decode(self, temporal_embedding):
        pred_sted = self.temporal_embed(temporal_embedding)
        pred_actioness = self.action_embed(temporal_embedding)
        return pred_sted, pred_actioness

  
    def box_loss2(self, src_boxes, target_boxes):
        src_boxes = torch.cat(src_boxes)
        target_boxes = torch.cat(target_boxes)

        loss_bbox = F.l1_loss(src_boxes, target_boxes, reduction='none')
        loss_bbox = self.masked_loss(loss_bbox, 0)
        mask = (src_boxes[2:] >= src_boxes[:2]).all(-1)
        src_boxes = src_boxes[mask]
        target_boxes = target_boxes[mask]

        return loss_bbox 

    def box_loss(self, src_boxes, target_boxes):
        src_boxes = torch.cat(src_boxes)
        target_boxes = torch.cat(target_boxes)

        loss_bbox = F.l1_loss(src_boxes, target_boxes, reduction='none')
        loss_bbox = self.masked_loss(loss_bbox, 0)
        mask = (src_boxes[:, 2:] >= src_boxes[ :, :2]).all(-1)
        src_boxes = src_boxes[mask]
        target_boxes = target_boxes[mask]

        if src_boxes.shape[0] > 0:
            loss_giou = 1 - torch.diag(generalized_box_iou(
                src_boxes,
                target_boxes))
            loss_giou = self.masked_loss(loss_giou, 0)
        else:
            loss_giou = torch.tensor(2, dtype=src_boxes.dtype)
        iou, union = box_iou(src_boxes, target_boxes)
        
        return loss_bbox * 2 + loss_giou / 5
    
    def find_boundaries_torch(self, mask):

        from skimage.segmentation import find_boundaries
        mask_np = mask.to(torch.bool).numpy()
        boundaries = find_boundaries(mask_np, mode='outer')
        boundary_points = np.argwhere(boundaries)
        if boundary_points.size == 0:
            return torch.tensor([-1, -1, -1, -1], dtype = torch.bfloat16)
        h0, w0 = boundary_points.min(axis=0)
        h1, w1 = boundary_points.max(axis=0)
        return torch.tensor([w0 / mask.shape[1], h0 / mask.shape[0],  w1 / mask.shape[1], h1 / mask.shape[0]], dtype = torch.bfloat16)


    def sam_loss(self, sam_outputs, gt_masks):
        bound1 = self.find_boundaries_torch(gt_masks[:,:,:1].squeeze(2).cpu())
        bound2 = self.find_boundaries_torch(sam_outputs[:,:,:1].squeeze(2).cpu()) 

        lossl1 = F.l1_loss(bound1, bound2, reduction='none')
        lossl1 = self.masked_loss(lossl1, 0)

        loss_iou = self.iou_loss(sam_outputs, gt_masks)
        loss_dice = self.dice_loss(sam_outputs, gt_masks)

        # print(f'mask loss:{loss_iou,  loss_dice}')
        return loss_iou + loss_dice + lossl1
    
    def masked_loss(self, loss, n):
        mask = torch.ones_like(loss)
        # mask[-n:] = 1e-10
        loss = (loss*mask).sum()/(mask.sum())
        return loss

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
        if instruction is not None:
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

    def generate_caption_bbox(
        self,
        input_ids,
        attention_mask,
        labels,
        image_idx = None,
        video_idx = None,
        image: Optional[torch.Tensor] = None,
        video: Optional[torch.Tensor] = None,
        num_beams=1,
        max_new_tokens=200,
        do_sample=True,
        top_p=0.9,
        top_k=None,
        temperature=0.9,
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
        decoded_text = self.tokenizer.decode(outputs[0], skip_special_tokens=True)
        # torch.save({'text':decoded_text, 'output':{outputs}}, 'tmp.pth')
        # print(decoded_text)
        return outputs
    
    def generate_temporal(self,
        input_ids: torch.LongTensor = None,
        attention_mask: Optional[torch.Tensor] = None,
        labels: Optional[torch.LongTensor] = None,
        image: Optional[torch.Tensor] = None,
        video: Optional[torch.Tensor] = None,
        instruction = None,
        video_idx = None,
        image_idx = None,
        boxes = None,
        text_input = None,
        video_info = None, 
        temporal_labels = None):

        if text_input is not None:
            time_instructions = self.get_clip_time_instruct(text_input)
        else:
            time_instructions = None
        text_embeds = self.pad_text_embeds(input_ids=input_ids, image=image, video=video, return_visual=False,
                                        video_idx=video_idx, image_idx=image_idx,  instruction = instruction, 
                                        boxes = boxes, time_instructions = time_instructions)
        
        # TODO
        outputs = self.lm(
            inputs_embeds=text_embeds,
            attention_mask=attention_mask,
            labels=labels,
            output_hidden_states=True,
            return_dict=True,
        )

        if temporal_labels is not None:
            start_sec = temporal_labels["start_sec"]
            end_sec = temporal_labels["end_sec"]
            fps = video_info['fps']
            frame_indices = video_info['frame_indices']

            last_hidden_states = outputs.hidden_states[-1]                                   # [2,1024, 4096]
            last_hidden_states = last_hidden_states.view(-1, last_hidden_states.size(-1))    # [2048, 4096]
            loc_positions = (input_ids.flatten()==self.tokenizer.temp_place_ids).nonzero().flatten()   #
            selected_hidden_states = last_hidden_states[loc_positions]
            selected_hidden_states = selected_hidden_states.view(input_ids.shape[0], -1 ,selected_hidden_states.shape[-1]) # [2, 64, 4096]

            # just for debug
            
            # vis_embed = vis_embed[:,:64,:]

            pred_sted, pred_actionness = self.temporal_decode(selected_hidden_states) # [2,64,2]  [2,64,1]

            pred_sted = self.postprocess(pred_sted, frame_indices)
            pred_sec_s = pred_sted[0][0] / fps[0][0].item()
            pred_sec_e = pred_sted[0][1] / fps[0][0].item()

            output_file = "predictions2.jsonl"
            prediction = {"pred_sec_s": round(pred_sec_s, 1), "pred_sec_e": round(pred_sec_e, 1), "start_sec":float(start_sec[0]), "end_sec": float(end_sec[0])}

            with open(output_file, 'a') as f:
                json.dump(prediction, f)
                f.write('\n')

            return outputs

    def generate_seg(self, input_ids, attention_mask, labels, image, image_idx, video, video_idx, input_boxes, size_hw, sam_images):
        device = input_ids.device
        prompt = input_ids
        l_prompt = len(input_ids)
        temperature = 1e-5
        max_new_tokens = 20
        guide_w = 5
        stop_str = '</s>'
        bbox = []
        output_ids = list(input_ids[0])
        text_embeds = self.pad_text_embeds(input_ids=input_ids, image=image, video=video, image_idx=image_idx, video_idx=video_idx, return_visual=False, 
                                        instruction = None, output_boxes=None, input_boxes=input_boxes)
        for i in range(max_new_tokens):
            if i == 0:
                outputs = self.lm(
                        inputs_embeds=text_embeds,
                        attention_mask=attention_mask,
                        output_hidden_states=True,
                        return_dict=True,
                    )
                logits = outputs.logits
                past_key_values = outputs.past_key_values
            else:
                attention_mask = torch.ones(1, past_key_values[0][0].shape[-2] + 1, device=device)
                last_text_embeds = self.lm.get_input_embeddings()(torch.tensor(output_ids[-1], device=device).long()).detach().unsqueeze(0)
                last_text_embeds = last_text_embeds.unsqueeze(0)
                
                out = self.lm(
                    input_ids=None,
                    use_cache=True,
                    attention_mask=attention_mask,
                    output_hidden_states=True,
                    inputs_embeds=last_text_embeds,
                    past_key_values=past_key_values,
                )
                logits = out.logits
                past_key_values = out.past_key_values
            if logits is not None:
                last_token_logits = logits[0][-1]
                if temperature < 1e-4:
                    token = int(torch.argmax(last_token_logits))
                else:
                    probs = torch.softmax(last_token_logits / temperature, dim=-1)
                    token = int(torch.multinomial(probs, num_samples=1))
                output_ids.append(token)
            ret = self.tokenizer.decode(token)
            if ret == '<box_begin>':
                attention_mask = torch.ones(1, past_key_values[0][0].shape[-2] + 1, device=device)
                bbox_embeds = self.box_token.bfloat16()
                out = self.lm(
                    inputs_embeds=bbox_embeds,
                    use_cache=True,
                    attention_mask=attention_mask,
                    output_hidden_states=True,
                    past_key_values=past_key_values
                )
                last_hidden_states = out.hidden_states[-1]
                selected_hidden_states = last_hidden_states[0][0]
                bbox.append(self.loc_decoder(selected_hidden_states))
                last_token_logits = logits[0][-1]
                if temperature < 1e-4:
                    token = int(torch.argmax(last_token_logits))
                else:
                    probs = torch.softmax(last_token_logits / temperature, dim=-1)
                    token = int(torch.multinomial(probs, num_samples=1))
            if ret == '<track_begin>':
                attention_mask = torch.ones(1, past_key_values[0][0].shape[-2] + 1, device=device)
                tracking_embeds = self.track_token
                out = self.lm(
                    inputs_embeds=tracking_embeds,
                    use_cache=True,
                    attention_mask=attention_mask,
                    output_hidden_states=True,
                    past_key_values=past_key_values
                )
                last_hidden_states = out.hidden_states[-1]
                selected_hidden_states = last_hidden_states[0][0].to(dtype = torch.bfloat16)
                
                embed_sam_boxes = self.track_embed(selected_hidden_states).reshape(1, 3, 256)
                
                inference_state = self.sam.init_state_images(sam_images, size_hw[0][0], size_hw[0][1])
                gt_embeds = self.sam.get_prompt_embeding(inference_state, None, None, False, input_boxes[0].cuda(), device = text_embeds.device) 
                ann_frame_idx = 0
                ann_obj_id = 0
                box = np.array([0, 0, 0, 0], dtype=np.float32)
                _, out_obj_ids, out_mask_logits = self.sam.add_new_box_embeding(
                    inference_state=inference_state,
                    frame_idx=ann_frame_idx,
                    obj_id=ann_obj_id,
                    box=box,
                    box_embeding=embed_sam_boxes,
                )
                video_segments = {}  # video_segments contains the per-frame segmentation results
                for out_frame_idx, out_obj_ids, out_mask_logits in self.sam.propagate_in_video(inference_state):
                    video_segments[out_frame_idx] = {
                        out_obj_id: (out_mask_logits[i] > 0.0)
                        for i, out_obj_id in enumerate(out_obj_ids)
                    }
                video_segments = [video_segments[tt][0] for tt in video_segments]
                # bbox = model.find_boundaries_torch(video_segments[0].squeeze(0).cpu())
                # return ret, [], video_segments

            if (ret == '</s>'):
                break
        ret = self.tokenizer.decode(output_ids)
        del past_key_values
        return ret, bbox, video_segments
    
    def generate_answer(self, tokenizer, instruction, msg, user_prompt, media_type="video",video_tensor=None, image_tensor=None, answer_prompt=None, chat_history=[],return_history=False, debug=False, generation_config={}):
        input_ids, attention_masks, labels = [], [], []

        conversation = ""
        if instruction:
            conversation += instruction
        conversation += (
                    "[INST]" + " "
                )

        if media_type == 'image':
            conversation +=( "<Image>" + IMG_TOKEN + "</Image>")
        else:
            conversation += ("<Video>" + VID_TOKEN + "</Video>")

        conversation += ( msg.rstrip() + "[/INST]")

        for q,a in chat_history:
            conversation += (" [INST] " + q + " [/INST]")
            conversation += (a + "</s>")
            
        conversation += (" [INST] " + user_prompt + " [/INST]")
        conversation += ("")
        if answer_prompt:
            conversation += ("Best Option: (")
        total_len = 0
        indexs = []
        if debug:
            print(conversation)

        tokenized = tokenizer.build_input_ids([conversation], 
                                          max_length=1024, 
                                          add_special_tokens=True, 
                                          truncation=False, 
                                          padding=False, 
                                          return_tensors='pt', 
                                          image=image_tensor,
                                          video=video_tensor,
                                          require_video=True)
        if video_tensor is not None:
            generation_output = self.generate_caption(
                    tokenized['input_ids'].unsqueeze(0).to(self.device), 
                    tokenized['attention_mask'].unsqueeze(0).to(self.device), 
                    video_idx = tokenized['video_index'].unsqueeze(0),
                    video = video_tensor.unsqueeze(0).to(self.device,dtype=torch.bfloat16), 
                    do_sample=False
                    )
        elif image_tensor is not None:
            generation_output = self.generate_caption(
                    tokenized['input_ids'].unsqueeze(0).to(self.device), 
                    tokenized['attention_mask'].unsqueeze(0).to(self.device), 
                    image_idx = tokenized['image_index'].unsqueeze(0),
                    image = image_tensor.unsqueeze(0).to(self.device,dtype=torch.bfloat16), 
                    do_sample=False
                    )
        response = tokenizer.batch_decode(generation_output, skip_special_tokens=True)[0]
        if debug:
            print(response)
        return response, chat_history