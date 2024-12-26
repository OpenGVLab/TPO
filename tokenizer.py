import torch.nn as nn
import torch
import os
from typing import Any, Dict, List, Optional, Union
from transformers import LlamaTokenizer

from PIL import Image
from torchvision import transforms
import torch.nn.functional as F
from inference.modeling_special_token import special_tokens,DEFAULT_IMAGE_TOKEN, DEFAULT_VIDEO_TOKEN, IMG_TOKEN, VID_TOKEN, BOX_START, TIME_START, TIME_PLACEHOLDER, BOXES_PLACEHOLDER, TRACK_PLACEHOLDER, TRACK_START, TRACK_START_BOX
import logging
logger = logging.getLogger(__name__)
class MultimodalLlamaTokenizer(LlamaTokenizer):
    def __init__(
        self,
        vocab_file,
        unk_token="<unk>",
        bos_token="<s>",
        eos_token="</s>",
        n_query=64,
        v_query=64,
        pad_token=None,
        sp_model_kwargs: Optional[Dict[str, Any]] = None,
        add_bos_token=True,
        add_eos_token=False,
        clean_up_tokenization_spaces=False,
        special_tokens: Optional[List[str]] = None,
        device='cuda',
        **kwargs
    ):
        super().__init__(vocab_file, unk_token, bos_token, eos_token, pad_token, sp_model_kwargs, add_bos_token, add_eos_token,
                         clean_up_tokenization_spaces, **kwargs)
        
        self.device = device
        self.pad_token = self.unk_token
        self.special_tokens = special_tokens
        
        if not self.pad_token:
            self.pad_token = self.eos_token
        # follow EMU
        # self.image_placeholder = DEFAULT_IMG_TOKEN + DEFAULT_IMAGE_TOKEN * n_query + DEFAULT_IMG_END_TOKEN
        # self.video_placeholder = DEFAULT_IMG_TOKEN + DEFAULT_VIDEO_TOKEN * v_query + DEFAULT_IMG_END_TOKEN
        
        # For mistral
        # Define the special tokens
        # special_tokens = [DEFAULT_IMAGE_TOKEN, DEFAULT_VIDEO_TOKEN]
        # Add the special tokens to the tokenizer
        # self.add_tokens(special_tokens)
        if special_tokens is not None:
            self.add_tokens(special_tokens)
        self.box_start_token = self.convert_tokens_to_ids([BOX_START])[0]
        self.time_start_token = self.convert_tokens_to_ids([TIME_START])[0]
        self.temp_token = self.convert_tokens_to_ids([TIME_PLACEHOLDER])[0]
        self.box_token = self.convert_tokens_to_ids([BOXES_PLACEHOLDER])[0]
        self.track_box_token = self.convert_tokens_to_ids([TRACK_START_BOX])[0]
        self.track_token = self.convert_tokens_to_ids([TRACK_PLACEHOLDER])[0]
        self.track_start_token = self.convert_tokens_to_ids([TRACK_START])[0]
        logger.info(f'tokenizer:{self.box_start_token, self.time_start_token, self.temp_token, self.box_token,self.track_token }')
        # self.box_place_ids = self.build_input_ids(
        #             text=['<boxes>'],
        #             max_length=1000000,
        #             add_special_tokens=False,
        #             truncation=False,
        #             padding=False,
        #             return_tensors='pt',
        #         )

        self.n_query = n_query
        self.v_query = v_query
        
    @property
    def processor(self):
        self._processor = None
        return self._processor


    @property
    def num_image_tokens(self):
        return 8192  # self.image_tokenizer.num_tokens # allow not load


    def to(self, device):
        self.device = device
        if hasattr(self, '_image_tokenizer'):
            self._image_tokenizer.to(device=device)


    def encode_image(
        self,
        image,
        image_size: int = 224,
    ):
        # image = self.processor(image)
        return image


    def decode_image(
        self
    ):
        return ...


    def prepare_image_input(self, images):
        # image_size: int = 224
        # images = [self.encode_image(image, image_size) for image in images]
        # return torch.stack(images, 0)
        return images


    def prepare_text_input(
        self,
        text: List[str],
        max_length,
        add_special_tokens,
        truncation,
        padding = "longest", 
        return_tensors = "pt",
        image_placeholder: str = IMG_TOKEN,
        video_placeholder: str = VID_TOKEN,
    ):
        text = text[0]
        start = 0
        total_len = 0
        
        input_ids = []
        attention_mask = []
        indexs = []
        
        while True:
            index1 = text.find(image_placeholder, start)
            index2 = text.find(video_placeholder, start)

            if index1 == -1 and index2 == -1:
                index = -1
            elif index1 == -1:
                index = index2
            elif index2 == -1:
                index = index1
            else:
                index = min(index1, index2)
                assert index != -1
            
            # print(start, index, text, len(text))
            
            if index == -1:
                inputs = self(text[start:], max_length=max_length-total_len, add_special_tokens=add_special_tokens, truncation=truncation, padding=padding, return_tensors=return_tensors)
            else:
                inputs = self(text[start:index], max_length=max_length, add_special_tokens=add_special_tokens, truncation=truncation, padding='longest', return_tensors=return_tensors)
            
            # print(input_ids)
            input_ids += inputs.input_ids
            attention_mask += inputs.attention_mask
            indexs += torch.zeros_like(inputs.input_ids)
            total_len += inputs.input_ids[0].shape[0]
            
            if index != -1:
                input_ids += [torch.zeros(self.n_query).long()]
                attention_mask += [torch.ones(self.n_query).long()]
                indexs += [torch.ones(self.n_query)]
            
            if index == -1:
                ret = {
                    'input_ids': torch.cat(input_ids).long(),
                    'attention_mask': torch.cat(attention_mask).long(),
                    'index': torch.cat(indexs).to(torch.bool),
                }
                # print (ret)
                return ret
            start = index + len(IMG_TOKEN)


    def build_input_ids(
        self,
        text: List[str],
        max_length,
        add_special_tokens,
        truncation,
        padding,
        return_tensors,
        image = None,
        video = None,
        require_image = False,
        require_video = False,
    ):
        if image is not None:
            image = self.prepare_image_input(image)
        if video is not None:
            video = self.prepare_image_input(video)

        add_special_tokens = self.special_tokens
        inputs = self.prepare_text_input(text, max_length, add_special_tokens, truncation, padding, return_tensors)
        input_ids = inputs['input_ids']
        attention_mask = inputs['attention_mask']
        # print('index',inputs['index'])
        # logger.info(f'input_idx:{inputs["index"]}')
        return {
            'input_ids': input_ids,
            'attention_mask': attention_mask,
            'image_index': inputs['index'] if image is not None or require_image else None,
            'video_index': inputs['index'] if video is not None or require_video else None,
            'image': image if image is not None else None,
            'video': video if video is not None else None,
        }

