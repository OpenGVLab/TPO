from utils.config import Config
config_file = "configs/config_1binternlm7b.json"
cfg = Config.from_file(config_file)
import os
import io
from petrel_client.client import Client
from decord import VideoReader, cpu
client = Client('~/petreloss.conf', enable_mc=False)
from models import VideoChat2_it_internlm
from utils.easydict import EasyDict
import torch

from transformers import StoppingCriteria, StoppingCriteriaList

from PIL import Image
import numpy as np
import numpy as np
from decord import VideoReader, cpu
import torchvision.transforms as T
from torchvision.transforms import PILToTensor
from torchvision import transforms
from dataset.video_transforms import (
    GroupNormalize, GroupScale, GroupCenterCrop, 
    Stack, ToTorchFormatTensor
)
from torch.utils.data import Dataset
from torchvision.transforms.functional import InterpolationMode

from torchvision import transforms

import matplotlib.pyplot as plt

from IPython.display import Video, HTML

from peft import get_peft_model, LoraConfig, TaskType
import copy

import json
from collections import OrderedDict

from tqdm import tqdm

import decord
decord.bridge.set_bridge("torch")

cfg.model.vision_encoder.num_frames = 16
model = VideoChat2_it_internlm(config=cfg.model)

peft_config = LoraConfig(
    task_type=TaskType.CAUSAL_LM, inference_mode=False, 
    r=16, lora_alpha=32, lora_dropout=0.,
    target_modules=['attention.wqkv', 'attention.wo', 'feed_forward.w1', 'feed_forward.w2', 'feed_forward.w3']
)
model.internlm_model = get_peft_model(model.internlm_model, peft_config)

sp = '/mnt/petrelfs/wangchenting/multimodalllm/1b_internlm7b_stage3_lora.pth'

new_state_dict = torch.load(sp, map_location='cpu')

state_dict = {}

for k in new_state_dict.keys():
    state_dict[k.replace('lm.', 'internlm_model.').replace('project_up', 'internlm_proj')] = new_state_dict[k]

from models.videochat_mistra.pos import interpolate_pos_embed_internvideo2_new
interpolate_pos_embed_internvideo2_new(state_dict, model.vision_encoder, orig_t_size=8)

msg = model.load_state_dict(state_dict, strict=False)
print(msg)

model.cuda()
model.eval()

def get_prompt(conv):
    if conv.system:
        ret = "<|im_start|>system\n" + conv.system + conv.sep
    else:
        ret = ""
    for role, message in conv.messages:
        if message:
            ret += role + message + conv.sep
        else:
            ret += role
    return ret


def get_prompt2(conv):
    if conv.system:
        ret = "<|im_start|>system\n" + conv.system + conv.sep
    else:
        ret = ""
    count = 0
    for role, message in conv.messages:
        count += 1
        if count == len(conv.messages):
            ret += role + message
        else:
            if message:
                ret += role + message + conv.sep
            else:
                ret += role
    return ret


def get_context_emb(conv, model, img_list, answer_prompt=None, print_res=False):
    if answer_prompt:
        prompt = get_prompt2(conv)
    else:
        prompt = get_prompt(conv)
    if print_res:
        print(prompt)
    if '<VideoHere>' in prompt:
        prompt_segs = prompt.split('<VideoHere>')
    else:
        prompt_segs = prompt.split('<ImageHere>')
    assert len(prompt_segs) == len(img_list) + 1, "Unmatched numbers of image placeholders and images."
    with torch.no_grad():
        seg_tokens = [
            model.internlm_tokenizer(
                seg, return_tensors="pt", add_special_tokens=i == 0).to("cuda:0").input_ids
            # only add bos to the first seg
            for i, seg in enumerate(prompt_segs)
        ]
        seg_embs = [model.internlm_model.get_input_embeddings()(seg_t) for seg_t in seg_tokens]
#         seg_embs = [model.mistral_model.model.embed_tokens(seg_t) for seg_t in seg_tokens]
    mixed_embs = [emb for pair in zip(seg_embs[:-1], img_list) for emb in pair] + [seg_embs[-1]]
    mixed_embs = torch.cat(mixed_embs, dim=1)
    return mixed_embs


def ask(text, conv):
    conv.messages.append([conv.roles[0], text])
        

class StoppingCriteriaSub(StoppingCriteria):
    def __init__(self, stops=[], encounters=1):
        super().__init__()
        self.stops = stops
    def __call__(self, input_ids: torch.LongTensor, scores: torch.FloatTensor):
        for stop in self.stops:
            if torch.all((stop == input_ids[0][-len(stop):])).item():
                return True
        return False
    
    
def answer(conv, model, img_list, do_sample=True, max_new_tokens=200, num_beams=1, min_length=1,
               repetition_penalty=1.0, length_penalty=1, temperature=1.0, answer_prompt=None, print_res=False):
    stop_words_ids = [
        torch.tensor([2]).to("cuda:0"),
        torch.tensor([92543]).to("cuda:0"),
        torch.tensor([92542]).to("cuda:0"),
    ]  
    stopping_criteria = StoppingCriteriaList([StoppingCriteriaSub(stops=stop_words_ids)])
    
    conv.messages.append([conv.roles[1], answer_prompt])
    embs = get_context_emb(conv, model, img_list, answer_prompt=answer_prompt, print_res=print_res)
    with torch.no_grad():
        print('Generating')
        outputs = model.internlm_model.generate(
            inputs_embeds=embs,
            max_new_tokens=max_new_tokens,
            stopping_criteria=stopping_criteria,
            num_beams=num_beams,
            do_sample=do_sample,
            min_length=min_length,
            repetition_penalty=repetition_penalty,
            length_penalty=length_penalty,
            temperature=temperature,
        )
        print('Finish Generating')
    output_token = outputs[0]
    if output_token[0] == 0:  # the model might output a unknow token <unk> at the beginning. remove it
            output_token = output_token[1:]
    if output_token[0] == 1:  # some users find that there is a start token <s> at the beginning. remove it
            output_token = output_token[1:]
    output_text = model.internlm_tokenizer.decode(output_token, add_special_tokens=False)
    output_text = output_text.split('<|im_end|>')[0]  # remove the stop sign </s>
    conv.messages[-1][1] = output_text + '<|im_end|>'
    return output_text, output_token.cpu().numpy()

from dataset.hd_utils import HD_transform_padding, HD_transform_no_padding

def get_index(num_frames, num_segments):
    seg_size = float(num_frames - 1) / num_segments
    start = int(seg_size / 2)
    offsets = np.array([
        start + int(np.round(seg_size * idx)) for idx in range(num_segments)
    ])
    return offsets


def load_video(video_path, num_segments=8, return_msg=False, resolution=224, hd_num=4, padding=False):
    if "s3://" in video_path:
        video_bytes = client.get(video_path)
        vr = VideoReader(io.BytesIO(video_bytes), ctx=cpu(0), num_threads=1)
    else:
        vr = VideoReader(video_path, ctx=cpu(0), num_threads=1)
    num_frames = len(vr)
    frame_indices = get_index(num_frames, num_segments)

    mean = (0.485, 0.456, 0.406)
    std = (0.229, 0.224, 0.225)

    transform = transforms.Compose([
        transforms.Lambda(lambda x: x.float().div(255.0)),
        transforms.Normalize(mean, std)
    ])

    frames = vr.get_batch(frame_indices)
    frames = frames.permute(0, 3, 1, 2)

    if padding:
        frames = HD_transform_padding(frames.float(), image_size=resolution, hd_num=hd_num)
    else:
        frames = HD_transform_no_padding(frames.float(), image_size=resolution, hd_num=hd_num)

    frames = transform(frames)
    print(frames.shape)
    
    if return_msg:
        fps = float(vr.get_avg_fps())
        sec = ", ".join([str(round(f / fps, 1)) for f in frame_indices])
        # " " should be added in the start and end
        msg = f"The video contains {len(frame_indices)} frames sampled at {sec} seconds."
        return frames, msg
    else:
        return frames
 
    
import pysubs2
import re

def clean_text(text):
    cleaned_text = re.sub(r'[^A-Za-z0-9\s]\[\]', '', text)
    return cleaned_text

def read_vtt_and_concatenate(file_path, tokenizer, max_len=4096):
    subs = pysubs2.load(file_path, encoding="utf-8")
        
    prev = ""
    subtitles = []
    for caption in subs:
        # Split the caption text into individual lines
        lines = caption.text.split('\n')
        for line in lines:
            # Clean the text and check for repetition
            line = clean_text(line)
            if prev != line and line:
                subtitles.append(line)
                prev = line

    # Join subtitles to check length
    full_text = ' '.join(subtitles)
    tokenized_ids = tokenizer(full_text, add_special_tokens=False).input_ids

    # If the tokenized length is within the limit, return the full text
    if len(tokenized_ids) <= max_len:
        return full_text

    # Otherwise, we need to trim the text to fit within the limit
    # We will keep the first half and the last half
    half_len = max_len // 2
    start_text = ' '.join(subtitles[:half_len])
    end_text = ' '.join(subtitles[-half_len:])
    
    # Re-tokenize to ensure the total length is within the limit
    start_tokenized_ids = tokenizer(start_text, add_special_tokens=False).input_ids
    end_tokenized_ids = tokenizer(end_text, add_special_tokens=False).input_ids

    # Adjust the lengths to fit within the max_len
    while len(start_tokenized_ids) + len(end_tokenized_ids) > max_len:
        if len(start_tokenized_ids) > len(end_tokenized_ids):
            start_tokenized_ids.pop()
        else:
            end_tokenized_ids.pop(0)
    
    # Combine the adjusted parts
    adjusted_text = tokenizer.decode(start_tokenized_ids) + ' ... ' + tokenizer.decode(end_tokenized_ids)
    
    return adjusted_text


class MME_dataset(Dataset):
    def __init__(
        self, 
        data_prefix="shdd:s3://VideoMME_0629/processed_1fps",
        subtitle_prefix="/mnt/petrelfs/share_data/likunchang/videomme/subtitle_0629",
        anno_path="/mnt/petrelfs/share_data/likunchang/videomme/Video-MME_0629.json",
        frame_dict_path="./demo/ablation_long/video_mme_1fps.json",
        num_segments=16, 
        stride=0, # if stride >= 1, will return all frames according to FPS (1/stride), else return partial frames
        resolution=224, 
        hd_num=6, 
        max_subtitle_len=4096, # max_tokens for subtitle
    ):
        self.data_prefix = data_prefix
        self.subtitle_prefix = subtitle_prefix
        with open(anno_path, 'r') as f:
            self.data_list = json.load(f)
        with open(frame_dict_path, 'r') as f:
            self.frame_dict = json.load(f)
        
        self.hd_num = hd_num
        self.num_segments = num_segments
        self.stride = stride
        self.resolution = resolution
        self.max_subtitle_len = max_subtitle_len

        # transform
        mean = (0.485, 0.456, 0.406)
        std = (0.229, 0.224, 0.225)
        self.transform = transforms.Compose([
            transforms.Lambda(lambda x: x.float().div(255.0)),
            transforms.Normalize(mean, std)
        ])
        self.hd_transform = HD_transform_no_padding
    
    def __str__(self):
        task_dict = {}
        total = 0
        for data in self.data_list:
            if data['duration_category'] not in ans_dict:
                task_dict[data['duration_category']] = {}
            for q in data['questions']:
                if q['task_type'] not in ans_dict[data['duration_category']]:
                    ans_dict[data['duration_category']][q['task_type']] = 0
                ans_dict[data['duration_category']][q['task_type']] += 1
                total += 1

        res = f"There are {len(self.data_list)} videos.\n"
        res += f"There are {total} QAs.\n"
        for k, v in task_dict.items():
            res += f"------{k}------\n"
            for kk, vv in task_dict.items():
                res += f"{kk}: {vv}\n"
                
        return res.rstrip()
        
    def __len__(self):
        return len(self.data_list)
    
    def get_index(self, max_frame):
        start_idx = 0
        end_idx = max_frame - 1
        seg_size = float(max_frame - start_idx) / self.num_segments
        frame_indices = np.array([
            max(int(start_idx + (seg_size / 2) + np.round(seg_size * idx)), end_idx)
            for idx in range(self.num_segments)
        ])
        return frame_indices

    def read_frame(self, video_name):
        full_frame_list = []
        for p in self.frame_dict[video_name]:
            full_frame_list.append(os.path.join(self.data_prefix, video_name, 'frames', p))
            
        images_group = list()
        if self.stride >= 1 and (len(full_frame_list) / self.stride) > self.num_segments:
            frame_list = full_frame_list[::self.stride]
        else:
            # if len(full_frame_list) < self.num_segments: # return all frames if not, seem to be a little lower
            #     frame_list = full_frame_list
            # else:
            frame_indices = get_index(len(full_frame_list), self.num_segments)
            frame_list = [full_frame_list[idx] for idx in frame_indices]
        # print(frame_list)
        
        for frame_path in frame_list:
            if "s3://" in frame_path:
                img_bytes = client.get(frame_path)
                img = Image.open(io.BytesIO(img_bytes))
            else:
                img = Image.open(frame_path)
            img = PILToTensor()(img).unsqueeze(0)
            img = self.hd_transform(img.float(), image_size=self.resolution, hd_num=self.hd_num)
            images_group.append(img)
        torch_imgs = self.transform(torch.vstack(images_group))
        print(torch_imgs.shape)
        return torch_imgs

    def qa_template(self, data):
        question = f"Question: {data['question']}\n"
        question += "Options:\n"
        answer = data['answer']
        answer = f"({answer}) {data['options'][ord(answer) - ord('A')][3:]}"
        for idx, c in enumerate(data['options']):
            cur_choice, cur_text = c[0], c[3:]
            question += f"({cur_choice}) {cur_text}\n"
        question = question.rstrip()
        return question, answer

    def __getitem__(self, idx):
        video_name = self.data_list[idx]['videoID']
        torch_imgs = self.read_frame(video_name)
        duration_category = self.data_list[idx]['duration']
        qa_list = []
        for qa in self.data_list[idx]['questions']:
            qa_list.append(self.qa_template(qa))

        subtitle = ""
        try:
            subtitle_path = os.path.join(self.subtitle_prefix, video_name + ".srt")
            if os.path.exists(subtitle_path):
                subtitle = read_vtt_and_concatenate(subtitle_path, model.internlm_tokenizer, self.max_subtitle_len)
        except Exception:
            subtitle = ""
            print(f"Error for {subtitle_path}")
            
        return {
            'subtitle': subtitle,
            'video': torch_imgs, 
            'qa_list': qa_list,
            'duration_category': duration_category
        }


import math

def infer_mme_long(
        data_sample, system="", 
        question_prompt='', # add in the end of question
        answer_prompt=None, # add in the begining of answer
        return_prompt='',  # add in the begining of return message
        system_q=False, # whether add question in the system prompt for QFormer
        print_res=True,
        system_llm=False,
        no_qformer_instruction=False,
        qformer_instruction=None,
        add_subtitle=False,
        only_global=False,
        window_size=16, # if window_size > 1, add window
        prompt_version="all",
        merge_num=8,
        split_num=1, # split the video for video_encoder
    ):
    assert system_q == False, "do not support system_q now"
    video = data_sample["video"]
    T_, C, H, W = video.shape
    merge_flag = False

    add = window_size - T_ % window_size
    if add != window_size:
        ## Method1: directly repeat the final frame
        # video = torch.cat([video, video[-1:, :, :, :].repeat(add, 1, 1, 1)], dim=0)
        ## Method2: repeat the middle frame
        video_list = []
        seg_size = T_ // add
        for i in range(add):
            start = i * seg_size
            end = T_ if i == (add - 1) else (i + 1) * seg_size
            video_list.extend(video[start:end])
            video_list.extend(video[end - 1].unsqueeze(0))
        video = torch.cat(video_list, dim=0)
        T_ += add
        print(f"Total Frame: {T_ - add} => {T_}")
    video = video.reshape(T_ // window_size, window_size, C, H, W).to("cuda:0")
    
    video_list = []
    batch_size = 1 if only_global else 1
    num_batches = math.ceil((T_ // window_size) / batch_size)
    video_emb_list = []
    for i in range(num_batches):
        start_idx = i * batch_size
        end_idx = min((i + 1) * batch_size, T_ // window_size)
        batch_video = video[start_idx:end_idx]
        
        with torch.no_grad():
            if no_qformer_instruction:
                batch_video_emb, _, _ = model.encode_img(batch_video, [""] * len(batch_video), only_global=only_global, split_num=split_num)
            elif qformer_instruction is not None:
                print(qformer_instruction)
                batch_video_emb, _, _ = model.encode_img(batch_video, [qformer_instruction] * len(batch_video), only_global=only_global, split_num=split_num)
            elif system_q:
                raise NotImplementedError
            else:
                batch_video_emb, _, _ = model.encode_img(batch_video, [system] * len(batch_video), only_global=only_global, split_num=split_num)

        video_emb_list.extend(batch_video_emb)
        torch.cuda.empty_cache()

    if prompt_version == "all":
        _, _, C = video_emb_list[0].shape
        video_list.append(
            torch.cat(video_emb_list, dim=0).view(1, -1, C)
        )
    elif "split" in prompt_version:
        _, _, C = video_emb_list[0].shape
        video_emb = torch.cat(video_emb_list, dim=0).view(T_ // window_size, -1, C)
        for v in range(T_ // window_size):
            video_list.append(video_emb[v].unsqueeze(0))
    elif prompt_version in ["mergeByIter", "dynmaicMergeByIter"]:
        _, N, C = video_emb_list[0].shape
        video_emb = torch.cat(video_emb_list, dim=0).view(-1, N, C)
        tmp_emb_list = [v for v in video_emb]
        print(f"Num: {len(tmp_emb_list)}, Shape: {tmp_emb_list[0].shape}")
        if len(tmp_emb_list) > merge_num:
            merge_flag = True

        # calculate similarity
        similar_list = []
        for idx in range(len(tmp_emb_list) -1):
            scores = tmp_emb_list[idx] @ tmp_emb_list[idx+1].transpose(-1, -2)
            frame_silimar = torch.mean(scores)
            similar_list.append(frame_silimar)
        # iteratively compress the queries
        while len(tmp_emb_list) > merge_num:
            max_value = max(similar_list)
            max_index = similar_list.index(max_value)
            new_frame_feature = (tmp_emb_list[max_index].cpu() + tmp_emb_list[max_index+1].cpu()) / 2
            tmp_emb_list[max_index] = new_frame_feature.cuda()
            del(tmp_emb_list[max_index+1])
            del(similar_list[max_index])
            if max_index > 0:
                similar_list[max_index - 1] = torch.mean(tmp_emb_list[max_index - 1] @ tmp_emb_list[max_index].transpose(-1, -2))
            if max_index < len(tmp_emb_list) - 1:
                similar_list[max_index] = torch.mean(tmp_emb_list[max_index] @ tmp_emb_list[max_index + 1].transpose(-1, -2))

        if prompt_version == "dynmaicMergeByIter" and merge_flag:
            for tmp in tmp_emb_list:
                video_list.append(tmp.unsqueeze(0))
        else:
            video_list.append(
                torch.cat(tmp_emb_list, dim=0).view(1, -1, C).to(torch.bfloat16)
            )
    elif prompt_version == "mergeByDiv":
        _, N, C = video_emb_list[0].shape
        video_emb = torch.cat(video_emb_list, dim=0).view(-1, N, C)
        if len(video_emb) <= merge_num:
            video_list.append(
                video_emb.view(1, -1, C)
            )
        else:
            tmp_emb_list = []
            div_size = len(video_emb) // merge_num
            for idx in range(merge_num):
                start = div_size * idx
                end = len(video_emb) if idx == (merge_num - 1) else div_size * (idx + 1)
                tmp_emb = video_emb[start:end] # (seg_size, N, C)
                if len(tmp_emb) == 1:
                    tmp_emb_list.append(tmp_emb)
                else:
                    pool_emb = tmp_emb.mean(dim=0, keepdim=True) # (1, N, C)
                    attn = pool_emb.permute(1, 0, 2) @ tmp_emb.permute(1, 2, 0) # (N, 1, seg_size)
                    attn = attn / (tmp_emb.shape[-1] ** 0.5)
                    final_emb = (attn.softmax(-1) @ tmp_emb.permute(1, 0, 2)).permute(1, 0, 2)
                    tmp_emb_list.append(final_emb)
            video_list.append(
                torch.cat(tmp_emb_list, dim=0).view(1, -1, C)
            )
    else:
        raise NotImplementedError
    print(video_list[0].shape)
    torch.cuda.empty_cache()

    pred_list = []
    gt_list = []
    for idx, qa in enumerate(data_sample['qa_list']):
        print(f"----------qa_{idx}---------", flush=True)
        chat = EasyDict({
            "system": system,
            "roles": ("<|im_start|>user\n", "<|im_start|>assistant\n"),
            "messages": [],
            "sep": "<|im_end|>\n"
        })

        chat_msg = ""
        if add_subtitle and data_sample['subtitle'] != '':
            subtitle = f"This video's subtitles are listed below: {data_sample['subtitle']}"
            chat_msg = f"{subtitle}\n"
        if prompt_version in ["all", "mergeByIter", "mergeByDiv"]:
            chat_msg += "<vid><VideoHere></vid> [/INST]"
        else:
            raise NotImplementedError
        
        chat.messages.append([chat.roles[0], chat_msg])
        
        if system_llm:
            prompt = system + qa[0] + question_prompt
        else:
            prompt = qa[0] + question_prompt
        
        ask(prompt, chat)
    
        llm_message = answer(
            conv=chat, model=model, do_sample=False, 
            img_list=video_list, max_new_tokens=100, 
            answer_prompt=answer_prompt, print_res=print_res
        )[0]
        # remove potential explanation
        llm_message = return_prompt + llm_message.strip().split('\n')[0]
        print(f"Pred: {llm_message}", flush=True)
        print(f"GT: {qa[1]}", flush=True)
        pred_list.append(llm_message[1])
        gt_list.append(qa[1][1])
    return pred_list, gt_list

save_path = "videomme_f16_internlm_true"

num_frame = 16
num_segments = 128
stride = 4 # 1/4 FPS
resolution = 336
hd_num = 6
max_subtitle_len = 8192
add_subtitle = False # no subtitle
no_qformer_instruction = True
qformer_instruction = None
only_global = True # only input global
window_size = 16
prompt_version = "mergeByIter"
merge_num = 10 # 10 x 96
split_num = 1
split_mode = None # if dynamic, do not split for long input

model.local_size = resolution

data_prefix = "shdd:s3://VideoMME_0629/processed_1fps"
anno_path = "/mnt/petrelfs/share_data/likunchang/videomme/Video-MME_0629.json"
frame_dict_path = "/mnt/petrelfs/wangchenting/fg-videochat/video_mme_1fps.json"
dataset = MME_dataset(
    data_prefix=data_prefix, 
    anno_path=anno_path, 
    frame_dict_path=frame_dict_path,
    num_segments=num_segments, 
    stride=stride,
    resolution=resolution,
    hd_num=hd_num,
    max_subtitle_len=max_subtitle_len,
)

with open(anno_path, 'r') as f:
    res_json_data = json.load(f)

correct = 0
total = 0
res_list = []
acc_dict = {}

for idx, example in enumerate(tqdm(dataset)):
# for idx in tqdm(range(len(dataset))[300:]):
#     example = dataset[idx]
    duration_category = example['duration_category']
    if duration_category not in acc_dict:
        acc_dict[duration_category] = [0, 0] # correct, total
    qa_count = len(example['qa_list'])
    acc_dict[duration_category][1] += qa_count
    total += qa_count
    
    cur_split_num = split_num
    
    pred_list, gt_list = infer_mme_long(
        example, 
#         "Carefully observe the video and choose the best option for the question. ", 
#         "Carefully watch the video and pay attention to the cause, sequence of events, and object details and movements. Based on your observations, select the best option that accurately addresses the question. ",  # newPrompt
#         "Carefully watch the video and pay attention to the cause and sequence of events, the detail and movement of objects, and the action and pose of persons. Based on your observations, select the best option that accurately addresses the question. ", # newPrompt2
        "Carefully watch the video and pay attention to the cause and sequence of events, the detail and movement of objects, and the action and pose of persons. Based on your observations, select the best option that accurately addresses the question.\n", # newPrompt2
        # "Carefully watch the video, read the related subtitles and pay attention to the cause and sequence of events, the detail and movement of objects, and the action and pose of persons. Based on your observations, select the best option that accurately addresses the question.\n", # newPrompt2
#         question_prompt="\nOnly give the best option without any explanation.",
#         question_prompt="\nThink it step by step. Only give the best option without any explanation.", # prompt2
        question_prompt="\nOnly give the best option.",  # prompt3
        answer_prompt="Best option:(",
        return_prompt='(',
        system_q=False,
        print_res=False,
        system_llm=True,
        no_qformer_instruction=no_qformer_instruction,
        qformer_instruction=qformer_instruction,
        add_subtitle=add_subtitle,
        only_global=only_global,
        window_size=window_size,
        prompt_version=prompt_version,
        merge_num=merge_num,
        split_num=cur_split_num,
    )
    torch.cuda.empty_cache()
    res_list.append({
        'pred': pred_list,
        'gt': gt_list
    })
    qa_idx = 0
    for pred, gt in zip(pred_list, gt_list):
        if pred == gt:
            acc_dict[duration_category][0] += 1
            correct += 1
        res_json_data[idx]['questions'][qa_idx]['response'] = pred
        qa_idx += 1
    print(f"Part  Acc: {acc_dict[duration_category][0] / acc_dict[duration_category][1] * 100 :.2f}%")
    print(f"Total Acc: {correct / total * 100 :.2f}%")
    print('-' * 50, duration_category, '-' * 50)

with open(f"{save_path}.json", "w") as f:
    json.dump({
        "acc_dict": acc_dict,
        "res_list": res_list
    }, f)

with open(f"{save_path}_full.json", "w") as f:
    json.dump(res_json_data, f)