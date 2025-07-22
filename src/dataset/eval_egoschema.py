from utils.config import Config
config_file = "configs/config_1bmistral.json"
cfg = Config.from_file(config_file)

import io
import os
from models import VideoChat2_it_mistral
from utils.easydict import EasyDict
import torch

from transformers import StoppingCriteria, StoppingCriteriaList
import time
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
model = VideoChat2_it_mistral(config=cfg.model)

# add lora to run stage3 model
peft_config = LoraConfig(
    task_type=TaskType.CAUSAL_LM, inference_mode=False, 
    r=16, lora_alpha=32, lora_dropout=0.,
    target_modules=[
        "q_proj", "k_proj", "v_proj", "o_proj",
         "gate_proj", "up_proj", "down_proj", "lm_head"
    ]
)
model.mistral_model = get_peft_model(model.mistral_model, peft_config)

sp = '/mnt/petrelfs/wangchenting/multimodalllm/logs/scripts/pt/1b_qformer_mistral/stage2_8f.sh_20240614_153730/20000-model.safetensors'
# import torch
from safetensors import safe_open
from safetensors.torch import save_file

tensors = {}
with safe_open(sp, framework="pt", device="cpu") as f:
    for key in f.keys():
        if 'bert.embeddings' not in key:
            tensors[key.replace('lm.', 'mistral_model.').replace('project_up', 'mistral_proj')] = f.get_tensor(key)

if 'module' in tensors.keys():
    tensors = tensors['module']

from models.videochat_mistra.pos import interpolate_pos_embed_internvideo2_new
interpolate_pos_embed_internvideo2_new(tensors, model.vision_encoder, orig_t_size=8)

msg = model.load_state_dict(tensors, strict=False)

print(msg)

model = model.to(torch.device(cfg.device))
model = model.eval()

def get_prompt(conv):
    ret = conv.system + conv.sep
    for role, message in conv.messages:
        if message:
            ret += role + " " + message + " " + conv.sep
        else:
            ret += role
    return ret


def get_prompt2(conv):
    ret = conv.system + conv.sep
    count = 0
    for role, message in conv.messages:
        count += 1
        if count == len(conv.messages):
            ret += role + " " + message
        else:
            if message:
                ret += role + " " + message + " " + conv.sep
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
            model.mistral_tokenizer(
                seg, return_tensors="pt", add_special_tokens=i == 0).to("cuda:0").input_ids
            # only add bos to the first seg
            for i, seg in enumerate(prompt_segs)
        ]
        seg_embs = [model.mistral_model.base_model.model.model.embed_tokens(seg_t) for seg_t in seg_tokens]
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
    
    
def answer(conv, model, img_list, do_sample=True, max_new_tokens=200, num_beams=1, min_length=1, top_p=0.9,
               repetition_penalty=1.0, length_penalty=1, temperature=1.0, answer_prompt=None, print_res=False):
    stop_words_ids = [
        torch.tensor([2]).to("cuda:0"),
        torch.tensor([29871, 2]).to("cuda:0")]  # '</s>' can be encoded in two different ways.
    stopping_criteria = StoppingCriteriaList([StoppingCriteriaSub(stops=stop_words_ids)])
    
    conv.messages.append([conv.roles[1], answer_prompt])
    embs = get_context_emb(conv, model, img_list, answer_prompt=answer_prompt, print_res=print_res)
    with torch.no_grad():
        outputs = model.mistral_model.generate(
            inputs_embeds=embs,
            max_new_tokens=max_new_tokens,
            stopping_criteria=stopping_criteria,
            num_beams=num_beams,
            do_sample=do_sample,
            min_length=min_length,
            top_p=top_p,
            repetition_penalty=repetition_penalty,
            length_penalty=length_penalty,
            temperature=temperature,
        )
    output_token = outputs[0]
    if output_token[0] == 0:  # the model might output a unknow token <unk> at the beginning. remove it
            output_token = output_token[1:]
    if output_token[0] == 1:  # some users find that there is a start token <s> at the beginning. remove it
            output_token = output_token[1:]
    output_text = model.mistral_tokenizer.decode(output_token, add_special_tokens=False)
    output_text = output_text.split('</s>')[0]  # remove the stop sign </s>
#     output_text = output_text.split('[/INST]')[-1].strip()
    conv.messages[-1][1] = output_text + '</s>'
    return output_text, output_token.cpu().numpy()

def get_index(num_frames, num_segments):
    seg_size = float(num_frames - 1) / num_segments
    start = int(seg_size / 2)
    offsets = np.array([
        start + int(np.round(seg_size * idx)) for idx in range(num_segments)
    ])
    return offsets


def load_video(video_path, num_segments=8, return_msg=False, resolution=224):
    if "s3://" in video_path:
        video_bytes = client.get(video_path)
        vr = VideoReader(io.BytesIO(video_bytes), ctx=cpu(0), num_threads=1)
    else:
        vr = VideoReader(video_path, ctx=cpu(0), num_threads=1)
    num_frames = len(vr)
    frame_indices = get_index(num_frames, num_segments)

    # transform
    crop_size = resolution
    scale_size = resolution
    input_mean = [0.48145466, 0.4578275, 0.40821073]
    input_std = [0.26862954, 0.26130258, 0.27577711]

    transform = T.Compose([
        GroupScale(int(scale_size), interpolation=InterpolationMode.BICUBIC),
        GroupCenterCrop(crop_size),
        Stack(),
        ToTorchFormatTensor(),
        GroupNormalize(input_mean, input_std) 
    ])

    images_group = list()
    for frame_index in frame_indices:
        img = Image.fromarray(vr[frame_index].numpy())
        images_group.append(img)
    torch_imgs = transform(images_group)
    if return_msg:
        fps = float(vr.get_avg_fps())
        sec = ", ".join([str(round(f / fps, 1)) for f in frame_indices])
        # " " should be added in the start and end
        msg = f"The video contains {len(frame_indices)} frames sampled at {sec} seconds."
        return torch_imgs, msg
    else:
        return torch_imgs

def check_answer_egoschema(pred, qid):
    correct = 0
    answer_content = ans_dict[qid]['content'].lower()
    if answer_content[-1] == ".":
        answer_content = answer_content[:-1]
    if ans_dict[qid]['answer'].lower() in pred.lower():
        flag = True
        for kk in ["(A)", "(B)", "(C)", "(D)", "(E)"]:
            if kk != ans_dict[qid]['answer'].lower() and kk in pred.lower():
                flag = ans_dict
                break
        if flag:
            correct += 1
    elif answer_content in pred.lower():
        correct = 1
    elif answer_content.replace("a ", "") in pred.lower():
        correct = 1
    elif answer_content.replace("an ", "") in pred.lower():
        correct = 1
    return correct

def infer_egoschema(
        data_sample, system="", 
        question_prompt='', # add in the end of question
        answer_prompt=None, # add in the begining of answer
        return_prompt='',  # add in the begining of return message
        system_q=False, # whether add question in the system prompt for QFormer
        print_res=True,
        system_llm=False,
        num_segments=8,
    ):
    vid_path = os.path.join("shdd:s3://egoschema/videos", data_sample['video'])
    print(vid_path)
    video, _ = load_video(vid_path, num_segments=num_segments, return_msg=True)
    TC, H, W = video.shape
    video = video.reshape(1, TC//3, 3, H, W).to("cuda:0")
    
    video_list = []
    with torch.no_grad():
        if system_q:
            video_emb, _ = model.encode_img(video, system + data_sample['question'])
        else:
            video_emb, _ = model.encode_img(video, system)
    video_list.append(video_emb)
#     video_list.append(torch.zeros_like(video_emb))

    chat = EasyDict({
        "system": system,
        "roles": ("[INST]", "[/INST]"),
        "messages": [],
        "sep": ""
    })

    chat.messages.append([chat.roles[0], f"<Video><VideoHere></Video> [/INST]"])
    
    if system_llm:
        prompt = system + data_sample['QA'][0]['q'] + question_prompt
    else:
        prompt = data_sample['QA'][0]['q'] + question_prompt
    
    ask(prompt, chat)

    llm_message = answer(
        conv=chat, model=model, do_sample=False, 
        img_list=video_list, max_new_tokens=100, 
        answer_prompt=answer_prompt, print_res=print_res
    )[0]
    # remove potential explanation
    llm_message = return_prompt + llm_message.strip().split('\n')[0]
    print(llm_message)
    print(f"GT: {data_sample['QA'][0]['a']}")
    return llm_message

from petrel_client.client import Client
from decord import VideoReader, cpu
client = Client('~/petreloss.conf', enable_mc=False)

import csv
with open("/mnt/petrelfs/wangchenting/IG-VLM/data/multiple_choice_qa/EgoSchema.csv", mode='r', encoding='utf-8') as file:
    reader = csv.reader(file)

    json_data = []
    ans_dict = {}
    
    for idx, msg in enumerate(reader):
        if idx == 0:
            print(msg)
            continue
            
        video = msg[1] + '.mp4'
        input_str = f"Question: {msg[3].capitalize()}\nOptions:\n"
    
        target_index = -1
        for i, candidate in enumerate(msg[5:]):
            option = chr(ord('A') + i)
            input_str += f"({option}) {candidate}\n"
            if candidate == msg[4]:
                target_index = i
            
        assert target_index != -1
        correct = chr(ord('A') + target_index)
        
        json_data.append({
            'video': video,
            "QA": [{
                "i": "",
                "q": input_str.strip(),
                "a": f"Answer: ({correct}) {msg[4]}",
            }]
        })

        ans_dict[idx - 1] = {
            'video': video,
            'answer': f"({correct})",
            'content': msg[4],
        }

#  position embedding
num_frame = 16
resolution = 224
# new_pos_emb = get_sinusoid_encoding_table(n_position=(resolution//16)**2*num_frame, cur_frame=num_frame)
# model.vision_encoder.encoder.pos_embed = new_pos_emb

correct = 0
total = 0
total_num = len(json_data)

output = ""

for idx, example in enumerate(tqdm(json_data)):
    start = time.time()
    llm_message = infer_egoschema(
        example, 
        "Carefully watch the video and pay attention to the cause and sequence of events, the detail and movement of objects, and the action and pose of persons. Based on your observations, select the best option that accurately addresses the question.\n", 
        question_prompt="\nOnly give the best option.", 
        answer_prompt="Best option:(",
        return_prompt='(',
        system_q=False,
        print_res=False,
        system_llm=False,
        num_segments=16
    )
    
    duration = time.time() - start
    output += (example["video"] + '\n')
    output += (llm_message + '\n')
    correct += check_answer_egoschema(llm_message, idx)
    total += 1
    print("Acc:", correct / total)
    print('-' * 20, f'{idx+1}/{total_num} done,', f'cost: {duration:.2f}s', '-' * 20)
    
with open("./demo/egoschema/1b_mistral_7b_frame16.txt", "a") as f:
    f.writelines(output)
    
with open("/mnt/petrelfs/share_data/xujilan/EgoSchema/questions.json", "r") as f:
    full_data = json.load(f)

full_egoschema = []
for data in full_data:
    video = data['q_uid'] + '.mp4'
    input_str = f"Question: {data['question'].capitalize()}\nOptions:\n"

    for i, candidate in enumerate(['option 0', 'option 1', 'option 2', 'option 3', 'option 4']):
        option = chr(ord('A') + i)
        input_str += f"({option}) {data[candidate]}\n"
    
    full_egoschema.append({
        'q_uid': data['q_uid'],
        'video': video,
        "QA": [{
            "i": "",
            "q": input_str.strip(),
            "a": "",
        }]
    })


def infer_full_egoschema(
        data_sample, system="", 
        question_prompt='', # add in the end of question
        answer_prompt=None, # add in the begining of answer
        return_prompt='',  # add in the begining of return message
        system_q=False, # whether add question in the system prompt for QFormer
        print_res=True,
        system_llm=False,
        num_segments=8,
    ):
    vid_path = os.path.join("shdd:s3://egoschema/videos", data_sample['video'])
    print(vid_path)
    video, _ = load_video(vid_path, num_segments=num_segments, return_msg=True)
    TC, H, W = video.shape
    video = video.reshape(1, TC//3, 3, H, W).to("cuda:0")
    
    video_list = []
    with torch.no_grad():
        if system_q:
            video_emb, _ = model.encode_img(video, system + data_sample['question'])
        else:
            video_emb, _ = model.encode_img(video, system)
    video_list.append(video_emb)

    chat = EasyDict({
        "system": system,
        "roles": ("[INST]", "[/INST]"),
        "messages": [],
        "sep": ""
    })

    chat.messages.append([chat.roles[0], f"<Video><VideoHere></Video> [/INST]"])
    
    if system_llm:
        prompt = system + data_sample['QA'][0]['q'] + question_prompt
    else:
        prompt = data_sample['QA'][0]['q'] + question_prompt
    
    ask(prompt, chat)

    llm_message = answer(
        conv=chat, model=model, do_sample=False, 
        img_list=video_list, max_new_tokens=100, 
        answer_prompt=answer_prompt, print_res=print_res
    )[0]
    # remove potential explanation
    llm_message = return_prompt + llm_message.strip().split('\n')[0]
    print(llm_message)
    return llm_message


#  position embedding
num_frame = 16
resolution = 224
# new_pos_emb = get_sinusoid_encoding_table(n_position=(resolution//16)**2*num_frame, cur_frame=num_frame)
# model.vision_encoder.encoder.pos_embed = new_pos_emb

ans_dict = {}

for idx, example in enumerate(tqdm(full_egoschema)):
    start = time.time()
    llm_message = infer_full_egoschema(
        example, 
        "Carefully watch the video and pay attention to the cause and sequence of events, the detail and movement of objects, and the action and pose of persons. Based on your observations, select the best option that accurately addresses the question.\n", 
        question_prompt="\nOnly give the best option.", 
        answer_prompt="Best option:(",
        return_prompt='(',
        system_q=False,
        print_res=False,
        system_llm=True,
        num_segments=16,
    )

    assert llm_message[0] == '(' and llm_message[2] == ')'
    ans = ord(llm_message[1]) - ord('A')
    assert ans in [0, 1, 2, 3, 4]
    ans_dict[example['q_uid']] = ans

import json 
out_file = open("egoschema_ans_f16_systemllm.json", "w") 
json.dump(ans_dict, out_file) 
out_file.close()