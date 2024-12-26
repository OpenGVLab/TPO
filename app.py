import ast
import io
import numpy as np
import os
import math
import argparse
from easydict import EasyDict
from torchvision import transforms
from torchvision.transforms import InterpolationMode

# 读数据需要的
import decord
from decord import VideoReader
decord.bridge.set_bridge("torch")
import random
import cv2
import matplotlib.pyplot as plt
from PIL import Image
import subprocess

from transformers import AutoModel, AutoTokenizer
from transformers import LlamaTokenizer

from tokenizer import MultimodalLlamaTokenizer


from third_party.cgdetr.cg_detr.span_utils import span_cxw_to_xx
from third_party.cgdetr.cg_detr.postprocessing_cg_detr import PostProcessorDETR
from third_party.cgdetr.utils.basic_utils import load_jsonl, l2_normalize_np_array
from third_party.internvideo2.models.internvideo2_clip import InternVideo2_CLIP

import torch
import torch.nn.functional as F

import gradio as gr
from gradio.themes.utils import colors, fonts, sizes

from inference.modeling_special_token import special_tokens



import logging

logger = logging.getLogger(__name__)


num_gpu = 1
device = f'cuda:0'



def images_to_video_ffmpeg(image_folder, output_video_path):
    image_pattern = os.path.join(image_folder, "%d.png")
    command = [
        "ffmpeg",
        "-y",
        "-i", image_pattern,           # 输入图片序列
        "-c:v", "libx264",             # 使用 H.264 编码器
        output_video_path              # 输出视频路径
    ]
    subprocess.run(command, check=True)
    print(f"视频已保存到 {output_video_path}")

def draw_contours(mask):
    if len(mask.shape) != 2 or mask.dtype != np.uint8:
        raise ValueError("输入必须是二值图像（单通道）")
    # 复制原始mask以便绘制边框
    output_image = mask.copy()  # 保持为灰度图像
    contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    cv2.drawContours(output_image, contours, -1, (255), 5)  # 白色边框，线宽为2
    return output_image

def show_mask(video_raw, mask, save_path, random_color=False):
    if random_color:
        color = np.concatenate([np.random.random(3), np.array([0.6])], axis=0)
    else:
        orange_yellow_rgb = np.array([255, 165, 0]) / 255.0  # 归一化
        alpha = np.array([0.6])  # 透明度
        color = np.concatenate([orange_yellow_rgb, alpha], axis=0)
    plt.cla()
    video_raw = video_raw.permute(1,2,0) # c,h,w -> h,w,c
    plt.imshow(video_raw)
    # import pdb; pdb.set_trace()
   
    mask2 = draw_contours(mask)
    h, w = mask.shape[-2:]
    mask_image = mask2.reshape(h, w, 1) * color.reshape(1, 1, -1)
    plt.imshow(mask_image)
    # plt.axis('off')
    plt.savefig(save_path, bbox_inches='tight', pad_inches=0)
    return mask

def show_points(coords, labels, ax, marker_size=200):
    pos_points = coords[labels==1]
    neg_points = coords[labels==0]
    ax.scatter(pos_points[:, 0], pos_points[:, 1], color='green', marker='*', s=marker_size, edgecolor='white', linewidth=1.25)
    ax.scatter(neg_points[:, 0], neg_points[:, 1], color='red', marker='*', s=marker_size, edgecolor='white', linewidth=1.25)
    return 

def show_box(vision_path, box, save_path):
    plt.cla()
    if isinstance(vision_path, str):
        plt.imshow(Image.open(vision_path))
    else:
        vision_path = vision_path.permute(1,2,0) # c,h,w -> h,w,c
        plt.imshow(vision_path)
    plt.axis('off')
    x0, y0 = box[0], box[1]
    w, h = box[2] - box[0], box[3] - box[1]
    ax = plt.gca()
    ax.add_patch(plt.Rectangle((x0, y0), w, h, edgecolor='green', facecolor=(0, 0, 0, 0), lw=2))
    plt.savefig(save_path, bbox_inches='tight', pad_inches=0)
    return

def show_hl(score, save_path):
    # x = [num - min(score) for num in score]
    f_x = np.exp(score) / np.sum(np.exp(score))
    y = range(len(f_x))
    plt.yticks([])

    fig, ax = plt.subplots(figsize=(100, 3))
    ax.plot(y, f_x, linewidth =20.0, color='darkorange')
    plt.savefig(save_path)
    return 

def show_temporal(scores, frames, relevant_windows):

    # x = [num - min(scores) for num in scores]
    # scores = np.exp(scores) / np.sum(np.exp(scores))
    scores = np.array(scores)  # 将 scores 转换为 numpy 数组
    scores = (scores - scores.min()) / (scores.max() - scores.min())

    fig, ax = plt.subplots(figsize=(len(scores), 6))

    x = np.arange(len(scores))
    # ax.plot(x, scores, marker='o', linestyle='-')
    # relevant_windows = [2,4]

    start = int(relevant_windows[0])
    end = int(relevant_windows[1])
    mid = (start + end) // 2
    indices = [start, mid, end]
    top_frames = frames[indices]
    images = []
    for i in range(3):
        if isinstance(top_frames, torch.Tensor):
            image_np = top_frames[i].cpu().numpy()
            image_np = np.transpose(image_np, (1, 2, 0))
            if image_np.dtype.kind == 'f':
                image_np = (image_np * 255).astype(np.uint8)
            image_pil = Image.fromarray(image_np)

        # images.append(image_pil.resize((50, 50)))
        images.append(image_pil)
    ratio = 100/images[0].size[0]
    # import pdb; pdb.set_trace()
    is_offset=False
    if indices[2]-indices[0] <= 2:
        is_offset=True

    for i, index in enumerate(indices):
        score = scores[index]
        image = images[i]
        im = OffsetImage(image, zoom=ratio)
        # import pdb; pdb.set_trace()
        offset = 0
        if is_offset and i==0:
            offset = -60
        if is_offset and i==2:
            offset = 60
        ab = AnnotationBbox(im, (index, score),
                            xybox=(offset, -50),  # 调整 xybox 值控制图片位置
                            xycoords='data',
                            boxcoords="offset points",
                            pad=0.5,
                            arrowprops=dict(arrowstyle="->"))
        ax.add_artist(ab)
        if i==0:
            ax.text(index+offset/60, scores[index]-0.28 , f"start:{index}s", ha="center", va="top")
        if i==2:
            ax.text(index+offset/60, scores[index]-0.28 , f"end:{index}s", ha="center", va="top")

    ax.plot(x, scores, marker='o', linestyle='-',zorder=10)
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)

    ax.set_xlabel("time/s")
    ax.set_ylabel("score")
    # ax.set_title("Highlight 分数")
    # ax.grid(True)

    # plt.show()
    plt.savefig("outputs/2.png", bbox_inches='tight', pad_inches=0)

class Inference():
    def __init__(self, model_path, special_tokens):
        self.model_path = model_path
        self.tokenizer = MultimodalLlamaTokenizer.from_pretrained(
            model_path,
            local_files_only=True,
            n_query=96,
            v_query=96,
            special_tokens = special_tokens, 
        ) 
        self.model = AutoModel.from_pretrained(model_path,  trust_remote_code=True, _tokenizer=self.tokenizer).eval()

    def get_transform(self, task_head):
        mean = (0.485, 0.456, 0.406)
        std = (0.229, 0.224, 0.225)
        normalize = transforms.Normalize(mean, std)
        type_transform = transforms.Lambda(lambda x: x.float().div(255.0))
        aug_transform = transforms.Lambda(lambda x: x)
        if task_head == "Mask":
            transform = transforms.Compose(
                [   
                    type_transform,
                    transforms.Resize(
                        (1024, 1024),
                        interpolation=InterpolationMode.BICUBIC,
                    ),
                    normalize,
                ]
            )
        elif task_head == "Region":
            transform = transforms.Compose(
                [   
                    aug_transform,
                    type_transform,
                    normalize,
                ]
            )
        else: # Temporal head & vision encoder
            transform = transforms.Compose(
                [
                    transforms.Resize(
                        (224, 224),
                        interpolation=InterpolationMode.BICUBIC,
                    ),
                    type_transform,
                    normalize,
                ]
            )
        return transform
    
    def get_frame_indices(self, num_frames, vlen, sample='rand', fix_start=None, input_fps=1, max_num_frames=-1):
        if sample in ["rand", "middle"]: # uniform sampling
            acc_samples = min(num_frames, vlen)
            # split the video into `acc_samples` intervals, and sample from each interval.
            intervals = np.linspace(start=0, stop=vlen, num=acc_samples + 1).astype(int)
            ranges = []
            # print(intervals)
            for idx, interv in enumerate(intervals[:-1]):
                ranges.append((interv, intervals[idx + 1] - 1))
            # print(ranges)
            if sample == 'rand':
                try:
                    frame_indices = [random.choice(range(x[0], x[1])) for x in ranges]
                except:
                    frame_indices = np.random.permutation(vlen)[:acc_samples]
                    frame_indices.sort()
                    frame_indices = list(frame_indices)
            elif fix_start is not None:
                frame_indices = [x[0] + fix_start for x in ranges]
            elif sample == 'middle':
                frame_indices = [(x[0] + x[1]) // 2 for x in ranges]
            else:
                raise NotImplementedError

            # print(frame_indices)
            
            if len(frame_indices) < num_frames:  # padded with last frame
                padded_frame_indices = [frame_indices[-1]] * num_frames
                padded_frame_indices[:len(frame_indices)] = frame_indices
                frame_indices = padded_frame_indices
            
            # print(frame_indices)
        elif "fps" in sample:  # fps0.5, sequentially sample frames at 0.5 fps
            output_fps = float(sample[3:])
            duration = float(vlen) / input_fps
            delta = 1 / output_fps  # gap between frames, this is also the clip length each frame represents
            frame_seconds = np.arange(0 + delta / 2, duration + delta / 2, delta)
            frame_indices = np.around(frame_seconds * input_fps).astype(int)
            frame_indices = [e for e in frame_indices if e < vlen]
            if max_num_frames > 0 and len(frame_indices) > max_num_frames:
                frame_indices = frame_indices[:max_num_frames]
        else:
            raise ValueError
        if "fps" in sample and len(frame_indices) % output_fps != 0:
            frame_indices = frame_indices[:-int(len(frame_indices)%output_fps)]
        return frame_indices
    
    def read_frames_decord(self, video_path, num_frames, sample='rand', fix_start=None, max_num_frames=-1, all_frames=False, client=None, clip=None):
        if 's3://' in video_path:
            video_bytes = client.get(video_path)
            video_reader = VideoReader(io.BytesIO(video_bytes), num_threads=1)
        else:
            video_reader = VideoReader(video_path, num_threads=1)
        vlen = len(video_reader)
        fps = video_reader.get_avg_fps()
        duration = vlen / float(fps)

        if clip:
            start, end = clip
            duration = end - start
            vlen = int(duration * fps)
            start_index = int(start * fps)

        if all_frames:
            frame_indices = [i for i in range(0, vlen, 1)]
            all_frames = video_reader.get_batch(frame_indices)  # (T, H, W, C), torch.uint8
            if isinstance(all_frames, torch.Tensor):
                all_frames = all_frames.permute(0, 3, 1, 2)
            else:
                all_frames =  torch.from_numpy(all_frames.asnumpy()).permute(0, 3, 1, 2)  # (T, C, H, W), torch.uint8
            return all_frames, frame_indices, float(fps), vlen, duration

        frame_indices = self.get_frame_indices(
            num_frames, vlen, sample=sample, fix_start=fix_start,
            input_fps=fps, max_num_frames=max_num_frames
        )
        if clip:
            frame_indices = [f + start_index for f in frame_indices]

        frames = video_reader.get_batch(frame_indices)  # (T, H, W, C), torch.uint8
        try:
            frames = frames.permute(0, 3, 1, 2)  # (T, C, H, W), torch.uint8
        except:
            frames = torch.from_numpy(frames.asnumpy()).permute(0, 3, 1, 2) 
        return frames, frame_indices, float(fps), vlen, duration
    
    def process_qa(self, text, media_type="video", msg="", max_length=1024):
        cur_instruction = ""
        input_ids, attention_masks, labels = [], [], []
        conversation = ""
        if cur_instruction:
            conversation += cur_instruction

        conversation += (
            "[INST]" + " "
        )
        if media_type == "image":
            conversation += "<Image>" + IMG_TOKEN + "</Image>"
        elif media_type == "video":
            conversation += "<Video>" + VID_TOKEN + "</Video>"

        conversation += (
            msg.rstrip() + "[/INST]"
        )
        total_len = 0
        indexs = []
        tokenized = self.tokenizer.build_input_ids(
            text=[conversation],
            max_length=max_length,
            add_special_tokens=False,
            truncation=False,
            require_image = (media_type == "image"),
            require_video = (media_type == "video"),
            padding=False,
            return_tensors='pt'
        )
        if media_type == "image":
            indexs.append(tokenized['image_index'])
        elif media_type == "video":
            indexs.append(tokenized['video_index'])
        # logger.info(f'video_index:{indexs}')
        input_ids.append(tokenized['input_ids'])
        attention_masks.append(tokenized['attention_mask'])
        labels.append(torch.ones_like(tokenized['input_ids']) * -100)
        total_len += tokenized['input_ids'].shape[0]
        gtext_input = conversation
        q = text
        if q != "":
            conversation_q = (" " + "[INST]" + " " + q + " " + "[/INST]")
        else:
            # no question, often in caption dataset
            conversation_q = ""
        conversation_q += (" ")
        tokenized = self.tokenizer.build_input_ids(
            text=[conversation_q],
            max_length=max_length,
            add_special_tokens=False,
            truncation=False,
            padding=False,
            return_tensors='pt'
        )
        indexs.append(torch.zeros_like(tokenized['input_ids']).to(torch.bool))
        input_ids.append(tokenized['input_ids'])
        attention_masks.append(tokenized['attention_mask'])
        labels.append(torch.ones_like(tokenized['input_ids']) * -100)
        total_len += tokenized['input_ids'].shape[0]

        gtext_input += conversation_q
            
        input_ids = torch.cat(input_ids)[:max_length]
        attention_masks = torch.cat(attention_masks)[:max_length]
        labels = torch.cat(labels)[:max_length]
        indexs = torch.cat(indexs)[:max_length]
        labels[labels == self.tokenizer.pad_token_id] = -100

        return input_ids, attention_masks, labels, indexs, cur_instruction, gtext_input

    def generate(self, input_ids, attention_mask, video, video_index):
        task_head = None
        temperature = 1e-5
        max_new_tokens = 20
        output_ids = list(input_ids[0])
        # import pdb; pdb.set_trace()
        prompt_token = None
        text_embeds = self.model.pad_text_embeds(input_ids=input_ids, video_idx=video_index, video=video)
        for i in range(max_new_tokens):
            if i == 0:
                outputs = self.model.lm(
                        inputs_embeds=text_embeds,
                        attention_mask=attention_mask,
                        output_hidden_states=True,
                        return_dict=True,
                    ) # 整个query输入得到的输出
                logits = outputs.logits
                past_key_values = outputs.past_key_values
            else:
                attention_mask = torch.ones(1, past_key_values[0][0].shape[-2] + 1, device=device)
                last_text_embeds = self.model.lm.get_input_embeddings()(torch.tensor(output_ids[-1], device=device).long()).detach().unsqueeze(0)
                last_text_embeds = last_text_embeds.unsqueeze(0)
                
                out = self.model.lm(
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
            # print(ret)
            if ret == '<time_begin>':
                attention_mask = torch.ones(1, past_key_values[0][0].shape[-2] + 1, device=device)
                temp_embeds = self.model.temporal_token.bfloat16()
                out = self.model.lm(inputs_embeds=temp_embeds,use_cache=True,attention_mask=attention_mask,output_hidden_states=True,past_key_values=past_key_values)
                prompt_token = out.hidden_states[-1]
                task_head = "Temporal"

            elif ret == '<box_begin>':
                attention_mask = torch.ones(1, past_key_values[0][0].shape[-2] + 1, device=device)
                bbox_embeds = self.model.box_token.bfloat16()
                # text_embeds_bbox = torch.cat((text_embeds.squeeze(0), bbox_embeds), dim=0).unsqueeze(0)
                out = self.model.lm(
                    inputs_embeds=bbox_embeds,
                    use_cache=True,
                    attention_mask=attention_mask,
                    output_hidden_states=True,
                    past_key_values=past_key_values
                )
                last_hidden_states = out.hidden_states[-1]
                # selected_hidden_states = last_hidden_states[0][len(attention_mask)-2]
                prompt_token = last_hidden_states[0][0]
                # print(f'{}')
                task_head = "Region"

            elif ret == '<track_begin>':
                attention_mask = torch.ones(1, past_key_values[0][0].shape[-2] + 1, device=device)
                tracking_embeds = self.model.track_token
                out = self.model.lm(
                    inputs_embeds=tracking_embeds,
                    use_cache=True,
                    attention_mask=attention_mask,
                    output_hidden_states=True,
                    past_key_values=past_key_values
                )
                last_hidden_states = out.hidden_states[-1]
                prompt_token = last_hidden_states[0][0].to(dtype = torch.bfloat16)
                task_head = "Mask"

            if (ret == '</s>'):
                break
        ret = self.tokenizer.decode(output_ids)
        del past_key_values

        return ret, prompt_token, task_head

    def load_and_transform_video(self, video_path, num_frames=16, transform=None, all_frames=False):
        frames, frame_indices, fps, vlen, duration = self.read_frames_decord(video_path=video_path, num_frames=num_frames, sample='middle', all_frames=all_frames)
        size_hw = frames.shape[-2:]  # h, w
        if transform:
            frames = transform(frames)
        sec = [str(round(f / fps, 1)) for f in frame_indices]
        return frames, duration, sec, size_hw

    def load_image(self, data_path, transform, use_dec=False):
        image = load_image_from_path(data_path)
        image_size = image.shape[-2:]  # h, w
        if use_dec:
            image = dec_transform(image)
        image = transform(image)
        return image, image_size
        
    def new_llama_forward(self, model, text):
        text_key_padding_mask = text > 0
        x = model.transformer(input_ids=text, attention_mask=text_key_padding_mask).last_hidden_state   # 4096
        all_feats = []
        for i in range(x.shape[0]):
            feats = x[i][:text_key_padding_mask[i].sum()]
            all_feats.append(feats)
        return all_feats

    def extract_video(self, video_path, transform, model, use_tef=True):
        frames, frame_indices, fps, vlen, duration = self.read_frames_decord(video_path=video_path, num_frames=16, sample='fps8.0')  # 
        frames = transform(frames)
        T, C, H, W = frames.shape
        frames = frames.reshape(T//8, 8, C, H, W)
        max_len = 100
        with torch.no_grad():
            if T//8 > max_len:
                l = []
                for i in range(math.ceil((T//8)/max_len)):  # 超过100个视频片段可能会爆显存
                    if (i+1)*max_len <= T//8:
                        feat = model.encode_vision(frames[i*max_len:(i+1)*max_len,:,:,:,:].to(device))
                    else:
                        feat = model.encode_vision(frames[i*max_len:,:,:,:,:].to(device))
                    feat = feat.to(torch.bfloat16)
                    l.append(feat)
                feat = torch.concat(l,dim=0)
            else:
                feat = model.encode_vision(frames.to(device))
            feat = feat.to(torch.bfloat16)

        ctx_l = len(feat)
        if use_tef:
            tef_st = torch.arange(0, ctx_l, 1.0) / ctx_l
            tef_ed = tef_st + 1.0 / ctx_l
            tef = torch.stack([tef_st, tef_ed], dim=1).to(torch.bfloat16).to(device)  # (L, 2)
            feat = torch.cat([feat, tef], dim=1)  # (L, D+2)
        feat_mask = torch.ones([feat.shape[0]]).to(device)
        return feat, feat_mask, duration

    def extract_text(self, text, model):
        text = [text]              
        text = model.tokenizer(text).cuda()
        with torch.no_grad():
            text = text.cuda()
            feat = self.new_llama_forward(model.text_encoder, text)
            feat = feat[0]
            feat = feat.to(torch.bfloat16)
        feat = feat[:100] # max_query len: 100
        feat_mask = torch.ones([feat.shape[0]]).to(device)
        return feat, feat_mask

    def run(self, text, vision_path, chatbot):
        output_path = "tmp/"
        os.makedirs(output_path, exist_ok=True)
        output_img = None
        output_vid = None
        self.model.to(device)
        self.model.eval()
        self.model = self.model.to(torch.bfloat16)
        if vision_path[-4:] == ".mp4":
            media_type = "video"
            transform = self.get_transform("vision_encoder")
            video, duration, sec, size_hw = self.load_and_transform_video(vision_path, transform=transform)
            # import pdb; pdb.set_trace()
            video = video.to(device)
            if "timestamp" in text:
                msg = f" The video contains {len(sec)} frames sampled at {', '.join(sec)} seconds. "
            else:
                msg = ""
            tracking_box = text[text.find('['): text.find(']')+1]
            if len(tracking_box) != 0:
                tracking_box = ast.literal_eval(tracking_box)
                if len(tracking_box) != 4:
                    tracking_box = None
        else:
            media_type = "image"
            transform = self.get_transform("Region")
            video, raw_size = self.load_image(vision_path, transform, use_dec=True)
            video = video.to(device)
            msg = ""

        input_ids, attention_masks, labels, video_index, cur_instruction, gtext_input = self.process_qa(text, media_type, msg=msg)
        input_ids, attention_masks, labels= input_ids.to(device), attention_masks.to(device), labels.to(device)
        input_ids, attention_masks, video, video_index = input_ids.unsqueeze(0), attention_masks.unsqueeze(0), video.unsqueeze(0), video_index.unsqueeze(0)
        ret, prompt_token, task_head = self.generate(input_ids, attention_masks, video.type(dtype=torch.bfloat16), video_index)
        # task_transform = self.get_transform(task_head)
        print(ret)
        if prompt_token == None:
            # print("Pred error!")
            print("No decoder required")

        if task_head == "Temporal":
            saliency_scores, relevant_windows, frames = self.temporal_head(text, vision_path, prompt_token)
            print(saliency_scores)
            print(relevant_windows)
            frames, frame_indices, fps, vlen, duration = self.read_frames_decord(vision_path, num_frames=16, sample='fps1.0')
            show_temporal(saliency_scores, frames, relevant_windows)
            # show_hl(saliency_scores, f"{output_path}/hl_score.png")
            output_img = f"{output_path}/hl_score.png"

        elif task_head == "Region":
            box = self.region_head(prompt_token).detach().to(torch.float).cpu().numpy()
            box = np.array([box[0]*raw_size[1], box[1]*raw_size[0], box[2]*raw_size[1],box[3]*raw_size[0]])
            show_box(vision_path, box, f"{output_path}/output.png")
            output_img = f"{output_path}/hl_score.png"

        elif task_head == "Mask":
            video_raw, duration, sec, size_hw = self.load_and_transform_video(vision_path, all_frames=True)
            if tracking_box:
                tracking_box = [tracking_box[0]/video_raw.shape[3], tracking_box[1]/video_raw.shape[2], tracking_box[2]/video_raw.shape[3], tracking_box[3]/video_raw.shape[2]]
            video_segments = self.mask_head(vision_path, prompt_token, tracking_box)
            if tracking_box:
                for idx, video_segment in enumerate(video_segments):
                    box = self.model.find_boundaries_torch(video_segment.squeeze(0).cpu()).to(torch.float)
                    box = np.array([box[0]*video_raw.shape[3], box[1]*video_raw.shape[2], box[2]*video_raw.shape[3],box[3]*video_raw.shape[2]])
                    show_box(video_raw[idx], box, f"{output_path}/box_{idx}.png")
                    # show_mask(video_raw[idx], np.array(video_segment[0].cpu(), dtype=np.uint8), f"{output_path}/seg_{idx}.png")
            else:
                for idx, video_segment in enumerate(video_segments):
                    show_mask(video_raw[idx], np.array(video_segment[0].cpu(), dtype=np.uint8), f"{output_path}/{idx}.png")
            images_to_video_ffmpeg(f"{output_path}", f"{output_path}/video.mp4")
            output_vid = f"{output_path}/video.mp4"

        chatbot = chatbot + [[text, ret]]
        return chatbot, output_img, output_vid
    
    def temporal_head(self, text, video_path, prompt_token):
        task_head = "Temporal"
        # query = text.split("'")[1]
        query = text
        model_config = dict(
            model_cls="InternVideo2_CLIP",
            vision_encoder=dict(
                name="internvideo2",
                in_chans=3,
                patch_size=14,
                img_size=224,
                qkv_bias=False,
                drop_path_rate=0.3,
                head_drop_path_rate=0.,
                embed_dim=1408,
                num_heads=16,
                mlp_ratio=48/11,
                init_values=0.1,
                qk_normalization=True,
                depth=40,
                use_flash_attn=False,
                use_fused_rmsnorm=False,
                use_fused_mlp=False,
                fused_mlp_heuristic=1,
                drop_cls_token=False,
                attn_pool_num_heads=16,
                clip_embed_dim=768,
                layerscale_no_force_fp32=True,
                num_frames=8,  # 8
                tubelet_size=1,
                sep_pos_embed=False,
                use_checkpoint=False,
                checkpoint_num=0,
            ),
            text_encoder=dict(
                use_flash_attn=True,
                transformer_width=4096,
                llama_path="/mnt/petrelfs/share_data/likunchang/model/chinese_alpaca_lora_7b",
                use_lora=True,
            ),
            temp=1 / 100.0,
            temp_min=1 / 100.0,
            freeze_vision=True,
            open_vision_clip_projector=False,
            freeze_text=True,
            open_text_projection=False,
            open_text_lora=False,
            tokenizer_path="/mnt/petrelfs/share_data/likunchang/model/chinese_alpaca_lora_7b",
            vision_ckpt_path="/mnt/petrelfs/share_data/lixinhao/models/internvideo2/avp_1b_f4_coco_smit_e4.pt",     # "/mnt/petrelfs/share_data/lixinhao/avp_1b_f4_coco_smit_e4.pt", "/mnt/petrelfs/share_data/lixinhao/avp_6b_f4_coco_smit_e4_mix4_e4_best.pt"
            load_vision_ckpt_from_internvideo2_stage2=True,
            text_ckpt_path="/mnt/petrelfs/share_data/likunchang/model/internvl/internvl_c_13b_224px.pth",
            extra_ckpt_path="/mnt/petrelfs/share_data/likunchang/model/internvideo2/clip/1B/1B_clip.pth"   # 6B
        )
        internvideo2_clip = InternVideo2_CLIP(EasyDict(model=model_config))
        internvideo2_clip.to(device)
        internvideo2_clip.eval()
        internvideo2_clip = internvideo2_clip.to(torch.bfloat16)

        transform = self.get_transform(task_head)

        v_feat, v_feat_mask, duration, frames = self.extract_video(video_path, transform, internvideo2_clip)
        q_feat, q_feat_mask = self.extract_text(query, internvideo2_clip)
        q_feat, q_feat_mask, v_feat, v_feat_mask = q_feat.unsqueeze(0), q_feat_mask.unsqueeze(0), v_feat.unsqueeze(0), v_feat_mask.unsqueeze(0)
        tvg_outputs = self.model.cg_model(q_feat, q_feat_mask, v_feat, v_feat_mask, prompt_token=prompt_token)
        prob = F.softmax(tvg_outputs["pred_logits"], -1)  # (batch_size, #queries, #classes=2)
        scores = prob[..., 0]  # * (batch_size, #queries)  foreground label is 0, we directly take it
        pred_spans = tvg_outputs["pred_spans"]  # (bsz, #queries, 2)
        _saliency_scores = tvg_outputs["saliency_scores"].half()  # (bsz, L)
        saliency_scores = []
        valid_vid_lengths = v_feat_mask.sum(1).cpu().tolist()
        for j in range(len(valid_vid_lengths)):
            saliency_scores.append(_saliency_scores[j, :int(valid_vid_lengths[j])].tolist())

        # compose predictions
        mr_res = []
        for idx, (spans, score) in enumerate(zip(pred_spans.cpu(), scores.cpu())):
            spans = span_cxw_to_xx(spans) * duration
            spans = torch.clamp(spans, 0, duration)
            cur_ranked_preds = torch.cat([spans, score[:, None]], dim=1).tolist()
            cur_ranked_preds = sorted(cur_ranked_preds, key=lambda x: x[2], reverse=True)
            cur_ranked_preds = [[float(f"{e:.4f}") for e in row] for row in cur_ranked_preds]

            cur_query_pred = dict(
                pred_relevant_windows=cur_ranked_preds,
                pred_saliency_scores=saliency_scores[idx],
            )
            mr_res.append(cur_query_pred)

        post_processor = PostProcessorDETR(
            clip_length=1, min_ts_val=0, max_ts_val=50000,         # clip_length(s) per segment. sta是1秒一个片段，qvhigh是2秒一个片段
            min_w_l=0, max_w_l=50000, move_window_method="left",
            process_func_names=(["round_multiple"])
        )
        mr_res = post_processor(mr_res)
        saliency_scores = mr_res[0]["pred_saliency_scores"]
        relevant_windows = mr_res[0]["pred_relevant_windows"][0]
        return saliency_scores, relevant_windows, frames
    
    def region_head(self, prompt_token):
        bbox = self.model.loc_decoder(prompt_token)
        return bbox
    
    def mask_head(self, vision_path, prompt_token, tracking_box=None):
        task_head = "Mask"
        transform = self.get_transform(task_head)
        video, duration, sec, raw_size_hw = self.load_and_transform_video(vision_path, transform=transform, all_frames=True) #
        video = video.to(device)#[:99]
        video = video.to(torch.bfloat16) 
        video = video.unsqueeze(0)
        inference_state = self.model.sam.init_state_images(video, raw_size_hw[0], raw_size_hw[1])

        if tracking_box:
            tracking_box = torch.tensor(tracking_box)
            embed_sam_boxes = self.model.sam.get_prompt_embeding(inference_state, None, None, False, tracking_box.cuda(), device=device)
        else:
            embed_sam_boxes = self.model.track_embed(prompt_token).reshape(1, 3, 256)
        # import pdb; pdb.set_trace()   

        ann_frame_idx = 0
        ann_obj_id = 0
        box = np.array([0,0,0,0], dtype=np.float32)
        _, out_obj_ids, out_mask_logits = self.model.sam.add_new_box_embeding(
            inference_state=inference_state,
            frame_idx=ann_frame_idx,
            obj_id=ann_obj_id,
            box=box,
            box_embeding=embed_sam_boxes,
        )
        video_segments = {}  # video_segments contains the per-frame segmentation results
        for out_frame_idx, out_obj_ids, out_mask_logits in self.model.sam.propagate_in_video(inference_state):
            video_segments[out_frame_idx] = {
                out_obj_id: (out_mask_logits[i] > 0.0)
                for i, out_obj_id in enumerate(out_obj_ids)
            }
        video_segments = [video_segments[tt][0] for tt in video_segments]

        # if tracking_box:
        # bboxes = []
        # for video_segment in video_segments:
        #     bbox = self.model.find_boundaries_torch(video_segment.squeeze(0).cpu()).to(torch.float)
        #     bboxes.append(bbox.cpu())

        return video_segments
    
tokenizer = MultimodalLlamaTokenizer.from_pretrained(
    # "/mnt/petrelfs/share_data/yanziang/expert_tokenizer", 
    '/mnt/petrelfs/share_data/likunchang/model/llm/Mistral-7B-Instruct-v0.2',
    local_files_only=True,
    n_query=96,
    v_query=96,
    special_tokens = special_tokens, 
) 

# videochat-tpo Inference
parser = argparse.ArgumentParser()
parser = initialize(parser)
opt = parser.parse_args()


class OpenGVLab(gr.themes.base.Base):
    def __init__(
        self,
        *,
        primary_hue=colors.blue,
        secondary_hue=colors.sky,
        neutral_hue=colors.gray,
        spacing_size=sizes.spacing_md,
        radius_size=sizes.radius_sm,
        text_size=sizes.text_md,
        font=(
            fonts.GoogleFont("Noto Sans"),
            "ui-sans-serif",
            "sans-serif",
        ),
        font_mono=(
            fonts.GoogleFont("IBM Plex Mono"),
            "ui-monospace",
            "monospace",
        ),
    ):
        super().__init__(
            primary_hue=primary_hue,
            secondary_hue=secondary_hue,
            neutral_hue=neutral_hue,
            spacing_size=spacing_size,
            radius_size=radius_size,
            text_size=text_size,
            font=font,
            font_mono=font_mono,
        )
        super().set(
            body_background_fill="*neutral_50",
        )

def clear_():
    return [], []

def gradio_reset(chat_state, img_list):
    if chat_state is not None:
        chat_state = []
    if img_list is not None:
        img_list = None
    return None, gr.update(value=None, interactive=True), gr.update(placeholder='Please upload your video first', interactive=False),gr.update(value="Upload & Start Chat", interactive=True), chat_state, img_list

gvlabtheme = OpenGVLab(primary_hue=colors.blue,
        secondary_hue=colors.sky,
        neutral_hue=colors.gray,
        spacing_size=sizes.spacing_md,
        radius_size=sizes.radius_sm,
        text_size=sizes.text_md,
        )

title = """<h1 align="center"><a href="https://github.com/OpenGVLab/Ask-Anything"><img src="https://s1.ax1x.com/2023/05/07/p9dBMOU.png" alt="Ask-Anything" border="0" style="margin: 0 auto; height: 100px;" /></a> </h1>"""
description ="""
        VideoChat2 powered by InternVideo!<br><p><a href='https://github.com/OpenGVLab/Ask-Anything'><img src='https://img.shields.io/badge/Github-Code-blue'></a></p><p>
        """
SYS_PROMPT =""

with gr.Blocks(title="InternVideo-VideoChat!",theme=gvlabtheme,css="#chatbot {overflow:auto; height:500px;} #InputVideo {overflow:visible; height:320px;} footer {visibility: none}") as demo:
    gr.Markdown(title)
    gr.Markdown(description)
    inference = Inference("OpenGVLab/VideoChat-TPO", special_tokens)
    with gr.Row():
        with gr.Column(scale=0.5, visible=True) as video_upload:
            with gr.Column(elem_id="image", scale=0.5) as img_part:
                # with gr.Tab("Video", elem_id='video_tab'):
                up_video = gr.Video(interactive=True, include_audio=True, elem_id="video_upload")
                # with gr.Tab("Image", elem_id='image_tab'):
                #     up_image = gr.Image(type="pil", interactive=True, elem_id="image_upload")
            upload_button = gr.Button(value="Dummy button", interactive=True, variant="primary")
            restart = gr.Button("Restart")
            sys_prompt = gr.State(f"{SYS_PROMPT}")

            num_beams = gr.Slider(
                minimum=1,
                maximum=10,
                value=1,
                step=1,
                interactive=True,
                                                                 label="beam search numbers)",
            )

            temperature = gr.Slider(
                minimum=0.1,
                maximum=2.0,
                value=1.0,
                step=0.1,
                interactive=True,label="Temperature",
            )

            num_segments = gr.Slider(
                minimum=8,
                maximum=64,
                value=8,
                step=1,
                interactive=True,
                label="Input Frames",
            )

            resolution = gr.Slider(
                minimum=224,
                maximum=224,
                value=224,
                step=1,
                interactive=True,
                label="Vision encoder resolution",
            )

            hd_num = gr.Slider(
                minimum=1,
                maximum=10,
                value=4,
                step=1,
                interactive=True,
                label="HD num",
            )

            padding = gr.Checkbox(
                label="padding",
                info=""
            )

        with gr.Column(visible=True)  as input_raws:
            chat_state = gr.State([])
            img_list = gr.State()
            chatbot = gr.Chatbot(elem_id="chatbot",label='VideoChat')
            with gr.Row():
                with gr.Column(scale=0.7):
                    text_input = gr.Textbox(show_label=False, placeholder='Please upload your video first', interactive=True)
                with gr.Column(scale=0.15, min_width=0):
                    run = gr.Button("💭Send")
                with gr.Column(scale=0.15, min_width=0):
                    clear = gr.Button("🔄Clear️")
            generate_video = gr.Video(interactive=False, include_audio=True, elem_id="video_output")
            generate_image = gr.Image(type="pil", interactive=False, elem_id="image_output")

    text_input.submit(inference.run, [text_input, up_video, chatbot], [chatbot,  generate_image, generate_video]).then(lambda: "", None, text_input)
    run.click(inference.run, [text_input, up_video, chatbot], [chatbot,  generate_image, generate_video]).then(lambda: "", None, text_input)

    clear.click(clear_, None, [chatbot, chat_state])
    restart.click(gradio_reset, [chat_state, img_list], [chatbot,  up_video, text_input, upload_button, chat_state, img_list], queue=False)

demo.launch(server_name='0.0.0.0')