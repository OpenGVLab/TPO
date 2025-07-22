import transformers
import hydra
import pyrootutils
import logging
import torch
import torch.distributed as dist
import wandb
import io
import numpy as np
import os
from petrel_client.client import Client
from src.model import *
from src.train.trainer import CustomTrainer
from typing import Optional, List
from dataclasses import dataclass, field
from omegaconf import OmegaConf
from src.share_utils.py_config_utils import Config, eval_dict_leaf
from src.train.cap_utils import compute_metrics_cider
from src.train.qa_utils import compute_metrics_vqa
from src.share_utils.distributed import is_main_process
from src.share_utils.logger import setup_logger
from src.train.utils import init_distributed_mode, setup_output_dir
from src.share_utils.serialize import local_broadcast_process_authkey
import deepspeed
from src.tokenizer import MultimodalLlamaTokenizer
from src.share_utils.special_token import special_tokens
from src.model.base_model import LLMConfig
from src.dataset.it_dataset_mistral_eval import ITTrainDataset_mistral_eval
from src.dataset.it_dataset_mistral_new import ITTrainDataset_mistral_new
from torch.utils.data import DataLoader
from src.share_utils.my_easydict import MyEasyDict as edict
from src.dataset.create import get_train_transform
from src.train.trainer import collate_hd
import pycocotools.mask as mask_util

import argparse
def initialize_parser():
    parser = argparse.ArgumentParser()

    # Arguments
    parser.add_argument("--dset_name", type=str, default='charadesSTA', choices=["hl", 'charadesSTA'])
    parser.add_argument("--eval_split_name", type=str, default="val",
                        help="should match keys in video_duration_idx_path, must set for VCMR")
    parser.add_argument("--data_ratio", type=float, default=1.0,
                        help="how many training and eval data to use. 1.0: use all, 0.1: use 10%.")
    parser.add_argument("--results_root", type=str, default="results")
    parser.add_argument("--exp_id", type=str, default=None, help="id of this run, required at training")
    parser.add_argument("--max_es_cnt", type=int, default=200, help="number of epochs to early stop")
    parser.add_argument("--eval_epoch", type=int, default=5, help="inference epoch")
    parser.add_argument("--grad_clip", type=float, default=0.1, help="perform gradient clip, -1: disable")
    parser.add_argument("--eval_untrained", action="store_true", help="Evaluate on un-trained model")
    parser.add_argument("--resume_all", action="store_true", help="if --resume_all, load optimizer/scheduler/epoch as well")
    parser.add_argument("--start_epoch", type=int, default=None, help="if None, will be set automatically when using --resume_all")

    # Data config
    parser.add_argument("--max_q_l", type=int, default=-1)
    parser.add_argument("--max_v_l", type=int, default=-1)
    parser.add_argument("--clip_length", type=float, default=1.0)
    parser.add_argument("--max_windows", type=int, default=5)
    parser.add_argument("--train_path", type=str, default=None)
    parser.add_argument("--eval_path", type=str, default=None, help="Evaluating during training")
    parser.add_argument("--no_norm_vfeat", action="store_true", help="Do not normalize video feat")
    parser.add_argument("--no_norm_tfeat", action="store_true", help="Do not normalize text feat")
    parser.add_argument("--v_feat_dirs", type=str, nargs="+", help="video feature dirs")
    parser.add_argument("--t_feat_dir", type=str, help="text/query feature dir")
    parser.add_argument("--v_feat_dim", type=int, default=770, help="video feature dim")
    parser.add_argument("--t_feat_dim", type=int, default=4096, help="text/query feature dim")
    parser.add_argument("--ctx_mode", type=str, default="video_tef")

    # Model config
    parser.add_argument('--position_embedding', default='sine', type=str, choices=('sine', 'learned'),
                        help="Type of positional embedding to use")
    parser.add_argument('--enc_layers', default=3, type=int, help="Number of encoding layers in the transformer")
    parser.add_argument('--dec_layers', default=3, type=int, help="Number of decoding layers in the transformer")
    parser.add_argument('--t2v_layers', default=2, type=int, help="Number of decoding layers in the transformer")
    parser.add_argument('--sent_layers', default=1, type=int, help="Number of decoding layers in the transformer")
    parser.add_argument('--moment_layers', default=1, type=int, help="Number of decoding layers in the transformer")
    parser.add_argument('--dummy_layers', default=2, type=int, help="Number of encoding layers in the transformer")
    parser.add_argument('--dim_feedforward', default=1024, type=int, help="Intermediate size of the feedforward layers")
    parser.add_argument('--hidden_dim', default=256, type=int, help="Size of the embeddings")
    parser.add_argument('--input_dropout', default=0.5, type=float, help="Dropout applied in input")
    parser.add_argument('--dropout', default=0.1, type=float, help="Dropout applied in the transformer")
    parser.add_argument("--txt_drop_ratio", default=0, type=float, help="drop tokens from text input")
    parser.add_argument("--use_txt_pos", action="store_true", help="use position_embedding for text as well.")
    parser.add_argument('--nheads', default=8, type=int, help="Number of attention heads inside the transformer")
    parser.add_argument('--num_queries', default=10, type=int, help="Number of query slots")
    parser.add_argument('--num_dummies', default=45, type=int, help="Number of dummy tokens")
    parser.add_argument('--total_prompts', default=10, type=int, help="Number of query slots")
    parser.add_argument('--num_prompts', default=1, type=int, help="Number of dummy tokens")
    parser.add_argument('--pre_norm', action='store_true')
    parser.add_argument("--n_input_proj", type=int, default=2, help="#layers to encoder input")
    parser.add_argument("--contrastive_hdim", type=int, default=64, help="dim for contrastive embeddings")
    parser.add_argument("--temperature", type=float, default=0.07, help="temperature for contrastive loss")

    # Loss
    parser.add_argument("--saliency_margin", type=float, default=0.2)
    parser.add_argument('--no_aux_loss', dest='aux_loss', action='store_false', help="Disables auxiliary decoding losses")
    parser.add_argument("--span_loss_type", default="l1", type=str, choices=['l1', 'ce'],
                        help="Loss type for span")
    parser.add_argument("--contrastive_align_loss", action="store_true",
                        help="Disable contrastive alignment loss")
    parser.add_argument('--set_cost_span', default=10, type=float, help="L1 span coefficient in the matching cost")
    parser.add_argument('--set_cost_giou', default=1, type=float, help="GIoU span coefficient in the matching cost")
    parser.add_argument('--set_cost_class', default=4, type=float, help="Class coefficient in the matching cost")

    # Loss coefficients
    parser.add_argument("--lw_saliency", type=float, default=4, help="weight for saliency loss")
    parser.add_argument("--lw_wattn", type=float, default=1., help="weight for attention loss")
    parser.add_argument("--lw_ms_align", type=float, default=1., help="weight for multi-scale alignment loss")
    parser.add_argument("--lw_distill", type=float, default=1., help="weight for distillation loss")
    parser.add_argument('--span_loss_coef', default=10, type=float)
    parser.add_argument('--giou_loss_coef', default=1, type=float)
    parser.add_argument('--label_loss_coef', default=4, type=float)
    parser.add_argument('--eos_coef', default=0.1, type=float,
                        help="Relative classification weight of the no-object class")
    parser.add_argument("--contrastive_align_loss_coef", default=0.02, type=float)

    parser.add_argument("--no_sort_results", action="store_true", help="do not sort results")
    parser.add_argument("--max_before_nms", type=int, default=10)
    parser.add_argument("--max_after_nms", type=int, default=10)
    parser.add_argument("--conf_thd", type=float, default=0.0, help="only keep windows with conf >= conf_thd")
    parser.add_argument("--nms_thd", type=float, default=-1,
                        help="additional use of non-maximum suppression")

    return parser

tokenizer = MultimodalLlamaTokenizer.from_pretrained(
    "/mnt/petrelfs/share_data/likunchang/model/llm/Mistral-7B-Instruct-v0.2", 
    local_files_only=True,
    n_query=96,
    v_query=96,
    special_tokens = special_tokens, 
) 
# print(cfg_path.model)
model_cfg = eval_dict_leaf(Config.from_file('/mnt/petrelfs/yanziang/trackingllm/videoagent/configs/model/expert_umtl_f8_s2.py'))

from src.model.base_model import LLMConfig
# _, _, cg_opt = parser.parse_args_into_dataclasses(parser)
cg_opt = initialize_parser()
config = LLMConfig.from_pretrained('/mnt/petrelfs/share_data/likunchang/model/llm/Mistral-7B-Instruct-v0.2', trust_remote_code=True)
config.model_config = model_cfg.model
# parser
config.cg_opt = cg_opt.parse_args()
config.model_tokenizer = tokenizer
# path = '/mnt/petrelfs/yanziang/trackingllm/videoagent/logs/scripts/pt/expert_tune/bbox_s2.sh_20240912_110740/checkpoint-3000'
# path = '/mnt/petrelfs/yanziang/trackingllm/videoagent/logs/scripts/pt/expert_tune/bbox_s2.sh_20240912_110740/checkpoint-last'
# path = '/mnt/petrelfs/yanziang/trackingllm/videoagent/logs/scripts/pt/expert_tune/bbox_s2.sh_20240912_220743/checkpoint-last'
# path = '/mnt/petrelfs/yanziang/trackingllm/videoagent/logs/scripts/pt/expert_tune/bbox_s2.sh_20240913_161108/checkpoint-last'
# path = '/mnt/petrelfs/yanziang/trackingllm/videoagent/logs/scripts/pt/expert_tune/bbox_s2.sh_20240914_003058/checkpoint-15000'
# path = '/mnt/petrelfs/yanziang/trackingllm/videoagent/logs/scripts/pt/expert_tune/bbox_s2.sh_20240914_003058/checkpoint-last'
# path = '/mnt/petrelfs/yanziang/trackingllm/videoagent/logs/scripts/pt/expert_tune/bbox_s2.sh_20240914_181722/checkpoint-last'
# path = '/mnt/petrelfs/yanziang/trackingllm/videoagent/logs/scripts/pt/expert_tune/test_grounding.sh_20240924_120231/checkpoint-last'
# path = '/mnt/petrelfs/yanziang/trackingllm/videoagent/logs/scripts/pt/expert_tune/test_grounding.sh_20240926_105157/checkpoint-15000'
# path = '/mnt/petrelfs/yanziang/trackingllm/videoagent/logs/scripts/pt/expert_tune/TVG_s2.sh_20241001_143316/checkpoint-last'
# path = '/mnt/petrelfs/yanziang/trackingllm/videoagent/logs/scripts/pt/expert_tune/TVG_s2.sh_20241004_004032/checkpoint-last'
# path = '/mnt/petrelfs/yanziang/trackingllm/videoagent/logs/scripts/pt/expert_tune/TVG_s2.sh_20241004_111411/checkpoint-last'
# path = '/mnt/petrelfs/yanziang/trackingllm/videoagent/logs/scripts/pt/expert_tune/TVG_s2.sh_20241010_004436/checkpoint-last'
# path = '/mnt/petrelfs/yanziang/trackingllm/videoagent/logs/scripts/pt/expert_tune/TVG_s2.sh_20241016_102651/checkpoint-100000'
# path = '/mnt/petrelfs/yanziang/trackingllm/videoagent/logs/scripts/pt/expert_tune/TVG_s2.sh_20241018_225600/checkpoint-last'
path = '/mnt/petrelfs/share_data/yanziang/reasoning_chat'
model = MultiModalLLM_PT.from_pretrained(path, config=config, local_files_only=True)
model = model.to(dtype=torch.bfloat16)
model = model.cuda()

Myd = ITTrainDataset_mistral_eval(
    ann_file = edict(
        anno_path=f"/mnt/petrelfs/yanziang/trackingllm/videoagent/davis17.json", 
        data_root="shdd:s3://",
        media_type='tracking2',
    ),
    transform=get_train_transform(config='', train_file='',mode='resize_noflip'),
    tokenizer=tokenizer,
    num_frames = 16
)


dataloader = DataLoader(
    Myd,
    batch_size=1,
    shuffle=False,
    num_workers=1,
    pin_memory=True,
    collate_fn=collate_hd,
)

def generate(input_ids, attention_mask, labels, image, image_idx, video, video_idx, input_boxes, size_hw, sam_images):
    device = input_ids.device
    prompt = input_ids
    l_prompt = len(input_ids)
    temperature = 1e-5
    max_new_tokens = 20
    guide_w = 5
    stop_str = '</s>'
    bbox = []
    output_ids = list(input_ids[0])
    input_boxes = [box.cuda() for box in input_boxes]
    text_embeds = model.pad_text_embeds(input_ids=input_ids, image=image, video=video, image_idx=image_idx, video_idx=video_idx, return_visual=False, 
                                       instruction = None, output_boxes=None, input_boxes=input_boxes)
    for i in range(max_new_tokens):
        if i == 0:
            outputs = model.lm(
                    inputs_embeds=text_embeds,
                    attention_mask=attention_mask,
                    output_hidden_states=True,
                    return_dict=True,
                )
            logits = outputs.logits
            past_key_values = outputs.past_key_values
        else:
            attention_mask = torch.ones(1, past_key_values[0][0].shape[-2] + 1, device=device)
            last_text_embeds = model.lm.get_input_embeddings()(torch.tensor(output_ids[-1], device=device).long()).detach().unsqueeze(0)
            last_text_embeds = last_text_embeds.unsqueeze(0)
            
            out = model.lm(
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
        ret = tokenizer.decode(token)
        if ret == '<box_begin>':
            attention_mask = torch.ones(1, past_key_values[0][0].shape[-2] + 1, device=device)
            bbox_embeds = model.box_token.bfloat16()
            out = model.lm(
                inputs_embeds=bbox_embeds,
                use_cache=True,
                attention_mask=attention_mask,
                output_hidden_states=True,
                past_key_values=past_key_values
            )
            last_hidden_states = out.hidden_states[-1]
            selected_hidden_states = last_hidden_states[0][0]
            bbox.append(model.loc_decoder(selected_hidden_states))
            last_token_logits = logits[0][-1]
            if temperature < 1e-4:
                token = int(torch.argmax(last_token_logits))
            else:
                probs = torch.softmax(last_token_logits / temperature, dim=-1)
                token = int(torch.multinomial(probs, num_samples=1))
        if ret == '<track_begin>':
            attention_mask = torch.ones(1, past_key_values[0][0].shape[-2] + 1, device=device)
            tracking_embeds = model.track_token
            out = model.lm(
                inputs_embeds=tracking_embeds,
                use_cache=True,
                attention_mask=attention_mask,
                output_hidden_states=True,
                past_key_values=past_key_values
            )
            last_hidden_states = out.hidden_states[-1]
            selected_hidden_states = last_hidden_states[0][0].to(dtype = torch.bfloat16)
            
            embed_sam_boxes = model.track_embed(selected_hidden_states).reshape(1, 3, 256)
            
            inference_state = model.sam.init_state_images(sam_images, size_hw[0][0], size_hw[0][1])
            gt_embeds = model.sam.get_prompt_embeding(inference_state, None, None, False, input_boxes[0].cuda(), device = text_embeds.device) 
            ann_frame_idx = 0
            ann_obj_id = 0
            box = np.array([0, 0, 0, 0], dtype=np.float32)
            _, out_obj_ids, out_mask_logits = model.sam.add_new_box_embeding(
                inference_state=inference_state,
                frame_idx=ann_frame_idx,
                obj_id=ann_obj_id,
                box=box,
                box_embeding=embed_sam_boxes,
            )
            video_segments = {}  # video_segments contains the per-frame segmentation results
            for out_frame_idx, out_obj_ids, out_mask_logits in model.sam.propagate_in_video(inference_state):
                video_segments[out_frame_idx] = {
                    out_obj_id: (out_mask_logits[i] > 0.0)
                    for i, out_obj_id in enumerate(out_obj_ids)
                }
            video_segments = [video_segments[tt][0] for tt in video_segments]
            # bbox = model.find_boundaries_torch(video_segments[0].squeeze(0).cpu())
            return ret, [], video_segments

        if (ret == '</s>'):
            break
    ret = tokenizer.decode(output_ids)
    del past_key_values
    return ret, bbox, video_segments
# dict_keys(['input_ids', 'attention_mask', 'labels', 'instruction', 'video', 'video_idx', 'sec', 'vlen', 'frame_indices', 'fps', 'input_boxes', 'gt_masks', 'sam_images', 'size_hw'])

import json
from tqdm import tqdm
for idx, dd in tqdm(enumerate(dataloader)):
    ret, bbox, video_segments = generate(dd['input_ids'].cuda(), dd['attention_mask'].cuda(), dd['labels'].cuda(),None, None, dd['video'].to(dtype=torch.bfloat16).cuda(), dd['video_idx'].cuda(), dd['input_boxes'], dd['size_hw'], dd['sam_images'].cuda().to(dtype=torch.bfloat16))
    output_file = "/mnt/petrelfs/yanziang/trackingllm/videoagent/logs/results/merge_new/seg/predictions7.jsonl"
    prediction = {"video": dd['path'][0], 'mask_path':dd['mask_path'][0], 
                    "pred": str(mask_util.encode(np.asfortranarray(np.array(torch.stack(video_segments, dim=-1).squeeze(0).cpu()))))
                    }
    with open(output_file, 'a') as f:
        json.dump(prediction, f)
        f.write('\n')