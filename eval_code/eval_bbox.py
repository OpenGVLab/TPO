from src.tokenizer.multimodal_llama_tokenizer import MultimodalLlamaTokenizer
from src.dataset.it_dataset_mistral_new import ITTrainDataset_mistral_new
from src.train.trainer import collate_hd
# tokenizer = MultimodalLlamaTokenizer.from_pretrained("/mnt/petrelfs/wangchenting/Mistral-7B-Instruct-v0.2", local_files_only=True) 
from src.share_utils.special_token import special_tokens
tokenizer = MultimodalLlamaTokenizer.from_pretrained(
    "/mnt/petrelfs/share_data/likunchang/model/llm/Mistral-7B-Instruct-v0.2", 
    local_files_only=True,
    n_query=96,
    v_query=96,
    special_tokens = special_tokens, 
) 
from src.share_utils.my_easydict import MyEasyDict as edict
from torchvision import transforms
from src.dataset.create import get_train_transform, create_dataset, create_loader, create_sampler
from src.model.multimodalllm_pt import MultiModalLLM_PT
from torch.utils.data import DataLoader
from omegaconf import OmegaConf
from src.share_utils.py_config_utils import Config, eval_dict_leaf

anno_root_it = "/mnt/petrelfs/share_data/videointern/annotations/anno_instruction/videochat_new"
import torch
# torch.backends.cudnn.enabled = False

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


from src.model.base_model import LLMConfig
config = LLMConfig.from_pretrained('/mnt/petrelfs/share_data/likunchang/model/llm/Mistral-7B-Instruct-v0.2', trust_remote_code=True)
cg_opt = initialize_parser()
model_cfg = eval_dict_leaf(Config.from_file('/mnt/petrelfs/yanziang/trackingllm/videoagent/configs/model/expert_umtl_f8_s2.py'))
# model_cfg = eval_dict_leaf(Config.from_file('/mnt/petrelfs/yanziang/trackingllm/videoagent/configs/model/i1b_qformer_mistral7b_f8.py'))
config.model_config = model_cfg.model
config.cg_opt = cg_opt.parse_args()
config.model_tokenizer = tokenizer

from src.share_utils.distributed import get_world_size, get_rank, gather_together
from src.dataset.dataloader import MetaLoader
from src.dataset.create import get_media_types
from src.dataset.create import create_dataset
num_gpu = 8
import os
os.environ['CUDA_VISIBLE_DEVICES'] = str(int(os.environ['SLURM_PROCID'])%num_gpu)
gpu_id = int(os.environ['SLURM_PROCID'])%num_gpu
anno_root_it = "/mnt/petrelfs/share_data/videointern/annotations/anno_instruction/videochat_new"
from configs.dataset.data_config.available_corpus import available_corpus
val_list = [
    edict(
        anno_path = '/mnt/petrelfs/share_data/yanziang/video_agent_data/additional/REC_refcoco+_unc_val.json',
        data_root="pnorm2:s3://coco-caption/train2014/",
        media_type='REC',
    ),
    edict(
        anno_path = '/mnt/petrelfs/share_data/yanziang/video_agent_data/additional/REC_refcoco_unc_val.json',
        data_root="pnorm2:s3://coco-caption/train2014/",
        media_type='REC',
    ),
    edict(
        anno_path = '/mnt/petrelfs/share_data/yanziang/video_agent_data/additional/REC_refcocog_umd_test.json',
        data_root="pnorm2:s3://coco-caption/train2014/",
        media_type='REC',
    ),
    edict(
        anno_path = '/mnt/petrelfs/share_data/yanziang/video_agent_data/additional/REC_refcoco_unc_testB.json',
        data_root="pnorm2:s3://coco-caption/train2014/",
        media_type='REC',
    ),
    edict(
        anno_path = '/mnt/petrelfs/share_data/yanziang/video_agent_data/additional/REC_refcoco_unc_testA.json',
        data_root="pnorm2:s3://coco-caption/train2014/",
        media_type='REC',
    ),
    edict(
        anno_path = '/mnt/petrelfs/share_data/yanziang/video_agent_data/additional/REC_refcoco+_unc_testB.json',
        data_root="pnorm2:s3://coco-caption/train2014/",
        media_type='REC',
    ),
    edict(
        anno_path = '/mnt/petrelfs/share_data/yanziang/video_agent_data/additional/REC_refcoco+_unc_testA.json',
        data_root="pnorm2:s3://coco-caption/train2014/",
        media_type='REC',
    ),
    edict(
        anno_path = '/mnt/petrelfs/share_data/yanziang/video_agent_data/additional/REC_refcocog_umd_val.json',
        data_root="pnorm2:s3://coco-caption/train2014/",
        media_type='REC',
    ),
]
    
from src.dataset.it_dataset_mistral_eval import ITTrainDataset_mistral_eval
Myd = ITTrainDataset_mistral_eval(
    ann_file = val_list[gpu_id],
    transform=get_train_transform(config='', train_file='',mode='padding'),
    tokenizer=tokenizer
)
dataloader = DataLoader(
    Myd,
    batch_size=1,
    shuffle=False,
    num_workers=1,
    pin_memory=True,
    collate_fn=collate_hd,
)

# model_cfg = eval_dict_leaf(Config.from_file('configs/model/expert_umtl_f8_s2.py'))

# from src.model.base_model import LLMConfig
# config = LLMConfig.from_pretrained('/mnt/petrelfs/share_data/likunchang/model/llm/Mistral-7B-Instruct-v0.2', trust_remote_code=True)
# config.model_config = model_cfg.model
# config.model_tokenizer = tokenizer
# path = '/mnt/petrelfs/yanziang/trackingllm/videoagent/logs/scripts/pt/expert_tune/bbox_s2.sh_20240813_005401/checkpoint-10000'
# path = '/mnt/petrelfs/yanziang/trackingllm/videoagent/logs/scripts/pt/expert_tune/bbox_s2.sh_20240813_224153/checkpoint-50000'
# path = '/mnt/petrelfs/yanziang/trackingllm/videoagent/logs/scripts/pt/expert_tune/bbox_s2.sh_20240814_232634/checkpoint-40000'
# path = '/mnt/petrelfs/yanziang/trackingllm/videoagent/logs/scripts/pt/expert_tune/bbox_s2.sh_20240815_001103/checkpoint-30000'
# path = '/mnt/petrelfs/yanziang/trackingllm/videoagent/logs/scripts/pt/expert_tune/bbox_s2.sh_20240816_140859/checkpoint-45000'
# path = '/mnt/petrelfs/yanziang/trackingllm/videoagent/logs/scripts/pt/expert_tune/bbox_s2.sh_20240818_234625/checkpoint-40000'
# path = '/mnt/petrelfs/yanziang/trackingllm/videoagent/logs/scripts/pt/expert_tune/bbox_s2.sh_20240818_234625/checkpoint-last'
# path = '/mnt/petrelfs/yanziang/trackingllm/videoagent/logs/scripts/pt/expert_tune/bbox_s2.sh_20240820_001320/checkpoint-last'
# path = '/mnt/petrelfs/yanziang/trackingllm/videoagent/logs/scripts/pt/expert_tune/test_grounding.sh_20240927_101411/checkpoint-last'
path = '/mnt/petrelfs/yanziang/trackingllm/videoagent/logs/scripts/pt/expert_tune/TVG_s2.sh_20241001_151247/checkpoint-last'
path = '/mnt/petrelfs/yanziang/trackingllm/videoagent/logs/scripts/pt/expert_tune/TVG_s2.sh_20241002_215104/checkpoint-80000'
path = '/mnt/petrelfs/yanziang/trackingllm/videoagent/logs/scripts/pt/expert_tune/TVG_s2.sh_20241002_215104/checkpoint-last'
path = '/mnt/petrelfs/yanziang/trackingllm/videoagent/logs/scripts/pt/expert_tune/TVG_s2.sh_20241010_004436/checkpoint-last'
path = '/mnt/petrelfs/yanziang/trackingllm/videoagent/logs/scripts/pt/expert_tune/TVG_s2.sh_20241007_162703/checkpoint-last'
path = '/mnt/petrelfs/yanziang/trackingllm/videoagent/logs/scripts/pt/expert_tune/TVG_s2.sh_20241016_102651/checkpoint-100000'
path = '/mnt/petrelfs/yanziang/trackingllm/videoagent/logs/scripts/pt/expert_tune/TVG_s2.sh_20241018_225600/checkpoint-last'
path = '/mnt/petrelfs/share_data/yanziang/reasoning_chat2'
# path = '/mnt/petrelfs/share_data/yanziang/merge_all'
model = MultiModalLLM_PT.from_pretrained(path, config=config, local_files_only=True)
device = f'cuda:{gpu_id}'
model = model.to(device)
# loc_decoder.0.bias
import pdb
def generate(input_ids, attention_mask, labels, image, image_idx):
    prompt = input_ids
    l_prompt = len(input_ids)
    temperature = 1e-5
    max_new_tokens = 20
    guide_w = 5
    stop_str = '</s>'
    bbox = []
    output_ids = list(input_ids[0])
    text_embeds = model.pad_text_embeds(input_ids=input_ids, image=image, video=None, image_idx=image_idx, video_idx=None)
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
            # if token != -1 and tokenizer.decode([token])[0] == '.':
            #     print(torch.softmax(last_token_logits, dim=-1)[32004], torch.softmax(last_token_logits, dim=-1).max())
            if temperature < 1e-4:
                token = int(torch.argmax(last_token_logits))
            else:
                probs = torch.softmax(last_token_logits / temperature, dim=-1)
                token = int(torch.multinomial(probs, num_samples=1))
            output_ids.append(token)
        ret = tokenizer.decode(token)
        # print(ret)
        if ret == '<box_begin>':
            attention_mask = torch.ones(1, past_key_values[0][0].shape[-2] + 1, device=device)
            bbox_embeds = model.box_token.bfloat16()
            # text_embeds_bbox = torch.cat((text_embeds.squeeze(0), bbox_embeds), dim=0).unsqueeze(0)
            
            out = model.lm(
                inputs_embeds=bbox_embeds,
                use_cache=True,
                attention_mask=attention_mask,
                output_hidden_states=True,
                past_key_values=past_key_values
            )
            last_hidden_states = out.hidden_states[-1]
            # selected_hidden_states = last_hidden_states[0][len(attention_mask)-2]
            selected_hidden_states = last_hidden_states[0][0]
            # print(f'{}')
            bbox.append(model.loc_decoder(selected_hidden_states))
            # pdb.set_trace()
            last_token_logits = logits[0][-1]
            if temperature < 1e-4:
                token = int(torch.argmax(last_token_logits))
            else:
                probs = torch.softmax(last_token_logits / temperature, dim=-1)
                token = int(torch.multinomial(probs, num_samples=1))
            # print(token)

        if (ret == '</s>'):
            break
    ret = tokenizer.decode(output_ids)
    del past_key_values
    return ret, bbox
i = 0
# print(len(dataloader))
all = []
from tqdm import tqdm
import json
file = f'/mnt/petrelfs/yanziang/trackingllm/videoagent/logs/results/merge_new/visual_grounding/{gpu_id}.jsonl'

for dd in tqdm(dataloader):
    try:
        ret, bbox = generate(dd['input_ids'].to(device),dd['attention_mask'].to(device), dd['labels'].to(device), dd['image'].to(device), dd['image_idx'].to(device))
        # all.append({'gt':dd['boxes'], 'pred':bbox, 'ret':ret})
        tt = {'gt':dd['output_boxes'][0].float().numpy().tolist(), 'pred':bbox[0].float().cpu().detach().numpy().tolist(), 'ret':ret, 'path' :dd['path']}
        with open(file, 'a') as f:
            json_record = json.dumps(tt)
            f.write(json_record + '\n')
    except Exception as e:
        tt = {'gt':dd['output_boxes'][0].float().numpy().tolist(), 'pred':[-1,-1,-1,-1], 'ret':ret}
        with open(file, 'a') as f:
            json_record = json.dumps(tt)
            f.write(json_record + '\n')
        print(e)