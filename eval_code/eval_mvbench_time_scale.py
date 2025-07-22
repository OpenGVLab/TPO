# from dataset.video_transforms import (
#     GroupNormalize, GroupScale, GroupCenterCrop, 
#     Stack, ToTorchFormatTensor
# )
from src.dataset.videoload import (
    GroupNormalize, GroupScale, GroupCenterCrop, 
    Stack, ToTorchFormatTensor
)
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


anno_root_it = "/mnt/petrelfs/share_data/videointern/annotations/anno_instruction/videochat_new"
import torch

from src.model.base_model import LLMConfig
config = LLMConfig.from_pretrained('/mnt/petrelfs/share_data/likunchang/model/llm/Mistral-7B-Instruct-v0.2', trust_remote_code=True)
cg_opt = initialize_parser()
model_cfg = eval_dict_leaf(Config.from_file('/mnt/petrelfs/yanziang/trackingllm/videoagent/configs/model/expert_umtl_f8_s2.py'))
# model_cfg = eval_dict_leaf(Config.from_file('/mnt/petrelfs/yanziang/trackingllm/videoagent/configs/model/i1b_qformer_mistral7b_f8.py'))
config.model_config = model_cfg.model
config.cg_opt = cg_opt.parse_args()
config.model_tokenizer = tokenizer

path = '/mnt/petrelfs/share_data/yanziang/all'
# path = '/mnt/petrelfs/yanziang/trackingllm/videoagent/logs/scripts/pt/expert_tune/TVG_s2.sh_20241007_115334/checkpoint-last'
# path = '/mnt/petrelfs/yanziang/trackingllm/videoagent/logs/scripts/pt/expert_tune/TVG_s2.sh_20241010_004436/checkpoint-120000'
# path = '/mnt/petrelfs/yanziang/trackingllm/videoagent/logs/scripts/pt/expert_tune/TVG_s2.sh_20241010_004436/checkpoint-last'
# path = '/mnt/petrelfs/yanziang/trackingllm/videoagent/logs/scripts/pt/expert_tune/TVG_s2.sh_20241013_101429/checkpoint-80000'
# path = '/mnt/petrelfs/yanziang/trackingllm/videoagent/logs/scripts/pt/expert_tune/TVG_s2.sh_20241016_115611/checkpoint-60000'
# path= '/mnt/petrelfs/yanziang/trackingllm/videoagent/logs/scripts/pt/expert_tune/TVG_s2.sh_20241016_102651/checkpoint-100000'
# path = '/mnt/petrelfs/yanziang/trackingllm/videoagent/logs/scripts/pt/expert_tune/TVG_s2.sh_20241018_225600/checkpoint-100000'
# path = '/mnt/petrelfs/yanziang/trackingllm/videoagent/logs/scripts/pt/expert_tune/TVG_s2.sh_20241016_102651/checkpoint-60000'
# path = '/mnt/petrelfs/yanziang/trackingllm/videoagent/logs/scripts/pt/expert_tune/TVG_s2.sh_20241007_162703/checkpoint-last'
# path = '/mnt/petrelfs/yanziang/trackingllm/videoagent/logs/scripts/pt/expert_tune/TVG_s2.sh_20241020_170132/checkpoint-last'
# path = '/mnt/petrelfs/yanziang/trackingllm/videoagent/logs/scripts/pt/expert_tune/decoder_to_qa/checkpoint-last'
# path = '/mnt/petrelfs/yanziang/trackingllm/videoagent/logs/scripts/pt/expert_tune/TVG_s2.sh_20241023_155748/checkpoint-last'
# path = '/mnt/petrelfs/yanziang/trackingllm/videoagent/logs/scripts/pt/expert_tune/TVG_s2.sh_20241024_135401/checkpoint-last'
# path = '/mnt/petrelfs/yanziang/trackingllm/videoagent/logs/scripts/pt/expert_tune/TVG_s2.sh_20241025_224615/checkpoint-last'
# path = '/mnt/petrelfs/share_data/yanziang/cg_to_mlp'
path = '/mnt/petrelfs/share_data/yanziang/reasoning_chat2'
model = MultiModalLLM_PT.from_pretrained(path, config=config, local_files_only=True, ignore_mismatched_sizes=True)
device = f'cuda:{0}'
model = model.to(device)
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
            model.tokenizer(
                seg, return_tensors="pt", add_special_tokens=i == 0).to("cuda:0").input_ids
            # only add bos to the first seg
            for i, seg in enumerate(prompt_segs)
        ]
        seg_embs = [model.lm.base_model.model.model.embed_tokens(seg_t) for seg_t in seg_tokens]
    mixed_embs = [emb for pair in zip(seg_embs[:-1], img_list) for emb in pair] + [seg_embs[-1]]
    mixed_embs = torch.cat(mixed_embs, dim=1)
    return mixed_embs


def ask(text, conv):
    conv.messages.append([conv.roles[0], text])

from transformers import StoppingCriteria, StoppingCriteriaList

class StoppingCriteriaSub(StoppingCriteria):
    def __init__(self, stops=[], encounters=1):
        super().__init__()
        self.stops = stops
    def __call__(self, input_ids: torch.LongTensor, scores: torch.FloatTensor):
        for stop in self.stops:
            if torch.all((stop == input_ids[0][-len(stop):])).item():
                return True
        return False


from transformers import StoppingCriteria, StoppingCriteriaList

    
def answer(conv, model, img_list, do_sample=True, max_new_tokens=200, num_beams=1, min_length=1, top_p=0.9,
               repetition_penalty=1.0, length_penalty=1, temperature=0.1, answer_prompt=None, print_res=False):
    stop_words_ids = [
        torch.tensor([2]).to("cuda:0"),
        torch.tensor([29871, 2]).to("cuda:0")]  # '</s>' can be encoded in two different ways.
    stopping_criteria = StoppingCriteriaList([StoppingCriteriaSub(stops=stop_words_ids)])
    
    conv.messages.append([conv.roles[1], answer_prompt])
    embs = get_context_emb(conv, model, img_list, answer_prompt=answer_prompt, print_res=print_res)
    with torch.no_grad():
        outputs = model.lm.generate(
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
    output_text = model.tokenizer.decode(output_token, add_special_tokens=False)
    output_text = output_text.split('</s>')[0]  # remove the stop sign </s>
#     output_text = output_text.split('[/INST]')[-1].strip()
    conv.messages[-1][1] = output_text + '</s>'
    return output_text, output_token.cpu().numpy()
import numpy as np
def get_index(num_frames, num_segments):
    seg_size = float(num_frames - 1) / num_segments
    start = int(seg_size / 2)
    offsets = np.array([
        start + int(np.round(seg_size * idx)) for idx in range(num_segments)
    ])
    return offsets

from torch.utils.data import Dataset
from torchvision.transforms.functional import InterpolationMode
from decord import VideoReader, cpu
import torchvision.transforms as T
from PIL import Image
def load_video(video_path, num_segments=8, return_msg=False, resolution=224):
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

def get_sinusoid_encoding_table(n_position=784, d_hid=1024, cur_frame=8, ckpt_num_frame=4, pre_n_position=784): 
    ''' Sinusoid position encoding table ''' 
    # TODO: make it with torch instead of numpy 
    def get_position_angle_vec(position): 
        return [position / np.power(10000, 2 * (hid_j // 2) / d_hid) for hid_j in range(d_hid)] 
    
    # generate checkpoint position embedding
    sinusoid_table = np.array([get_position_angle_vec(pos_i) for pos_i in range(pre_n_position)]) 
    sinusoid_table[:, 0::2] = np.sin(sinusoid_table[:, 0::2]) # dim 2i 
    sinusoid_table[:, 1::2] = np.cos(sinusoid_table[:, 1::2]) # dim 2i+1 
    sinusoid_table = torch.tensor(sinusoid_table, dtype=torch.float, requires_grad=False).unsqueeze(0)
    
    print(f"n_position: {n_position}")
    print(f"pre_n_position: {pre_n_position}")
    
    if n_position != pre_n_position:
        T = ckpt_num_frame # checkpoint frame
        P = 14 # checkpoint size
        C = d_hid
        new_P = int((n_position // cur_frame) ** 0.5) # testing size
        if new_P != 14:
            print(f'Pretraining uses 14x14, but current version is {new_P}x{new_P}')
            print(f'Interpolate the position embedding')
            sinusoid_table = sinusoid_table.reshape(-1, T, P, P, C)
            sinusoid_table = sinusoid_table.reshape(-1, P, P, C).permute(0, 3, 1, 2)
            sinusoid_table = torch.nn.functional.interpolate(
                sinusoid_table, size=(new_P, new_P), mode='bicubic', align_corners=False)
            # BT, C, H, W -> BT, H, W, C ->  B, T, H, W, C
            sinusoid_table = sinusoid_table.permute(0, 2, 3, 1).reshape(-1, T, new_P, new_P, C)
            sinusoid_table = sinusoid_table.flatten(1, 3)  # B, THW, C
    
    if cur_frame != ckpt_num_frame:
        print(f'Pretraining uses 4 frames, but current frame is {cur_frame}')
        print(f'Interpolate the position embedding')
        T = ckpt_num_frame # checkpoint frame
        new_T = cur_frame # testing frame
        # interpolate
        P = int((n_position // cur_frame) ** 0.5) # testing size
        C = d_hid
        sinusoid_table = sinusoid_table.reshape(-1, T, P, P, C)
        sinusoid_table = sinusoid_table.permute(0, 2, 3, 4, 1).reshape(-1, C, T)  # BHW, C, T
        sinusoid_table = torch.nn.functional.interpolate(sinusoid_table, size=new_T, mode='linear')
        sinusoid_table = sinusoid_table.reshape(1, P, P, C, new_T).permute(0, 4, 1, 2, 3) # B, T, H, W, C
        sinusoid_table = sinusoid_table.flatten(1, 3)  # B, THW, C
        
    return sinusoid_table

vid_path = "/mnt/petrelfs/yanziang/output.mp4"
# vid_path = "./demo/example/jesse_dance.mp4"
# from src.utils.easydict import EasyDict
from src.share_utils.my_easydict import MyEasyDict as EasyDict

# num_frame = 8
num_frame = 16
# resolution = 384
resolution = 224
vid, msg = load_video(vid_path, num_segments=num_frame, return_msg=True, resolution=resolution)
new_pos_emb = get_sinusoid_encoding_table(n_position=(resolution//16)**2*num_frame, cur_frame=num_frame)
model.vision_encoder.encoder.pos_embed = new_pos_emb

print(msg)
    
# The model expects inputs of shape: T x C x H x W
TC, H, W = vid.shape
video = vid.reshape(1, TC//3, 3, H, W).to("cuda:0")
            # B, T, C, H, W = image.shape
video = video.permute(0, 2, 1, 3, 4)
img_list = []
with torch.no_grad():
    image_emb = model.encode_vision(video, "Watch the video and answer the question.")
    image_emb = model.project_up(image_emb).to(torch.bfloat16)
    # image_emb, _ = model.encode_img(video, "Watch the video and answer the question.")
#     image_emb, _ = model.encode_img(video, "")

img_list.append(image_emb)
chat = EasyDict({
    "system": "",
    "roles": ("[INST]", "[/INST]"),
    "messages": [],
    "sep": ""
})

chat.messages.append([chat.roles[0], "<Video><VideoHere></Video> [/INST]"])
ask("Describe the video in details.", chat)

llm_message = answer(conv=chat, model=model, do_sample=False, img_list=img_list, max_new_tokens=512, temperature=1e-5, print_res=True)[0]
# print(llm_message)
data_list = {
    "Action Sequence": ("action_sequence.json", "p2:s3://star/Charades_v1_480/", "video", True), # has start & end
    "Action Prediction": ("action_prediction.json", "p2:s3://star/Charades_v1_480/", "video", True), # has start & end
    "Action Antonym": ("action_antonym.json", "pnorm2:s3://ssv2-video", "video", False),
    "Fine-grained Action": ("fine_grained_action.json", "p2hdd:s3://Moments_in_Time_Raw/videos/", "video", False),
    "Unexpected Action": ("unexpected_action.json", "pnorm2:s3://funqa-test/test/", "video", False),
    "Object Existence": ("object_existence.json", "p2:s3://clevrer/video_validation/", "video", False),
    "Object Interaction": ("object_interaction.json", "p2:s3://star/Charades_v1_480/", "video", True), # has start & end
    "Object Shuffle": ("object_shuffle.json", "p2:s3://perception/videos/", "video", False),
    "Moving Direction": ("moving_direction.json", "p2:s3://clevrer/video_validation/", "video", False),
    "Action Localization": ("action_localization.json", "p2:s3://sta/sta_video/", "video", True),  # has start & end
    "Scene Transition": ("scene_transition.json", "pnorm2:s3://scene-qa/video/", "video", False),
    "Action Count": ("action_count.json", "p2:s3://perception/videos/", "video", False),
    "Moving Count": ("moving_count.json", "p2:s3://clevrer/video_validation/", "video", False),
    "Moving Attribute": ("moving_attribute.json", "p2:s3://clevrer/video_validation/", "video", False),
    "State Change": ("state_change.json", "p2:s3://perception/videos/", "video", False),
    "Fine-grained Pose": ("fine_grained_pose.json", "p2:s3://nturgbd/", "video", False),
    "Character Order": ("character_order.json", "p2:s3://perception/videos/", "video", False),
    "Egocentric Navigation": ("egocentric_navigation.json", "p2:s3://vlnqa/", "video", False),
    "Episodic Reasoning": ("episodic_reasoning.json", "p2:s3://tvqa/frames_fps3_hq/", "frame", True),  # has start & end, read frame
    "Counterfactual Inference": ("counterfactual_inference.json", "p2:s3://clevrer/video_validation/", "video", False),
}

data_dir = "/mnt/petrelfs/share_data/likunchang/mvbench/json"

from petrel_client.client import Client
from decord import VideoReader, cpu
client = Client('~/petreloss.conf', enable_mc=False)
import os
import json
import io
class MVBench_dataset(Dataset):
    def __init__(self, data_dir, data_list, num_segments=8, resolution=224):
        self.data_list = []
        for k, v in data_list.items():
            with open(os.path.join(data_dir, v[0]), 'r') as f:
                json_data = json.load(f)
            for data in json_data:
                self.data_list.append({
                    'task_type': k,
                    'prefix': v[1],
                    'data_type': v[2],
                    'bound': v[3],
                    'data': data
                })
        
        self.decord_method = {
            'video': self.read_video,
            'gif': self.read_gif,
            'frame': self.read_frame,
        }
        
        self.num_segments = num_segments
        
        # transform
        crop_size = resolution
        scale_size = resolution
        input_mean = [0.48145466, 0.4578275, 0.40821073]
        input_std = [0.26862954, 0.26130258, 0.27577711]
        self.transform = T.Compose([
            GroupScale(int(scale_size), interpolation=InterpolationMode.BICUBIC),
            GroupCenterCrop(crop_size),
            Stack(),
            ToTorchFormatTensor(),
            GroupNormalize(input_mean, input_std) 
        ])
    
    def __str__(self):
        len_list = {}
        option_list = {}
        for data in self.data_list:
            if data['task_type'] not in len_list:
                len_list[data['task_type']] = 0
            len_list[data['task_type']] += 1
            if data['task_type'] not in option_list:
                option_list[data['task_type']] = 0
            option_list[data['task_type']] += len(data['data']['candidates'])
        
        correct = 0
        total = 0
        res = f"There are {len(self.data_list)} videos as follow:\n"
        for k, v in len_list.items():
            correct += len_list[k]
            total += option_list[k]
            res += f"{v} for {k} ({option_list[k]} options => {len_list[k]/option_list[k]*100:.2f}%)\n"
            correct = correct + 1 / option_list[k]
        res += f"Total random accuracy: {correct/total*100:.2f}%"
        return res.rstrip()
        
    def __len__(self):
        return len(self.data_list)
    
    def get_index(self, bound, fps, max_frame, first_idx=0):
        if bound:
            start, end = bound[0], bound[1]
        else:
            start, end = -100000, 100000
        start_idx = max(first_idx, round(start * fps))
        end_idx = min(round(end * fps), max_frame)
        seg_size = float(end_idx - start_idx) / self.num_segments
        frame_indices = np.array([
            int(start_idx + (seg_size / 2) + np.round(seg_size * idx))
            for idx in range(self.num_segments)
        ])
        return frame_indices
    
    def read_video(self, video_path, bound=None):
        if "s3://" in video_path:
            video_bytes = client.get(video_path)
            vr = VideoReader(io.BytesIO(video_bytes), ctx=cpu(0), num_threads=1)
        else:
            vr = VideoReader(video_path, ctx=cpu(0), num_threads=1)
        max_frame = len(vr) - 1
        fps = float(vr.get_avg_fps())
        
        images_group = list()
        frame_indices = self.get_index(bound, fps, max_frame, first_idx=0) 
        for frame_index in frame_indices:
            img = Image.fromarray(vr[frame_index].numpy())
            images_group.append(img)
        torch_imgs = self.transform(images_group)
        return torch_imgs
    
    def read_gif(self, video_path, bound=None, fps=25):
        if "s3://" in video_path:
            video_bytes = client.get(video_path)
            gif = imageio.get_reader(io.BytesIO(video_bytes))
        else:
            gif = imageio.get_reader(video_path)
        max_frame = len(gif) - 1
        
        images_group = list()
        frame_indices = self.get_index(bound, fps, max_frame, first_idx=0) 
        for index, frame in enumerate(gif):
            if index in frame_indices:
                img = cv2.cvtColor(frame, cv2.COLOR_RGBA2RGB)
                img = Image.fromarray(img)
                images_group.append(img)
        torch_imgs = self.transform(images_group)
        return torch_imgs
    
    def read_frame(self, video_path, bound=None, fps=3):
        if os.path.exists(video_path):
            max_frame = len(os.listdir(video_path))
        else:
            max_frame = len([k for k in client.list(video_path)])
        images_group = list()
        frame_indices = self.get_index(bound, fps, max_frame, first_idx=1) # frame_idx starts from 1
        for frame_index in frame_indices:
            if "s3://" in video_path:
                img_bytes = client.get(os.path.join(video_path, f"{frame_index:05d}.jpg"))
                img = Image.open(io.BytesIO(img_bytes))
            else:
                img = Image.open(os.path.join(video_path, f"{frame_index:05d}.jpg"))
            images_group.append(img)
        torch_imgs = self.transform(images_group)
        return torch_imgs

    def qa_template(self, data):
        question = f"Question: {data['question']}\n"
        question += "Options:\n"
        answer = data['answer']
        answer_idx = -1
        for idx, c in enumerate(data['candidates']):
            question += f"({chr(ord('A') + idx)}) {c}\n"
            if c == answer:
                answer_idx = idx
        question = question.rstrip()
        answer = f"({chr(ord('A') + answer_idx)}) {answer}"
        return question, answer

    def __getitem__(self, idx):
        decord_method = self.decord_method[self.data_list[idx]['data_type']]
        bound = None
        if self.data_list[idx]['bound']:
            bound = (
                self.data_list[idx]['data']['start'],
                self.data_list[idx]['data']['end'],
            )
        video_path = os.path.join(self.data_list[idx]['prefix'], self.data_list[idx]['data']['video'])
        torch_imgs = decord_method(video_path, bound)
        question, answer = self.qa_template(self.data_list[idx]['data'])
            
        return {
            'video': torch_imgs, 
            'question': question, 
            'answer': answer,
            'task_type': self.data_list[idx]['task_type']
        }

num_frame = 16
resolution = 224
new_pos_emb = get_sinusoid_encoding_table(n_position=(resolution//16)**2*num_frame, cur_frame=num_frame)
model.vision_encoder.encoder.pos_embed = new_pos_emb

dataset = MVBench_dataset(data_dir, data_list, num_segments=num_frame, resolution=resolution)

def infer_mvbench(
        data_sample, system="", 
        question_prompt='', # add in the end of question
        answer_prompt=None, # add in the begining of answer
        return_prompt='',  # add in the begining of return message
        system_q=False, # whether add question in the system prompt for QFormer
        print_res=True,
        system_llm=False
    ):
    video = data_sample["video"]
    TC, H, W = video.shape
    video = video.reshape(1, TC//3, 3, H, W).to("cuda:0")
    
    video_list = []
    with torch.no_grad():
        if system_q:
            video = video.permute(0, 2, 1, 3, 4)
            image_emb = model.encode_vision(video, system + data_sample['question'])
            video_emb = model.project_up(image_emb).to(torch.bfloat16)
            # video_emb, _ = model.encode_img(video, system + data_sample['question'])
        else:
            video = video.permute(0, 2, 1, 3, 4)
            image_emb = model.encode_vision(video, data_sample['question'])
            video_emb = model.project_up(image_emb).to(torch.bfloat16)
            # video_emb, _ = model.encode_img(video, system)
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
        prompt = system + data_sample['question'] + question_prompt
    else:
        prompt = data_sample['question'] + question_prompt
    
    ask(prompt, chat)

    llm_message = answer(
        conv=chat, model=model, do_sample=True, 
        img_list=video_list, max_new_tokens=100,
        answer_prompt=answer_prompt, print_res=print_res
    )[0]
    # remove potential explanation
    llm_message = return_prompt + llm_message.strip().split('\n')[0]
    # print(llm_message)
    # print(f"GT: {data_sample['answer']}")
    return llm_message

def check_ans(pred, gt):
    flag = False
    
    pred_list = pred.lower().split(' ')
    pred_option, pred_content = pred_list[0], ' '.join(pred_list[1:])
    gt_list = gt.lower().split(' ')
    gt_option, gt_content = gt_list[0], " ".join(gt_list[1:])
    if gt_content[-1] == '.':
        gt_content = gt_content[:-1]
    
    if pred_option.replace('.', '') in gt_option:
        flag = True
    elif gt_option in pred_option:
        flag = True
#     elif gt_content in pred_content:
#         flag = True
#     elif gt_content.replace("a ", "") in pred_content:
#         flag = True
#     elif gt_content.replace("an ", "") in pred_content:
#         flag = True
        
    return flag
save_path = '/mnt/petrelfs/yanziang/trackingllm/videoagent/time_scale/3'
correct = 0
total = 0
res_list = []
acc_dict = {}
from tqdm import tqdm
for example in tqdm(dataset):
    task_type = example['task_type']
    if task_type not in acc_dict:
        acc_dict[task_type] = [0, 0] # correct, total
    acc_dict[task_type][1] += 1
    total += 1
    pred = infer_mvbench(
        example, 
#         "Carefully observe the video and choose the best option for the question. ", 
#         "Carefully watch the video and pay attention to the cause, sequence of events, and object details and movements. Based on your observations, select the best option that accurately addresses the question. ",  # newPrompt
#         "Carefully watch the video and pay attention to the cause and sequence of events, the detail and movement of objects, and the action and pose of persons. Based on your observations, select the best option that accurately addresses the question. ", # newPrompt2
        "Carefully watch the video and pay attention to the cause and sequence of events, the detail and movement of objects, and the action and pose of persons. Based on your observations, select the best option that accurately addresses the question.\n", # newPrompt2
#         question_prompt="\nOnly give the best option without any explanation.",
#         question_prompt="\nThink it step by step. Only give the best option without any explanation.", # prompt2
        question_prompt="\nOnly give the best option.",  # prompt3
        answer_prompt="Best option:(",
        return_prompt='(',
        system_q=False,
#         system_q=True,
        print_res=True,
        system_llm=True
    )
    gt = example['answer']
    res_list.append({
        'pred': pred,
        'gt': gt
    })
    if check_ans(pred=pred, gt=gt):
        acc_dict[task_type][0] += 1
        correct += 1
    
    print(f"Part  Acc: {acc_dict[task_type][0] / acc_dict[task_type][1] * 100 :.2f}%")
    print(f"Total Acc: {correct / total * 100 :.2f}%")
    print('-' * 50, task_type, '-' * 50)

with open(f"{save_path}.json", "w") as f:
    json.dump({
        "acc_dict": acc_dict,
        "res_list": res_list
    }, f)

final_res = dict()
correct = 0
total = 0
for k, v in acc_dict.items():
    final_res[k] = v[0] / v[1] * 100
    correct += v[0]
    total += v[1]    
final_res['Avg'] = correct / total * 100

print(final_res)

save_path = save_path + '1'

out1 = "AS	AP	AA	FA	UA	OE	OI	OS	MD	AL	ST	AC	MC	MA	SC	FP	CO	EN	ER	CI	Avg"
out2 = ""
correct = 0
total = 0
with open(f"{save_path}.json", "r") as f:
    json_data = json.load(f)
    for k, v in json_data["acc_dict"].items():
        correct += v[0]
        total += v[1]    
        out2 += f"{v[0]/v[1]*100:.2f}\t"
out2 += f"{correct/total*100:.2f}"
print(out1)
print(out2)
# with open("upload_leaderboard.json", "w") as f:
#     json.dump(final_res, f)