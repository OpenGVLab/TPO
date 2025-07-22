from src.tokenizer.multimodal_llama_tokenizer import MultimodalLlamaTokenizer
from src.share_utils.my_easydict import MyEasyDict as edict
# from src.dataset.cap_dataset import VideoCapEvalDataset
from src.share_utils.py_config_utils import Config, eval_dict_leaf
from src.model import *
from torchvision import transforms
from torchvision.transforms import InterpolationMode
import io
import torch
from src.model.multimodalllm_pt import MultiModalLLM_PT
from petrel_client.client import Client
from src.share_utils.special_token import special_tokens
tokenizer = MultimodalLlamaTokenizer.from_pretrained(
    "/mnt/petrelfs/share_data/likunchang/model/llm/Mistral-7B-Instruct-v0.2", 
    local_files_only=True,
    n_query=96,
    v_query=96,
    special_tokens = special_tokens, 
) 
print(len(tokenizer))
from src.model.base_model import LLMConfig
config = LLMConfig.from_pretrained('/mnt/petrelfs/share_data/likunchang/model/llm/Mistral-7B-Instruct-v0.2', trust_remote_code=True)
model_config_path = "/mnt/petrelfs/yanziang/trackingllm/videoagent/configs/model/expert_umtl_f8.py"
model_cfg = eval_dict_leaf(Config.from_file(model_config_path))
model_cfg.model_tokenizer = tokenizer
config.model_config = model_cfg.model

config.model_tokenizer = tokenizer
model = MultiModalLLM_PT(config=config)
model.add_tokens()
model.add_lora()
model.load_pretrained_weights()
tokenizer.save_pretrained('/mnt/petrelfs/share_data/yanziang/expert_tokenizer')
model.save_pretrained('/mnt/petrelfs/share_data/yanziang/umt-l_mistral7b_stage3_tracking')
# import torch
# # pth = '/mnt/petrelfs/yanziang/trackingllm/videoagent/umt-l.pth'
# pth = '/mnt/petrelfs/share_data/likunchang/model/videochat2/videochat2_mistral_7b_stage3.pth'
# sd = torch.load(pth, map_location='cpu')

# new_sd = {}
# for k in sd.keys():
#     new_sd[k.replace('mistral_model', 'lm').replace('mistral_proj', 'project_up')] = sd[k]

# for tt in new_sd:
#     print(tt, new_sd[tt].shape)
#     if 'lora' in tt:
#         print(f'new_sd[tt].shape:{new_sd[tt].shape}')
#         if new_sd[tt].shape[0] == 32000:
#             new_sd[tt] = torch.cat((new_sd[tt], torch.zeros(7,16)))

# torch.save(new_sd, 'umt-l.pth')


# # breakpoint()

# sp = '/mnt/petrelfs/wangchenting/multimodalllm/logs/scripts/pt/1b_qformer_mistral/stage2.sh_20240613_132147/checkpoint-4000/model.safetensors'

# import torch
# from safetensors import safe_open
# from safetensors.torch import save_file


# tensors = {}
# with safe_open(sp, framework="pt", device="cpu") as f:
#    for key in f.keys():
#        tensors[key] = f.get_tensor(key)

# torch.save(tensors, 'model.bin')