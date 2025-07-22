import transformers
import hydra
import pyrootutils
import logging
import torch
import torch.distributed as dist
import wandb
import io
import os
from petrel_client.client import Client
pyrootutils.setup_root(__file__, indicator='.project-root', pythonpath=True)
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
logger = logging.getLogger(__name__)
from src.share_utils.special_token import special_tokens
from safetensors.torch import load_file 

@dataclass
class ConfigPathArguments:
    model: Optional[str] = field(default=None, metadata={"help": "config path of model used to initialize LM model"})
    tokenizer: Optional[str] = field(default=None, metadata={"help": "config path of tokenizer used to initialize tokenizer"})
    train_data: Optional[str] = field(default=None, metadata={"help": "config path of train dataset"})
    eval_data: Optional[str] = field(default=None, metadata={"help": "config path of eval dataset"})
    pretrained_path: Optional[str] = field(default=None, metadata={"help": "config path of pretrained model"})
    is_eval: Optional[bool] = field(default=None, metadata={"help": "train or eval"})
    mode: str = field(
        default="pt", metadata={"help": "The output directory where the model predictions and checkpoints will be written."})

@dataclass
class TrainingArguments(transformers.TrainingArguments):
    output_dir: str = field(
        default=None, metadata={"help": "The output directory where the model predictions and checkpoints will be written."})
    overwrite_output_dir: bool = field(default=False, metadata={"help": "Overwrite the content of the output directory"})
    optim: str = field(default="adamw_hf")
    gradient_accumulation_steps: int = field(
        default=1,
        metadata={"help": "Number of updates steps to accumulate before performing a backward/update pass."},
    )
    learning_rate: float = field(default=5e-5, metadata={"help": "The initial learning rate for Adam."})
    min_lr_ratio: float = field(default=0.1, metadata={"help": "The min lr ratio reqpect to the learning rate, only used to cosine lr scheduler"})
    weight_decay: float = field(default=0.0, metadata={"help": "Weight decay if we apply some."})
    num_train_epochs: float = field(default=3.0, metadata={"help": "Total number of training epochs to perform."})
    max_steps: int = field(default=-1, metadata={"help": "If > 0: set total number of training steps to perform. Override num_train_epochs."})

    lr_scheduler_type: str = field(default='cosine', metadata={"help": "The scheduler type to use."})
    report_to: Optional[str] = field(default=None, metadata={"help": "The list of integrations to report the results and logs to."})
    save_steps: int = field(default=1000, metadata={"help": "The interval between saving the model checkpoint."})
    bf16: bool = field(default=False, metadata={"help": "Whether to use 16-bit (mixed) precision (through NVIDIA apex) instead of 32-bit"})
    fp16: bool = field(default=False, metadata={"help": "Whether to use 16-bit (mixed) precision (through NVIDIA apex) instead of 32-bit"})
    dataloader_num_workers: int = field(default=8, metadata={"help": "The number of workers to use for data loading."})
    per_device_train_batch_size: int = field(default=8, metadata={"help": "Batch size per GPU/CPU for training."})
    per_device_eval_batch_size: int = field(default=8, metadata={"help": "Batch size per GPU/CPU for evaluation."})
    run_name: str = field(default=None, metadata={"help": "The name of the run."})
    torch_compile: bool = field(default=False, metadata={"help": "Whether to use torch.jit.trace to compile the model."})
    custom_lr_wd_dict: Optional[dict] = field(default=None, metadata={"help": "custom lr and wd for special params."})
    custom_per_device_train_bs_dict: Optional[dict] = field(default=None, metadata={"help": "custom bs for special media."})

    save_only_model: bool = field(default=False, metadata={"help": "not storing the optimizer, scheduler & rng state"})


def initialize(parser):
    # self.initialized = True
    # parser = argparse.ArgumentParser()
    parser.add_argument("--dset_name", type=str, default='charadesSTA', choices=["hl", 'charadesSTA', ])
    # parser.add_argument("--dset_domain", type=str, 
    #                     help="Domain to train for tvsum dataset. (Only used for tvsum and youtube-hl)")
    
    parser.add_argument("--eval_split_name", type=str, default="val",
                        help="should match keys in video_duration_idx_path, must set for VCMR")
    # parser.add_argument("--debug", action="store_true",
    #                     help="debug (fast) mode, break all loops, do not load all data into memory.")
    parser.add_argument("--data_ratio", type=float, default=1.0,
                        help="how many training and eval data to use. 1.0: use all, 0.1: use 10%."
                                "Use small portion for debug purposes. Note this is different from --debug, "
                                "which works by breaking the loops, typically they are not used together.")
    parser.add_argument("--results_root", type=str, default="results")
    parser.add_argument("--exp_id", type=str, default=None, help="id of this run, required at training")
    # parser.add_argument("--seed", type=int, default=2018, help="random seed")
    # parser.add_argument("--device", type=int, default=0, help="0 cuda, -1 cpu")
    # parser.add_argument("--num_workers", type=int, default=0,
    #                     help="num subprocesses used to load the data, 0: use main process")
    # parser.add_argument("--no_pin_memory", action="store_true",
    #                     help="Don't use pin_memory=True for dataloader. "
    #                          "ref: https://discuss.pytorch.org/t/should-we-set-non-blocking-to-true/38234/4")

    # training config
    # parser.add_argument("--lr", type=float, default=2e-4, help="learning rate")
    # parser.add_argument("--lr_drop", type=int, default=800, help="drop learning rate to 1/10 every lr_drop epochs")
    # parser.add_argument("--wd", type=float, default=1e-4, help="weight decay")
    # parser.add_argument("--n_epoch", type=int, default=200, help="number of epochs to run")
    parser.add_argument("--max_es_cnt", type=int, default=200, help="number of epochs to early stop, use -1 to disable early stop")
    # parser.add_argument("--bsz", type=int, default=32, help="mini-batch size")
    # parser.add_argument("--eval_bsz", type=int, default=100, help="mini-batch size at inference, for query")
    parser.add_argument("--eval_epoch", type=int, default=5,help="inference epoch")
    parser.add_argument("--grad_clip", type=float, default=0.1, help="perform gradient clip, -1: disable")
    parser.add_argument("--eval_untrained", action="store_true", help="Evaluate on un-trained model")
    # parser.add_argument("--resume", type=str, default=None, help="checkpoint path to resume or evaluate, without --resume_all this only load weights")
    parser.add_argument("--resume_all", action="store_true", help="if --resume_all, load optimizer/scheduler/epoch as well")
    parser.add_argument("--start_epoch", type=int, default=None, help="if None, will be set automatically when using --resume_all")

    # Data config
    parser.add_argument("--max_q_l", type=int, default=-1)
    parser.add_argument("--max_v_l", type=int, default=-1)
    parser.add_argument("--clip_length", type=float, default=1)   # 
    parser.add_argument("--max_windows", type=int, default=5)

    parser.add_argument("--train_path", type=str, default=None)
    parser.add_argument("--eval_path", type=str, default=None, help="Evaluating during training, for Dev set. If None, will only do training, ")
    parser.add_argument("--no_norm_vfeat", action="store_true", help="Do not do normalize video feat")
    parser.add_argument("--no_norm_tfeat", action="store_true", help="Do not do normalize text feat")
    parser.add_argument("--v_feat_dirs", type=str, nargs="+", help="video feature dirs. If more than one, will concat their features. " "Note that sub ctx features are also accepted here.")
    parser.add_argument("--t_feat_dir", type=str, help="text/query feature dir")
    # parser.add_argument("--a_feat_dir", type=str, help="audio feature dir")
    parser.add_argument("--v_feat_dim", type=int, default=770, help="video feature dim")
    parser.add_argument("--t_feat_dim", type=int, default=4096, help="text/query feature dim")
    # parser.add_argument("--a_feat_dim", type=int, help="audio feature dim")
    parser.add_argument("--ctx_mode", type=str, default="video_tef")

    # Model config
    parser.add_argument('--position_embedding', default='sine', type=str, choices=('sine', 'learned'),
                        help="Type of positional embedding to use on top of the image features")
    # * Transformer
    parser.add_argument('--enc_layers', default=3, type=int,
                        help="Number of encoding layers in the transformer")
    parser.add_argument('--dec_layers', default=3, type=int,
                        help="Number of decoding layers in the transformer")
    parser.add_argument('--t2v_layers', default=2, type=int,
                        help="Number of decoding layers in the transformer")
    parser.add_argument('--sent_layers', default=1, type=int,
                        help="Number of decoding layers in the transformer")
    parser.add_argument('--moment_layers', default=1, type=int,
                        help="Number of decoding layers in the transformer")
    parser.add_argument('--dummy_layers', default=2, type=int,
                        help="Number of encoding layers in the transformer")
    parser.add_argument('--dim_feedforward', default=1024, type=int,
                        help="Intermediate size of the feedforward layers in the transformer blocks")
    parser.add_argument('--hidden_dim', default=256, type=int,
                        help="Size of the embeddings (dimension of the transformer)")
    parser.add_argument('--input_dropout', default=0.5, type=float,
                        help="Dropout applied in input")
    parser.add_argument('--dropout', default=0.1, type=float,
                        help="Dropout applied in the transformer")
    parser.add_argument("--txt_drop_ratio", default=0, type=float,
                        help="drop txt_drop_ratio tokens from text input. 0.1=10%")
    parser.add_argument("--use_txt_pos", action="store_true", help="use position_embedding for text as well.")
    parser.add_argument('--nheads', default=8, type=int,
                        help="Number of attention heads inside the transformer's attentions")
    parser.add_argument('--num_queries', default=10, type=int,
                        help="Number of query slots")
    parser.add_argument('--num_dummies', default=45, type=int,
                        help="Number of dummy tokens")
    parser.add_argument('--total_prompts', default=10, type=int,
                        help="Number of query slots")
    parser.add_argument('--num_prompts', default=1, type=int,
                        help="Number of dummy tokens")
    parser.add_argument('--pre_norm', action='store_true')
    # other model configs
    parser.add_argument("--n_input_proj", type=int, default=2, help="#layers to encoder input")
    parser.add_argument("--contrastive_hdim", type=int, default=64, help="dim for contrastive embeddings")
    parser.add_argument("--temperature", type=float, default=0.07, help="temperature nce contrastive_align_loss")
    # Loss

    parser.add_argument("--saliency_margin", type=float, default=0.2)
    parser.add_argument('--no_aux_loss', dest='aux_loss', action='store_false',
                        help="Disables auxiliary decoding losses (loss at each layer)")
    parser.add_argument("--span_loss_type", default="l1", type=str, choices=['l1', 'ce'],
                        help="l1: (center-x, width) regression. ce: (st_idx, ed_idx) classification.")
    parser.add_argument("--contrastive_align_loss", action="store_true",
                        help="Disable contrastive_align_loss between matched query spans and the text.")
    # * Matcher
    parser.add_argument('--set_cost_span', default=10, type=float,
                        help="L1 span coefficient in the matching cost")
    parser.add_argument('--set_cost_giou', default=1, type=float,
                        help="giou span coefficient in the matching cost")
    parser.add_argument('--set_cost_class', default=4, type=float,
                        help="Class coefficient in the matching cost")

    # * Loss coefficients
    parser.add_argument("--lw_saliency", type=float, default=4,
                        help="weight for saliency loss, set to 0 will ignore")
    parser.add_argument("--lw_wattn", type=float, default=1.,
                        help="weight for saliency loss, set to 0 will ignore")
    parser.add_argument("--lw_ms_align", type=float, default=1.,
                        help="weight for saliency loss, set to 0 will ignore")
    parser.add_argument("--lw_distill", type=float, default=1.,
                        help="weight for saliency loss, set to 0 will ignore")
    parser.add_argument('--span_loss_coef', default=10, type=float)
    parser.add_argument('--giou_loss_coef', default=1, type=float)
    parser.add_argument('--label_loss_coef', default=4, type=float)
    parser.add_argument('--eos_coef', default=0.1, type=float,
                        help="Relative classification weight of the no-object class")
    parser.add_argument("--contrastive_align_loss_coef", default=0.02, type=float)

    parser.add_argument("--no_sort_results", action="store_true",
                        help="do not sort results, use this for moment query visualization")
    parser.add_argument("--max_before_nms", type=int, default=10)
    parser.add_argument("--max_after_nms", type=int, default=10)
    parser.add_argument("--conf_thd", type=float, default=0.0, help="only keep windows with conf >= conf_thd")
    parser.add_argument("--nms_thd", type=float, default=-1,
                        help="additionally use non-maximum suppression "
                                "(or non-minimum suppression for distance)"
                                "to post-processing the predictions. "
                                "-1: do not use nms. [0, 1]")
    return parser


def load_part_model(model, checkpoint, part_name):
    weights = {}
    for file_name in os.listdir(checkpoint):
        if file_name.endswith('.safetensors'):
            file_path = os.path.join(checkpoint, file_name)
            loaded_weights = load_file(file_path)
            weights.update(loaded_weights)

    part_params = {k: v for k, v in weights.items() if part_name in k}
    model_state_dict = model.state_dict()
    model_state_dict.update(part_params)
    model.load_state_dict(model_state_dict, strict=False)

    return model

def load_merge_llm(model, checkpoint, n = 0.5):
    weights = {}
    for file_name in os.listdir(checkpoint):
        if file_name.endswith('.safetensors'):
            file_path = os.path.join(checkpoint, file_name)
            loaded_weights = load_file(file_path)
            weights.update(loaded_weights)

    lm_params_new = {k: v for k, v in weights.items() if 'lm.' in k}
    
    model_state_dict = model.state_dict()
    lm_params_old = {k: v for k, v in model_state_dict.items() if 'lm.' in k}
    # lm_params = {k: v * n + lm_params_old[k] * (1-n) for k, v in lm_params_new}
    lm_params = {}
    for k in lm_params_new:
        lm_params[k] = lm_params_new[k] * n + lm_params_old[k] * (1-n)
    model_state_dict.update(lm_params)
    model.load_state_dict(model_state_dict, strict=False)

    return model

def train_or_eval():
    parser = transformers.HfArgumentParser((ConfigPathArguments, TrainingArguments))
    parser = initialize(parser)
    cfg_path, training_args, cg_opt = parser.parse_args_into_dataclasses()

    if is_main_process():
        setup_output_dir(training_args.output_dir, excludes=["code"])
        setup_logger(output=training_args.output_dir, color=True, name="mllm")
    
    dist.barrier()
    local_broadcast_process_authkey()

    # tokenizer = hydra.utils.instantiate(OmegaConf.load(cfg_path.tokenizer))
    # tokenizer.pad_token = tokenizer.unk_token
    tokenizer = MultimodalLlamaTokenizer.from_pretrained(
        "/mnt/petrelfs/share_data/likunchang/model/llm/Mistral-7B-Instruct-v0.2", 
        local_files_only=True,
        n_query=96,
        v_query=96,
        special_tokens = special_tokens, 
    ) 
    print(cfg_path.model)
    model_cfg = eval_dict_leaf(Config.from_file(cfg_path.model))
 
    from src.model.base_model import LLMConfig
    config = LLMConfig.from_pretrained('/mnt/petrelfs/share_data/likunchang/model/llm/Mistral-7B-Instruct-v0.2', trust_remote_code=True)
    config.model_config = model_cfg.model
    config.model_tokenizer = tokenizer
    config.cg_opt = cg_opt
    # print(config.cg_opt)

    # model = eval(model_cfg.model.get('model_cls'))(config=model_cfg.model, tokenizer=tokenizer)
    # model = eval(model_cfg.model.get('model_cls')).from_pretrained('/mnt/petrelfs/share_data/yanziang/umt-l_mistral7b_stage3', config=config, local_files_only=True)
    # path = '/mnt/petrelfs/yanziang/trackingllm/videoagent/logs/scripts/pt/expert_tune/bbox_s2.sh_20240813_224153/checkpoint-40000'
    # path = '/mnt/petrelfs/yanziang/trackingllm/videoagent/logs/scripts/pt/expert_tune/bbox_s2.sh_20240815_001103/checkpoint-30000'
    # path = '/mnt/petrelfs/share_data/yanziang/umt-l_mistral7b_stage3'
    # path = '/mnt/petrelfs/share_data/yanziang/videochat-e'
    # path = '/mnt/petrelfs/yanziang/trackingllm/videoagent/logs/scripts/pt/expert_tune/bbox_s2.sh_20240905_234551/checkpoint-10000'
    # path = '/mnt/petrelfs/yanziang/trackingllm/videoagent/logs/scripts/pt/expert_tune/bbox_s2.sh_20240904_232322/checkpoint-15000'
    # path = '/mnt/petrelfs/yanziang/trackingllm/videoagent/logs/scripts/pt/expert_tune/bbox_s2.sh_20240904_003656/checkpoint-10000'
    # path = '/mnt/petrelfs/yanziang/trackingllm/videoagent/logs/scripts/pt/expert_tune/bbox_s2.sh_20240831_232258/checkpoint-10000'
    # path = '/mnt/petrelfs/yanziang/trackingllm/videoagent/logs/scripts/pt/expert_tune/bbox_s2.sh_20240902_130711/checkpoint-3000'
    # path = '/mnt/petrelfs/yanziang/trackingllm/videoagent/logs/scripts/pt/expert_tune/bbox_s2.sh_20240902_130711/checkpoint-last'
    # path = '/mnt/petrelfs/yanziang/trackingllm/videoagent/logs/scripts/pt/expert_tune/test_grounding.sh_20240919_165743/checkpoint-last'
    # path = '/mnt/petrelfs/share_data/yanziang/videochat-e'
    # path = '/mnt/petrelfs/yanziang/trackingllm/videoagent/logs/scripts/pt/expert_tune/test_grounding.sh_20240920_215134/checkpoint-last'
    # path = '/mnt/petrelfs/yanziang/trackingllm/videoagent/logs/scripts/pt/expert_tune/bbox_s2.sh_20240911_234701/checkpoint-last'
    # path = '/mnt/petrelfs/yanziang/trackingllm/videoagent/logs/scripts/pt/expert_tune/bbox_s2.sh_20240909_192146/checkpoint-last'
    # path = '/mnt/petrelfs/yanziang/trackingllm/videoagent/logs/scripts/pt/expert_tune/bbox_s2.sh_20240908_174336/checkpoint-last'
    # path = '/mnt/petrelfs/yanziang/trackingllm/videoagent/logs/scripts/pt/expert_tune/bbox_s2.sh_20240831_113046/checkpoint-3000'
    # path = '/mnt/petrelfs/yanziang/trackingllm/videoagent/logs/scripts/pt/expert_tune/bbox_s2.sh_20240831_004240/checkpoint-6000'
    # path = '/mnt/petrelfs/yanziang/trackingllm/videoagent/logs/scripts/pt/expert_tune/bbox_s2.sh_20240830_095727/checkpoint-6000'
    # path = '/mnt/petrelfs/yanziang/trackingllm/videoagent/logs/scripts/pt/expert_tune/bbox_s2.sh_20240816_140859/checkpoint-45000'
    # /mnt/petrelfs/yanziang/trackingllm/videoagent/logs/scripts/pt/expert_tune/bbox_s2.sh_20240902_130711/checkpoint-last
    # path = '/mnt/petrelfs/yanziang/trackingllm/videoagent/logs/scripts/pt/expert_tune/TVG_s2.sh_20241002_215104/checkpoint-last'
    path = '/mnt/petrelfs/share_data/yanziang/all'
    # path = '/mnt/petrelfs/yanziang/trackingllm/videoagent/logs/scripts/pt/expert_tune/TVG_s2.sh_20241010_004436/checkpoint-last'
    model = MultiModalLLM_PT.from_pretrained(path, config=config, local_files_only=True, ignore_mismatched_sizes=True)
    # ck_path = '/mnt/petrelfs/share_data/lizhilin/CheckPoint/lr2e-4_bs32_DTNQHA/checkpoint-10000'
    ck_path = '/mnt/petrelfs/share_data/lizhilin/CheckPoint/checkpoint-600'
    # model = load_merge_llm(model, ck_path)
    model = load_part_model(model, ck_path, "cg_model")

    logger.info(f'Init tokenizer: {tokenizer}')
    # print(tokenizer.max)
    if cfg_path.mode == 'pt':
        train_data = hydra.utils.instantiate(OmegaConf.load('configs/dataset/dataset_pt_train.yaml'), tokenizer=tokenizer, config_path=cfg_path.train_data)
    elif cfg_path.mode == 'it_mistral_hd':
        train_data = hydra.utils.instantiate(OmegaConf.load('configs/dataset/dataset_it_mistral_hd_train.yaml'), tokenizer=tokenizer, config_path=cfg_path.train_data)
    elif cfg_path.mode == 'bbox':
        train_data = hydra.utils.instantiate(OmegaConf.load('configs/dataset/dataset_bbox.yaml'), tokenizer=tokenizer, config_path=cfg_path.train_data)
    else:
        train_data = hydra.utils.instantiate(OmegaConf.load('configs/dataset/dataset_it_train.yaml'), tokenizer=tokenizer, config_path=cfg_path.train_data)
    
    train_data_cfg = eval_dict_leaf(Config.from_file(cfg_path.train_data))
    training_args.custom_per_device_train_bs_dict = train_data_cfg.get('custom_per_device_train_bs_dict', None)
    logger.info(f'Init train data: {train_data} config_path: {cfg_path.train_data}')
    
    training_args.custom_lr_wd_dict = model_cfg.get('custom_lr_wd_dict', None)



    logger.info("training_args:")
    logger.info(training_args)

    if cfg_path.is_eval:
        trainer = CustomTrainer(
            self_defined_logging_dir=training_args.output_dir,
            model=model,
            args=training_args,
            train_dataset=train_data,
            eval_dataset=[],
            tokenizer=tokenizer,
            compute_metrics=compute_metrics_cider
        )
        trainer.evaluate()
    else:
        # if is_main_process():
        #     wandb.init(project="mllm", entity="likunchang", name=os.path.basename(training_args.output_dir), reinit=True)
        
        model.lm.config.use_cache = False
        trainer = CustomTrainer(
            self_defined_logging_dir=training_args.output_dir,
            model=model,
            args=training_args,
            train_dataset=train_data,
            eval_dataset=[],
            tokenizer=tokenizer,
            compute_metrics=None
        )
        trainer.train(resume_from_checkpoint=training_args.resume_from_checkpoint)
        trainer.save_model(output_dir=os.path.join(training_args.output_dir, "checkpoint-last"))
        # model.save_pretrained(os.path.join(training_args.output_dir, "checkpoint-last-huggingface"))
        # state_dict = model.state_dict()
        # torch.save(state_dict, os.path.join(training_args.output_dir, "checkpoint-last.pth"))
        # trainer.evaluate()


if __name__ == "__main__":
    # os.environ["WANDB_DISABLED"] = "true"
    init_distributed_mode()
    train_or_eval()
