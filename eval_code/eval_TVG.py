from src.tokenizer.multimodal_llama_tokenizer import MultimodalLlamaTokenizer
from src.dataset.it_dataset_mistral_new import ITTrainDataset_mistral_new
from src.train.trainer import collate_hd
from src.share_utils.special_token import special_tokens
from src.share_utils.my_easydict import MyEasyDict as edict
from torchvision import transforms
from src.dataset.create import get_train_transform, create_dataset, create_loader, create_sampler
from src.model.multimodalllm_pt import MultiModalLLM_PT
from torch.utils.data import DataLoader
from omegaconf import OmegaConf
from src.share_utils.py_config_utils import Config, eval_dict_leaf
import torch
from src.share_utils.distributed import get_world_size, get_rank, gather_together
from src.dataset.dataloader import MetaLoader
from src.dataset.create import get_media_types
from src.dataset.create import create_dataset
from src.model.cgdetr_main.cg_detr.start_end_dataset import StartEndDataset, start_end_collate, prepare_batch_inputs

from src.model.cgdetr_main.cg_detr.span_utils import span_cxw_to_xx
from src.model.cgdetr_main.cg_detr.postprocessing_cg_detr import PostProcessorDETR
import torch.nn.functional as F
import argparse
import pprint

num_gpu = 8
import os
os.environ['CUDA_VISIBLE_DEVICES'] = str(int(os.environ['SLURM_PROCID'])%num_gpu)
gpu_id = int(os.environ['SLURM_PROCID'])%num_gpu
# gpu_id=0
device = f'cuda:{gpu_id}'

from configs.dataset.data_config.available_corpus import available_corpus
from tqdm import tqdm
import json
from src.model.cgdetr_main.standalone_eval.eval import eval_submission
from src.model.cgdetr_main.utils.basic_utils import save_jsonl, save_json
from src.model.cgdetr_main.utils.temporal_nms import temporal_nms
import logging

from src.train.train_or_eval_bbox_s2 import initialize
from collections import OrderedDict, defaultdict
from src.model.cgdetr_main.utils.basic_utils import AverageMeter

logger = logging.getLogger(__name__)
logging.basicConfig(format="%(asctime)s.%(msecs)03d:%(levelname)s:%(name)s - %(message)s",
                    datefmt="%Y-%m-%d %H:%M:%S",
                    level=logging.INFO)


parser = argparse.ArgumentParser()
parser = initialize(parser)
opt = parser.parse_args()


tokenizer = MultimodalLlamaTokenizer.from_pretrained(
    # "/mnt/petrelfs/share_data/yanziang/expert_tokenizer", 
    '/mnt/petrelfs/share_data/likunchang/model/llm/Mistral-7B-Instruct-v0.2',
    local_files_only=True,
    n_query=96,
    v_query=96,
    special_tokens = special_tokens, 
) 

val_list = [
        edict(
        anno_path="/mnt/petrelfs/share_data/lizhilin/AnnoData/Temporal-Grounding/charades_tvg_cg/cg_test1_1.json",
        data_root="pssd:s3://star/Charades_v1_480",
        media_type="TVG",
        v_feat_dirs = "/mnt/petrelfs/share_data/lizhilin/AnnoData/tvg_features/charades/charade_sta_1b",
        q_feat_dir = "/mnt/petrelfs/share_data/lizhilin/AnnoData/tvg_features/charades/text_feature_4096",
        ),
        edict(
        anno_path="/mnt/petrelfs/share_data/lizhilin/AnnoData/Temporal-Grounding/charades_tvg_cg/cg_test1_2.json",
        data_root="pssd:s3://star/Charades_v1_480",
        media_type="TVG",
        v_feat_dirs = "/mnt/petrelfs/share_data/lizhilin/AnnoData/tvg_features/charades/charade_sta_1b",
        q_feat_dir = "/mnt/petrelfs/share_data/lizhilin/AnnoData/tvg_features/charades/text_feature_4096",
        ),
        edict(
        anno_path="/mnt/petrelfs/share_data/lizhilin/AnnoData/Temporal-Grounding/charades_tvg_cg/cg_test1_3.json",
        data_root="pssd:s3://star/Charades_v1_480",
        media_type="TVG",
        v_feat_dirs = "/mnt/petrelfs/share_data/lizhilin/AnnoData/tvg_features/charades/charade_sta_1b",
        q_feat_dir = "/mnt/petrelfs/share_data/lizhilin/AnnoData/tvg_features/charades/text_feature_4096",
        ),
        edict(
        anno_path="/mnt/petrelfs/share_data/lizhilin/AnnoData/Temporal-Grounding/charades_tvg_cg/cg_test1_4.json",
        data_root="pssd:s3://star/Charades_v1_480",
        media_type="TVG",
        v_feat_dirs = "/mnt/petrelfs/share_data/lizhilin/AnnoData/tvg_features/charades/charade_sta_1b",
        q_feat_dir = "/mnt/petrelfs/share_data/lizhilin/AnnoData/tvg_features/charades/text_feature_4096",
        ),
        edict(
        anno_path="/mnt/petrelfs/share_data/lizhilin/AnnoData/Temporal-Grounding/charades_tvg_cg/cg_test1_5.json",
        data_root="pssd:s3://star/Charades_v1_480",
        media_type="TVG",
        v_feat_dirs = "/mnt/petrelfs/share_data/lizhilin/AnnoData/tvg_features/charades/charade_sta_1b",
        q_feat_dir = "/mnt/petrelfs/share_data/lizhilin/AnnoData/tvg_features/charades/text_feature_4096",
        ),
        edict(
        anno_path="/mnt/petrelfs/share_data/lizhilin/AnnoData/Temporal-Grounding/charades_tvg_cg/cg_test1_6.json",
        data_root="pssd:s3://star/Charades_v1_480",
        media_type="TVG",
        v_feat_dirs = "/mnt/petrelfs/share_data/lizhilin/AnnoData/tvg_features/charades/charade_sta_1b",
        q_feat_dir = "/mnt/petrelfs/share_data/lizhilin/AnnoData/tvg_features/charades/text_feature_4096",
        ),
        edict(
        anno_path="/mnt/petrelfs/share_data/lizhilin/AnnoData/Temporal-Grounding/charades_tvg_cg/cg_test1_7.json",
        data_root="pssd:s3://star/Charades_v1_480",
        media_type="TVG",
        v_feat_dirs = "/mnt/petrelfs/share_data/lizhilin/AnnoData/tvg_features/charades/charade_sta_1b",
        q_feat_dir = "/mnt/petrelfs/share_data/lizhilin/AnnoData/tvg_features/charades/text_feature_4096",
        ),
        edict(
        anno_path="/mnt/petrelfs/share_data/lizhilin/AnnoData/Temporal-Grounding/charades_tvg_cg/cg_test1_8.json",
        data_root="pssd:s3://star/Charades_v1_480",
        media_type="TVG",
        v_feat_dirs = "/mnt/petrelfs/share_data/lizhilin/AnnoData/tvg_features/charades/charade_sta_1b",
        q_feat_dir = "/mnt/petrelfs/share_data/lizhilin/AnnoData/tvg_features/charades/text_feature_4096",
        ),

        # edict(
        # anno_path="/mnt/petrelfs/share_data/lizhilin/AnnoData/Temporal-Grounding/qvhl_tvg_cg/cg_val_1.json",
        # data_root="pssd:s3://qvhighlight/videos",
        # media_type="TVG+HL",
        # v_feat_dirs = "/mnt/petrelfs/share_data/lizhilin/AnnoData/tvg_features/qvhl/qvhighlight_1b",
        # q_feat_dir = "/mnt/petrelfs/share_data/lizhilin/AnnoData/tvg_features/qvhl/qvhighlight_llama_text_feature",
        # ),
        # edict(
        # anno_path="/mnt/petrelfs/share_data/lizhilin/AnnoData/Temporal-Grounding/qvhl_tvg_cg/cg_val_2.json",
        # data_root="pssd:s3://qvhighlight/videos",
        # media_type="TVG+HL",
        # v_feat_dirs = "/mnt/petrelfs/share_data/lizhilin/AnnoData/tvg_features/qvhl/qvhighlight_1b",
        # q_feat_dir = "/mnt/petrelfs/share_data/lizhilin/AnnoData/tvg_features/qvhl/qvhighlight_llama_text_feature",
        # ),
        # edict(
        # anno_path="/mnt/petrelfs/share_data/lizhilin/AnnoData/Temporal-Grounding/qvhl_tvg_cg/cg_val_3.json",
        # data_root="pssd:s3://qvhighlight/videos",
        # media_type="TVG+HL",
        # v_feat_dirs = "/mnt/petrelfs/share_data/lizhilin/AnnoData/tvg_features/qvhl/qvhighlight_1b",
        # q_feat_dir = "/mnt/petrelfs/share_data/lizhilin/AnnoData/tvg_features/qvhl/qvhighlight_llama_text_feature",
        # ),
        # edict(
        # anno_path="/mnt/petrelfs/share_data/lizhilin/AnnoData/Temporal-Grounding/qvhl_tvg_cg/cg_val_4.json",
        # data_root="pssd:s3://qvhighlight/videos",
        # media_type="TVG+HL",
        # v_feat_dirs = "/mnt/petrelfs/share_data/lizhilin/AnnoData/tvg_features/qvhl/qvhighlight_1b",
        # q_feat_dir = "/mnt/petrelfs/share_data/lizhilin/AnnoData/tvg_features/qvhl/qvhighlight_llama_text_feature",
        # ),
        # edict(
        # anno_path="/mnt/petrelfs/share_data/lizhilin/AnnoData/Temporal-Grounding/qvhl_tvg_cg/cg_val_5.json",
        # data_root="pssd:s3://qvhighlight/videos",
        # media_type="TVG+HL",
        # v_feat_dirs = "/mnt/petrelfs/share_data/lizhilin/AnnoData/tvg_features/qvhl/qvhighlight_1b",
        # q_feat_dir = "/mnt/petrelfs/share_data/lizhilin/AnnoData/tvg_features/qvhl/qvhighlight_llama_text_feature",
        # ),
        # edict(
        # anno_path="/mnt/petrelfs/share_data/lizhilin/AnnoData/Temporal-Grounding/qvhl_tvg_cg/cg_val_6.json",
        # data_root="pssd:s3://qvhighlight/videos",
        # media_type="TVG+HL",
        # v_feat_dirs = "/mnt/petrelfs/share_data/lizhilin/AnnoData/tvg_features/qvhl/qvhighlight_1b",
        # q_feat_dir = "/mnt/petrelfs/share_data/lizhilin/AnnoData/tvg_features/qvhl/qvhighlight_llama_text_feature",
        # ),
        # edict(
        # anno_path="/mnt/petrelfs/share_data/lizhilin/AnnoData/Temporal-Grounding/qvhl_tvg_cg/cg_val_7.json",
        # data_root="pssd:s3://qvhighlight/videos",
        # media_type="TVG+HL",
        # v_feat_dirs = "/mnt/petrelfs/share_data/lizhilin/AnnoData/tvg_features/qvhl/qvhighlight_1b",
        # q_feat_dir = "/mnt/petrelfs/share_data/lizhilin/AnnoData/tvg_features/qvhl/qvhighlight_llama_text_feature",
        # ),
        # edict(
        # anno_path="/mnt/petrelfs/share_data/lizhilin/AnnoData/Temporal-Grounding/qvhl_tvg_cg/cg_val_8.json",
        # data_root="pssd:s3://qvhighlight/videos",
        # media_type="TVG+HL",
        # v_feat_dirs = "/mnt/petrelfs/share_data/lizhilin/AnnoData/tvg_features/qvhl/qvhighlight_1b",
        # q_feat_dir = "/mnt/petrelfs/share_data/lizhilin/AnnoData/tvg_features/qvhl/qvhighlight_llama_text_feature",
        # ),
        ]

from src.dataset.it_dataset_mistral_eval_TVG import ITTrainDataset_mistral_eval

Myd = ITTrainDataset_mistral_eval(
    ann_file = val_list[gpu_id],   #gpu_id
    transform=get_train_transform(config='', train_file='',mode='resize'),
    tokenizer=tokenizer,
    num_frames = 16,
    cgdetr_config=dict(
        # dset_name = "charadesSTA",
        q_feat_type = "last_hidden_state",
        ctx_mode = "video_tef",
        clip_len = 1,
        max_windows = 5,
        max_q_l = 100,
        max_v_l = 100000000,
        span_loss_type = "l1",
        use_feature = True,
        ),
)
# import pdb; pdb.set_trace()

dataloader = DataLoader(
    Myd,
    batch_size=1,
    shuffle=False,
    num_workers=1,
    pin_memory=True,
    collate_fn=collate_hd,
)


from src.model.base_model import LLMConfig
config = LLMConfig.from_pretrained('/mnt/petrelfs/share_data/likunchang/model/llm/Mistral-7B-Instruct-v0.2', trust_remote_code=True)
model_cfg = eval_dict_leaf(Config.from_file('/mnt/petrelfs/yanziang/trackingllm/videoagent/configs/model/expert_umtl_f8_s2.py'))
config.model_config = model_cfg.model
config.model_tokenizer = tokenizer
config.cg_opt = opt

hl = False
if val_list[gpu_id].media_type == 'TVG+HL':
    opt.clip_length=2
    hl = True

# model_path = 'logs/scripts/pt/expert_tune/lr3e-4_bs32_DTNQHA/checkpoint-20000'
# model_path = '/mnt/petrelfs/yanziang/trackingllm/videoagent/logs/scripts/pt/expert_tune/TVG_s2.sh_20241001_143316/checkpoint-last'
# model_path = '/mnt/petrelfs/yanziang/trackingllm/videoagent/logs/scripts/pt/expert_tune/TVG_s2.sh_20241004_111411/checkpoint-last'
# model_path = '/mnt/petrelfs/yanziang/trackingllm/videoagent/logs/scripts/pt/expert_tune/TVG_s2.sh_20241016_102651/checkpoint-60000'
# model_path = '/mnt/petrelfs/yanziang/trackingllm/videoagent/logs/scripts/pt/expert_tune/TVG_s2.sh_20241018_225600/checkpoint-last'
# model_path = '/mnt/petrelfs/yanziang/trackingllm/videoagent/logs/scripts/pt/expert_tune/TVG_s2.sh_20241020_170132/checkpoint-80000'
# model_path = '/mnt/petrelfs/yanziang/trackingllm/videoagent/logs/scripts/pt/expert_tune/chat_cotraining_1e-5/checkpoint-last'
# model_path = '/mnt/petrelfs/share_data/yanziang/reasoning_chat2'
model_path = '/mnt/petrelfs/share_data/yanziang/tvg_ablate'
model = MultiModalLLM_PT.from_pretrained(model_path, config=config, local_files_only=True)
# import pdb; pdb.set_trace()

# model = model.to(device)
model.eval()
model = model.to(dtype=torch.bfloat16, device=device)

def post_processing_mr_nms(mr_res, nms_thd, max_before_nms, max_after_nms):
    mr_res_after_nms = []
    for e in mr_res:
        e["pred_relevant_windows"] = temporal_nms(
            e["pred_relevant_windows"][:max_before_nms],
            nms_thd=nms_thd,
            max_after_nms=max_after_nms
        )
        mr_res_after_nms.append(e)
    return mr_res_after_nms

def eval_epoch_post_processing(submission, gt_data, submission_path):
    # IOU_THDS = (0.5, 0.7)
    logger.info("Saving/Evaluating before nms results")
    # submission_path = os.path.join(save_submission_filename)
    save_jsonl(submission, submission_path)

    if opt.eval_split_name in ["val"]:  # since test_public has no GT
        metrics = eval_submission(
            submission, gt_data,
            verbose=False,
            hl=hl, 
        )
        save_metrics_path = submission_path.replace(".jsonl", "_metrics.json")
        save_json(metrics, save_metrics_path, save_pretty=True, sort_keys=False)
        latest_file_paths = [submission_path, save_metrics_path]
    else:
        metrics = None
        latest_file_paths = [submission_path, ]

    if opt.nms_thd != -1:
        logger.info("[MR] Performing nms with nms_thd {}".format(opt.nms_thd))
        submission_after_nms = post_processing_mr_nms(
            submission, nms_thd=opt.nms_thd,
            max_before_nms=opt.max_before_nms, max_after_nms=opt.max_after_nms
        )

        logger.info("Saving/Evaluating nms results")
        submission_nms_path = submission_path.replace(".jsonl", "_nms_thd_{}.jsonl".format(opt.nms_thd))
        save_jsonl(submission_after_nms, submission_nms_path)
        if opt.eval_split_name == "val":
            metrics_nms = eval_submission(
                submission_after_nms, gt_data,
                verbose=opt.debug,
            )
            save_metrics_nms_path = submission_nms_path.replace(".jsonl", "_metrics.json")
            save_json(metrics_nms, save_metrics_nms_path, save_pretty=True, sort_keys=False)
            latest_file_paths += [submission_nms_path, save_metrics_nms_path]
        else:
            metrics_nms = None
            latest_file_paths = [submission_nms_path, ]
    else:
        metrics_nms = None
    return metrics, metrics_nms, latest_file_paths

def compute_mr_results(model, eval_loader, criterion=None):
    model.eval()
    if criterion:
        assert eval_loader.dataset.load_labels
        criterion.eval()

    loss_meters = defaultdict(AverageMeter)
    # write_tb = tb_writer is not None and epoch_i is not None

    mr_res = []
    for batch in tqdm(eval_loader):
        inputs = prepare_input(batch[0])
        query_meta = prepare_input(batch[1])
        tvg_data = prepare_input(batch[2])
        tvg_inputs, tvg_targets = prepare_batch_inputs(tvg_data)
        # import pdb; pdb.set_trace()

        tvg_outputs = model(**tvg_inputs)
        prob = F.softmax(tvg_outputs["pred_logits"], -1)  # (batch_size, #queries, #classes=2)
        if opt.span_loss_type == "l1":
            scores = prob[..., 0]  # * (batch_size, #queries)  foreground label is 0, we directly take it
            pred_spans = tvg_outputs["pred_spans"]  # (bsz, #queries, 2)
            _saliency_scores = tvg_outputs["saliency_scores"].half()  # (bsz, L)
            saliency_scores = []
            valid_vid_lengths = tvg_inputs["src_vid_mask"].sum(1).cpu().tolist()
            for j in range(len(valid_vid_lengths)):
                saliency_scores.append(_saliency_scores[j, :int(valid_vid_lengths[j])].tolist())
        else:
            bsz, n_queries = tvg_outputs["pred_spans"].shape[:2]  # # (bsz, #queries, max_v_l *2)
            pred_spans_logits = tvg_outputs["pred_spans"].view(bsz, n_queries, 2, opt.max_v_l)
            pred_span_scores, pred_spans = F.softmax(pred_spans_logits, dim=-1).max(-1)  # 2 * (bsz, #queries, 2)
            scores = torch.prod(pred_span_scores, 2)  # (bsz, #queries)
            pred_spans[:, 1] += 1
            pred_spans *= opt.clip_length

        # compose predictions
        for idx, (meta, spans, score) in enumerate(zip(query_meta, pred_spans.cpu(), scores.cpu())):
            if opt.span_loss_type == "l1":
                spans = span_cxw_to_xx(spans) * meta["duration"]
                spans = torch.clamp(spans, 0, meta["duration"])
            # # (#queries, 3), [st(float), ed(float), score(float)]
            cur_ranked_preds = torch.cat([spans, score[:, None]], dim=1).tolist()
            if not opt.no_sort_results:
                cur_ranked_preds = sorted(cur_ranked_preds, key=lambda x: x[2], reverse=True)
            cur_ranked_preds = [[float(f"{e:.4f}") for e in row] for row in cur_ranked_preds]
            cur_query_pred = dict(
                qid=meta["qid"],
                query=meta["query"],
                vid=meta["vid"],
                pred_relevant_windows=cur_ranked_preds,
                pred_saliency_scores=saliency_scores[idx]
            )
            mr_res.append(cur_query_pred)

        if criterion:
            loss_dict = criterion(tvg_outputs, tvg_targets)
            weight_dict = criterion.weight_dict
            losses = sum(loss_dict[k] * weight_dict[k] for k in loss_dict.keys() if k in weight_dict)
            loss_dict["loss_overall"] = float(losses)  # for logging only
            for k, v in loss_dict.items():
                loss_meters[k].update(float(v) * weight_dict[k] if k in weight_dict else float(v))

    # if write_tb and criterion:
    #     for k, v in loss_meters.items():
    #         tb_writer.add_scalar("Eval/{}".format(k), v.avg, epoch_i + 1)

    if val_list[gpu_id].media_type in ['TVG+HL']:
        post_processor = PostProcessorDETR(
            clip_length=opt.clip_length, min_ts_val=0, max_ts_val=150,
            min_w_l=2, max_w_l=150, move_window_method="left",
            process_func_names=("clip_ts", "round_multiple")
        )
    elif val_list[gpu_id].media_type in ['TVG']:
        if opt.v_feat_dim == 4096: # vgg
            post_processor = PostProcessorDETR(
                clip_length=opt.clip_length, min_ts_val=0, max_ts_val=360,
                min_w_l=12, max_w_l=360, move_window_method="left",
                process_func_names=("clip_ts", "round_multiple")
            )
        else:
            post_processor = PostProcessorDETR(
                clip_length=opt.clip_length, min_ts_val=0, max_ts_val=150,
                min_w_l=2, max_w_l=60, move_window_method="left",
                process_func_names=("clip_ts", "round_multiple")
            )
    else:
        post_processor = PostProcessorDETR(
            clip_length=opt.clip_length, min_ts_val=0, max_ts_val=50000,
            min_w_l=0, max_w_l=50000, move_window_method="left",
            process_func_names=(["round_multiple"])
        )

    mr_res = post_processor(mr_res)
    return mr_res, loss_meters

def compute_mr_results_generate(model, eval_loader, criterion=None):
    model.eval()
    if criterion:
        assert eval_loader.dataset.load_labels
        criterion.eval()

    loss_meters = defaultdict(AverageMeter)
    # write_tb = tb_writer is not None and epoch_i is not None

    mr_res = []
    count = 0
    for batch in tqdm(eval_loader):
        inputs = prepare_input(batch[0])
        query_meta = prepare_input(batch[1])
        tvg_data = prepare_input(batch[2])
        tvg_inputs, tvg_targets = prepare_batch_inputs(tvg_data)
        # import pdb; pdb.set_trace()

        ret, prompt_token = generate(inputs['input_ids'], inputs['attention_mask'], inputs['labels'],inputs['video'].type(dtype=torch.bfloat16), inputs['video_idx'])
        if prompt_token == None:
            count += 1
        
            continue
        
        import pdb; pdb.set_trace()
        tvg_outputs = model(**tvg_inputs, prompt_token=prompt_token)
        prob = F.softmax(tvg_outputs["pred_logits"], -1)  # (batch_size, #queries, #classes=2)
        if opt.span_loss_type == "l1":
            scores = prob[..., 0]  # * (batch_size, #queries)  foreground label is 0, we directly take it
            pred_spans = tvg_outputs["pred_spans"]  # (bsz, #queries, 2)
            _saliency_scores = tvg_outputs["saliency_scores"].half()  # (bsz, L)
            saliency_scores = []
            valid_vid_lengths = tvg_inputs["src_vid_mask"].sum(1).cpu().tolist()
            for j in range(len(valid_vid_lengths)):
                saliency_scores.append(_saliency_scores[j, :int(valid_vid_lengths[j])].tolist())
        else:
            bsz, n_queries = tvg_outputs["pred_spans"].shape[:2]  # # (bsz, #queries, max_v_l *2)
            pred_spans_logits = tvg_outputs["pred_spans"].view(bsz, n_queries, 2, opt.max_v_l)
            pred_span_scores, pred_spans = F.softmax(pred_spans_logits, dim=-1).max(-1)  # 2 * (bsz, #queries, 2)
            scores = torch.prod(pred_span_scores, 2)  # (bsz, #queries)
            pred_spans[:, 1] += 1
            pred_spans *= opt.clip_length

        # compose predictions
        for idx, (meta, spans, score) in enumerate(zip(query_meta, pred_spans.cpu(), scores.cpu())):
            if opt.span_loss_type == "l1":
                spans = span_cxw_to_xx(spans) * meta["duration"]
                spans = torch.clamp(spans, 0, meta["duration"])
            # # (#queries, 3), [st(float), ed(float), score(float)]
            cur_ranked_preds = torch.cat([spans, score[:, None]], dim=1).tolist()
            if not opt.no_sort_results:
                cur_ranked_preds = sorted(cur_ranked_preds, key=lambda x: x[2], reverse=True)
            cur_ranked_preds = [[float(f"{e:.4f}") for e in row] for row in cur_ranked_preds]
            cur_query_pred = dict(
                qid=meta["qid"],
                query=meta["query"],
                vid=meta["vid"],
                pred_relevant_windows=cur_ranked_preds,
                pred_saliency_scores=saliency_scores[idx],
                ans=ret
            )
            mr_res.append(cur_query_pred)

        if criterion:
            loss_dict = criterion(tvg_outputs, tvg_targets)
            weight_dict = criterion.weight_dict
            losses = sum(loss_dict[k] * weight_dict[k] for k in loss_dict.keys() if k in weight_dict)
            loss_dict["loss_overall"] = float(losses)  # for logging only
            for k, v in loss_dict.items():
                loss_meters[k].update(float(v) * weight_dict[k] if k in weight_dict else float(v))

    if val_list[gpu_id].media_type in ['TVG+HL']:
        post_processor = PostProcessorDETR(
            clip_length=opt.clip_length, min_ts_val=0, max_ts_val=150,
            min_w_l=2, max_w_l=150, move_window_method="left",
            process_func_names=("clip_ts", "round_multiple")
        )
    elif val_list[gpu_id].media_type in ['TVG']:
        if opt.v_feat_dim == 4096: # vgg
            post_processor = PostProcessorDETR(
                clip_length=opt.clip_length, min_ts_val=0, max_ts_val=360,
                min_w_l=12, max_w_l=360, move_window_method="left",
                process_func_names=("clip_ts", "round_multiple")
            )
        else:
            post_processor = PostProcessorDETR(
                clip_length=opt.clip_length, min_ts_val=0, max_ts_val=150,
                min_w_l=2, max_w_l=60, move_window_method="left",
                process_func_names=("clip_ts", "round_multiple")
            )
    else:
        post_processor = PostProcessorDETR(
            clip_length=opt.clip_length, min_ts_val=0, max_ts_val=50000,
            min_w_l=0, max_w_l=50000, move_window_method="left",
            process_func_names=(["round_multiple"])
        )

    mr_res = post_processor(mr_res)
    print("No TimeStart :", count)
    return mr_res, loss_meters

def prepare_input(data):
    """
    Prepares one `data` before feeding it to the model, be it a tensor or a nested list/dictionary of tensors.
    """
    if isinstance(data, dict):
        return type(data)({k: prepare_input(v) for k, v in data.items()})
    elif isinstance(data, (tuple, list)):
        return type(data)(prepare_input(v) for v in data)
    elif isinstance(data, torch.Tensor):
        return data.to(device)
    return data

def generate(input_ids, attention_mask, labels, video, video_idx):
    temperature = 1e-5
    max_new_tokens = 20
    output_ids = list(input_ids[0])
    # import pdb; pdb.set_trace()
    prompt_token = None
    text_embeds = model.pad_text_embeds(input_ids=input_ids, video=video, video_idx=video_idx)
    for i in range(max_new_tokens):
        if i == 0:
            outputs = model.lm(
                    inputs_embeds=text_embeds,
                    attention_mask=attention_mask,
                    output_hidden_states=True,
                    return_dict=True,
                ) # 整个query输入得到的输出
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
        if ret == '<time_begin>':
            attention_mask = torch.ones(1, past_key_values[0][0].shape[-2] + 1, device=device)
            temp_embeds = model.temporal_token.bfloat16()
            # text_embeds_bbox = torch.cat((text_embeds.squeeze(0), bbox_embeds), dim=0).unsqueeze(0)
            # import pdb; pdb.set_trace()
            out = model.lm(inputs_embeds=temp_embeds,use_cache=True,attention_mask=attention_mask,output_hidden_states=True,past_key_values=past_key_values)
            prompt_token = out.hidden_states[-1]
            # import pdb; pdb.set_trace()

            # last_hidden_states = out.hidden_states[-1]
            # selected_hidden_states = last_hidden_states[0][0]
            # bbox.append(model.temporal_embed(selected_hidden_states))

            # last_token_logits = logits[0][-1]
            # if temperature < 1e-4:
            #     token = int(torch.argmax(last_token_logits))
            # else:
            #     probs = torch.softmax(last_token_logits / temperature, dim=-1)
            #     token = int(torch.multinomial(probs, num_samples=1))
            # print(token)
        if (ret == '</s>'):
            break
    ret = tokenizer.decode(output_ids)
    del past_key_values

    return ret, prompt_token


# file = f'/mnt/petrelfs/lizhilin/videoagent/outputs/EXP_0816_230559/{gpu_id}.jsonl'
submission_path = f"/mnt/petrelfs/yanziang/trackingllm/videoagent/logs/results/merge_new/grounding/{gpu_id}-c.jsonl"

# submission, eval_loss_meters = compute_mr_results(model.cg_model, dataloader)
submission, eval_loss_meters = compute_mr_results_generate(model.cg_model, dataloader)
# import pdb; pdb.set_trace()

metrics_no_nms, metrics_nms, latest_file_paths = eval_epoch_post_processing(submission, Myd.anno, submission_path)

if opt.eval_split_name == 'val':
    logger.info("metrics_no_nms {}".format(pprint.pformat(metrics_no_nms["brief"], indent=4)))
if metrics_nms is not None:
    logger.info("metrics_nms {}".format(pprint.pformat(metrics_nms["brief"], indent=4)))


