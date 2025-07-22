import logging
import torch
import torch.utils.checkpoint
from torch import nn
from torch.nn import MSELoss
from transformers.modeling_outputs import (
    CausalLMOutputWithPast,
)
from typing import List, Optional, Tuple, Union
import torch.nn.functional as F
class MLP(nn.Module):
    """Very simple multi-layer perceptron (also called FFN)"""

    def __init__(self, input_dim, hidden_dim, output_dim, num_layers, dropout=0):
        super().__init__()
        self.num_layers = num_layers
        h = [hidden_dim] * (num_layers - 1)
        self.layers = nn.ModuleList(
            nn.Linear(n, k) for n, k in zip([input_dim] + h, h + [output_dim])
        )
        self.dropout = dropout
        if dropout:
            self.dropout = nn.Dropout(dropout)

    def forward(self, x):
        for i, layer in enumerate(self.layers):
            x = F.relu(layer(x)) if i < self.num_layers - 1 else layer(x)
            if self.dropout and i < self.num_layers:
                x = self.dropout(x)
        return x

class Temporal_Loss(nn.Module):
    def __init__(self, sigma, eos_coef):
        super().__init__()
        self.sigma = sigma
        self.eos_coef = eos_coef

    def get_temp_bound(self, start_sec, end_sec, fps, frame_idx):
        gt_temp_bound = []
        actionness = []
        for bs in range(len(start_sec)):
            tmp_start = round(float(start_sec[bs]) * fps[0][bs].item())
            tmp_end = round(float(end_sec[bs]) * fps[0][bs].item())
            # tmp_end = round(float(end_sec[0]) * fps[0][0].item())
            index_start = 0
            index_end = 0
            for i in range(len(frame_idx[bs])):
                if frame_idx[bs][i] < tmp_start:
                    index_start = i+1
                if frame_idx[bs][i] < tmp_end:
                    index_end = i

            act = torch.zeros([len(frame_idx[bs])])
            act[index_start:index_end+1] = 1
            actionness.append(act)
            
            inter = torch.where(torch.Tensor(act))[0].cpu().numpy().tolist()
            # print(inter)
            if len(inter) != 0:
                gt_temp_bound.append([inter[0],inter[-1]])
            else:
                gt_temp_bound.append([0,0])
        # import pdb; pdb.set_trace()
        actionness = torch.stack(actionness, dim=0).float()
        return actionness, gt_temp_bound

    def loss_actioness(self, pred_actioness, target_actionness, gt_temp_bound):
        # pred_actioness = outputs['pred_actioness'].squeeze(-1)  # [1, 64] 
        # target_actioness = torch.stack([target["actioness"] for target in targets], dim=0).float()  # [1, 64]  值为0或1
        weight = torch.full(pred_actioness.shape, self.eos_coef, device=pred_actioness.device)        # [1, 64]  值全为eos_coef=0.3
        
        for i_b in range(len(weight)):
            temp_bound = gt_temp_bound[i_b]
            weight[i_b][temp_bound[0] : temp_bound[1] + 1] = 1                # 将target_actioness中值为0的数替换为eos_coef=0.3
    
        loss_actioness = F.binary_cross_entropy_with_logits(pred_actioness, \
                target_actionness.to(pred_actioness.device), weight=weight, reduction='none')            # [1, 64]
        
        loss_actioness = loss_actioness.mean()

        return loss_actioness

    def loss_sted(self, pred_sted, gt_temp_bound, time_mask=None):
        # sted = outputs["pred_sted"]   # [1, 64, 2]
        sted = pred_sted
        
        target_start = torch.tensor([x[0] for x in gt_temp_bound], dtype=torch.long).to(sted.device)  # 采样的64帧中，gt开始的帧的idx
        target_end = torch.tensor([x[1] for x in gt_temp_bound], dtype=torch.long).to(sted.device)
        sted = sted.masked_fill(~time_mask[:, :, None], -1e32)  # put very low probability on the padded positions before softmax  [1,64,2]
        eps = 1e-6
        
        ######### start loss
        sigma = self.sigma
        start_distrib = (
            -(
                (
                    torch.arange(sted.shape[1])[None, :].to(sted.device)
                    - target_start[:, None]
                )
                ** 2
            )
            / (2 * sigma ** 2)
        ).exp()  # gaussian target  

        start_distrib = F.normalize(start_distrib + eps, p=1, dim=1)
        pred_start_prob = (sted[:, :, 0]).softmax(1)

        loss_start = (
            pred_start_prob * ((pred_start_prob + eps) / start_distrib).log()
        )
        loss_start = loss_start * time_mask

        ######## end loss
        end_distrib = (
            -(
                (
                    torch.arange(sted.shape[1])[None, :].to(sted.device)
                    - target_end[:, None]
                )
                ** 2
            )
            / (2 * sigma ** 2)
        ).exp()  # gaussian target
        end_distrib = F.normalize(end_distrib + eps, p=1, dim=1)
        pred_end_prob = (sted[:, :, 1]).softmax(1)
        loss_end = (
            pred_end_prob * ((pred_end_prob + eps) / end_distrib).log()
        )
        loss_end = loss_end * time_mask

        loss_sted = loss_start + loss_end
        loss_sted = loss_sted.mean()
        return loss_sted

    def forward(self, start_sec, end_sec, fps, frame_idx, pred_sted, pred_actionness):
        # import pdb; pdb.set_trace()
        time_mask = torch.ones(pred_actionness.shape[0], pred_actionness.shape[1]).bool().to(pred_sted.device)
        gt_actionness, gt_temp_bound = self.get_temp_bound(start_sec, end_sec, fps, frame_idx)
        # import pdb; pdb.set_trace()
        loss1 = self.loss_actioness(pred_actionness.squeeze(-1), gt_actionness, gt_temp_bound)
        loss2 = self.loss_sted(pred_sted, gt_temp_bound, time_mask)
        return loss1, loss2

class PostProcess(nn.Module):
    """ This module converts the model's output into the format expected by the coco api"""
    
    @torch.no_grad()
    def forward(self, out_sted, frames_id):
        """Perform the computation for inference evaluation
        """
        # import pdb; pdb.set_trace()

        b, t, _ = out_sted.shape
        device = out_sted.device
        temp_prob_map = torch.zeros(b,t,t).to(device)
        inf = -1e32
        for i_b in range(len(frames_id)):  # batch数
            duration = len(frames_id[0])
            sted_prob = (torch.ones(t, t) * inf).tril(0).to(device)
            sted_prob[duration:,:] = inf
            sted_prob[:,duration:] = inf
            temp_prob_map[i_b,:,:] = sted_prob
        
        temp_prob_map += F.log_softmax(out_sted[:, :, 0], dim=1).unsqueeze(2) + \
                F.log_softmax(out_sted[:, :, 1], dim=1).unsqueeze(1)
        
        # import pdb; pdb.set_trace()
        pred_steds = []
        for i_b in range(b):
            prob_map = temp_prob_map[i_b]  # [T * T]
            frame_id_seq = frames_id[i_b]
            prob_seq = prob_map.flatten(0)  
            max_tstamp = prob_seq.max(dim=0)[1].item()
            start_idx = max_tstamp // t
            end_idx = max_tstamp % t
            pred_sted = [frame_id_seq[start_idx], frame_id_seq[end_idx]+1]
            pred_steds.append(pred_sted)
    
        return pred_steds



#     def get_cg_item(self, meta):
#         # meta = self.anno[index]
#         # print(meta.keys())
#         # exit()
#         # meta = self.data[index]

#         if self.use_feature == False:
#             return dict(meta=meta, model_inputs=dict())

#         model_inputs = dict()

#         if self.use_glove:  # False
#             model_inputs["query_feat"] = self.get_query(meta["query"])
#         else:
#             model_inputs["query_feat"] = self._get_query_feat_by_qid(meta["qid"])  # (Dq, ) or (Lq, Dq)   # [16, 4096]
            
#         if self.use_video : # True
#             model_inputs["video_feat"] = self._get_video_feat_by_vid(meta["vid"])  # (Lv, Dv)
#             ctx_l = len(model_inputs["video_feat"])
#         else:
#             ctx_l = self.max_v_l

#         if self.use_tef:
#             tef_st = torch.arange(0, ctx_l, 1.0) / ctx_l
#             tef_ed = tef_st + 1.0 / ctx_l
#             tef = torch.stack([tef_st, tef_ed], dim=1)  # (Lv, 2)
#             if self.use_video :
#                 model_inputs["video_feat"] = torch.cat([model_inputs["video_feat"], tef], dim=1)  # (Lv, Dv+2)
#             else:
#                 model_inputs["video_feat"] = tef

#         if "relevant_windows" in meta: ## For Qvhighlights test set
#             model_inputs["span_labels"] = self.get_span_labels(meta["relevant_windows"], ctx_l)  # (#windows, 2)
#             if self.dset_name in ['charadesSTA', 'tacos', 'activitynet']: ## charades, tacos, nlq
#                 model_inputs["saliency_pos_labels"], model_inputs["saliency_neg_labels"], model_inputs["saliency_all_labels"] = \
#                     self.get_saliency_labels_sub_as_query(meta["relevant_windows"][0], self.duration, ctx_l)  # only one gt
#             elif "subs_train" not in self.data_path:
#                 model_inputs["saliency_pos_labels"], model_inputs["saliency_neg_labels"], model_inputs["saliency_all_labels"] = \
#                     self.get_saliency_labels_all(meta["relevant_clip_ids"], meta["saliency_scores"], ctx_l)
#             else:
#                 model_inputs["saliency_pos_labels"], model_inputs["saliency_neg_labels"], model_inputs["saliency_all_labels"] = \
#                     self.get_saliency_labels_sub_as_query(meta["relevant_windows"][0], meta["duration"], ctx_l)  # only one gt

#         if 'qvhighlight' in self.data_path:
#             model_inputs["relevant_clip_ids"] = meta["relevant_clip_ids"]
#         model_inputs["vid"] = meta["vid"]
#         model_inputs["qid"] = meta["qid"]
#         return dict(meta=meta, model_inputs=model_inputs)
    
#     def _get_query_feat_by_qid(self, qid):
#         # QVhighlight dataset
#         q_feat_path = join(self.q_feat_dir, f"qid{qid}.pt")
#         # q_feat = np.load(q_feat_path)[self.q_feat_type].astype(np.float32)
#         q_feat = torch.load(q_feat_path).numpy().astype(np.float32)
#         if self.q_feat_type == "last_hidden_state":
#             q_feat = q_feat[:self.max_q_l]
#         if self.normalize_t:
#             q_feat = l2_normalize_np_array(q_feat)
#         if self.txt_drop_ratio > 0:
#             q_feat = self.random_drop_rows(q_feat)
#         return torch.from_numpy(q_feat)  # (D, ) or (Lq, D)

#     def get_saliency_labels_sub_as_query(self, gt_window, duration, ctx_l, max_n=2):
#         clip_len = duration / ctx_l
#         gt_st = int(gt_window[0] / clip_len)
#         gt_ed = max(0, min(int(gt_window[1] / clip_len), ctx_l) - 1)
#         if gt_st > gt_ed:
#             gt_st = gt_ed

#         if gt_st != gt_ed:
#             pos_clip_indices = random.sample(range(gt_st, gt_ed + 1), k=max_n)  # 在GT frame idx中随机选两个
#         else:
#             if self.dset_name == 'nlq':
#                 pos_clip_indices = [gt_st] * 2
#             else:
#                 pos_clip_indices = [gt_st, gt_st]

#         neg_pool = list(range(0, gt_st)) + list(range(gt_ed+1, ctx_l)) # 非GT的frame idx
#         try:
#             neg_clip_indices = random.sample(neg_pool, k=max_n)  # 在非GT frame idx中随机选两个
#         except:
#             neg_clip_indices = pos_clip_indices

#         # For charades_sta
#         score_array = np.zeros(ctx_l)
#         score_array[gt_st:gt_ed + 1] = 1

#         return pos_clip_indices, neg_clip_indices, score_array
        
#     def get_saliency_labels(self, rel_clip_ids, scores, ctx_l, max_n=1, add_easy_negative=True):
#         """Sum the scores from the three annotations, then take the two clips with the
#         maximum scores as positive, and two with the minimum scores as negative.
#         Args:
#             rel_clip_ids: list(int), list of relevant clip ids
#             scores: list([anno1_score, anno2_score, anno3_score]),
#             ctx_l: int
#             max_n: int, #clips to use as positive and negative, for easy and hard negative, respectively.
#             add_easy_negative: bool, if True, sample eay negative outside the relevant_clip_ids.
#         """
#         # indices inside rel_clip_ids
#         scores = np.array(scores)  # (#rel_clips, 3)
#         agg_scores = np.sum(scores, 1)  # (#rel_clips, )
#         sort_indices = np.argsort(agg_scores)  # increasing

#         # indices in the whole video
#         # the min(_, ctx_l-1) here is incorrect, but should not cause
#         # much troubles since this should be rarely used.
#         hard_pos_clip_indices = [min(rel_clip_ids[idx], ctx_l-1) for idx in sort_indices[-max_n:]]
#         hard_neg_clip_indices = [min(rel_clip_ids[idx], ctx_l-1) for idx in sort_indices[:max_n]]
#         easy_pos_clip_indices = []
#         easy_neg_clip_indices = []
#         if add_easy_negative:
#             easy_neg_pool = list(set(range(ctx_l)) - set(rel_clip_ids))
#             if len(easy_neg_pool) >= max_n:
#                 easy_pos_clip_indices = random.sample(rel_clip_ids, k=max_n)
#                 easy_neg_clip_indices = random.sample(easy_neg_pool, k=max_n)
#             else:  # copy the hard ones
#                 easy_pos_clip_indices = hard_pos_clip_indices
#                 easy_neg_clip_indices = hard_neg_clip_indices

#         pos_clip_indices = hard_pos_clip_indices + easy_pos_clip_indices
#         neg_clip_indices = hard_neg_clip_indices + easy_neg_clip_indices
#         return pos_clip_indices, neg_clip_indices

#     def get_saliency_labels_all(self, rel_clip_ids, scores, ctx_l, max_n=1, add_easy_negative=True):
#         """Sum the scores from the three annotations, then take the two clips with the
#         maximum scores as positive, and two with the minimum scores as negative.
#         Args:
#             rel_clip_ids: list(int), list of relevant clip ids
#             scores: list([anno1_score, anno2_score, anno3_score]),
#             ctx_l: int
#             max_n: int, #clips to use as positive and negative, for easy and hard negative, respectively.
#             add_easy_negative: bool, if True, sample eay negative outside the relevant_clip_ids.
#         """
#         # indices inside rel_clip_ids
#         scores = np.array(scores)  # (#rel_clips, 3)
#         agg_scores = np.sum(scores, 1)  # (#rel_clips, )
#         sort_indices = np.argsort(agg_scores)  # increasing

#         # score_array = [min(agg_scores[idx], ctx_l-1) for idx in range(ctx_l)]
#         score_array = np.zeros(ctx_l)
#         max_len=ctx_l
#         for idx in range(len(rel_clip_ids)):
#             if rel_clip_ids[idx] >= ctx_l:
#                 max_len=max(max_len,rel_clip_ids[idx])
#                 # score_array_new = np.zeros(ctx_l + 1)
#                 score_array_new = np.zeros(max_len+1)
#                 # score_array_new[:ctx_l] = score_array
#                 score_array_new[:len(score_array)] = score_array
#                 score_array = score_array_new
#             score_array[rel_clip_ids[idx]] = agg_scores[idx]

#         # indices in the whole video
#         # the min(_, ctx_l-1) here is incorrect, but should not cause
#         # much troubles since this should be rarely used.
#         hard_pos_clip_indices = [min(rel_clip_ids[idx], ctx_l-1) for idx in sort_indices[-max_n:]]
#         hard_neg_clip_indices = [min(rel_clip_ids[idx], ctx_l-1) for idx in sort_indices[:max_n]]
#         easy_pos_clip_indices = []
#         easy_neg_clip_indices = []
#         if add_easy_negative:
#             easy_neg_pool = list(set(range(ctx_l)) - set(rel_clip_ids))
#             if len(easy_neg_pool) >= max_n:
#                 easy_pos_clip_indices = random.sample(rel_clip_ids, k=max_n)
#                 easy_neg_clip_indices = random.sample(easy_neg_pool, k=max_n)
#             else:  # copy the hard ones
#                 easy_pos_clip_indices = hard_pos_clip_indices
#                 easy_neg_clip_indices = hard_neg_clip_indices

#         pos_clip_indices = hard_pos_clip_indices + easy_pos_clip_indices
#         neg_clip_indices = hard_neg_clip_indices + easy_neg_clip_indices
#         return pos_clip_indices, neg_clip_indices, score_array   
    
#     def get_span_labels(self, windows, ctx_l):
#         """
#         windows: list([st, ed]) in seconds. E.g. [[26, 36]], corresponding st_ed clip_indices [[13, 17]] (inclusive)
#             Note a maximum of `self.max_windows` windows are used.
#         returns Tensor of shape (#windows, 2), each row is [center, width] normalized by video length
#         """
#         if len(windows) > self.max_windows:
#             random.shuffle(windows)
#             windows = windows[:self.max_windows]
#         if self.span_loss_type == "l1":
#             windows = torch.Tensor(windows) / (ctx_l * self.clip_len)  # normalized windows in xx
#             windows = span_xx_to_cxw(windows)  # normalized windows in cxw
#         elif self.span_loss_type == "ce":
#             windows = torch.Tensor([
#                 [int(w[0] / self.clip_len), min(int(w[1] / self.clip_len), ctx_l) - 1]
#                 for w in windows]).long()  # inclusive
#         else:
#             raise NotImplementedError
#         return windows

#     def _get_query_feat_by_qid(self, qid):
#         # QVhighlight dataset
#         q_feat_path = join(self.q_feat_dir, f"qid{qid}.pt")
#         # q_feat = np.load(q_feat_path)[self.q_feat_type].astype(np.float32)
#         q_feat = torch.load(q_feat_path).numpy().astype(np.float32)
#         if self.q_feat_type == "last_hidden_state":
#             # print(self.max_q_l)
#             # exit()
#             q_feat = q_feat[:self.max_q_l]
#         if self.normalize_t:
#             q_feat = l2_normalize_np_array(q_feat)
#         if self.txt_drop_ratio > 0:
#             q_feat = self.random_drop_rows(q_feat)
#         return torch.from_numpy(q_feat)  # (D, ) or (Lq, D)

#     def random_drop_rows(self, embeddings):
#         """randomly mask num_drop rows in embeddings to be zero.
#         Args:
#             embeddings: np.ndarray (L, D)
#         """
#         num_drop_rows = round(len(embeddings) * self.txt_drop_ratio)
#         if num_drop_rows > 0:
#             row_indices = np.random.choice(
#                 len(embeddings), size=num_drop_rows, replace=False)
#             embeddings[row_indices] = 0
#         return embeddings

#     def _get_video_feat_by_vid(self, vid):
#         v_feat_list = []
#         for _feat_dir in self.v_feat_dirs:
#             try:
#                 _feat_path = join(_feat_dir, f"{vid}.pt")
#                 _feat = torch.load(_feat_path)["features"][:self.max_v_l].numpy().astype(np.float32)
#             except:
#                 _feat_path = join(_feat_dir, f"{vid}.pt")
#                 _feat = torch.load(_feat_path)[:self.max_v_l].numpy().astype(np.float32)
#             if self.normalize_v:
#                 _feat = l2_normalize_np_array(_feat)
#             v_feat_list.append(_feat)
#         # some features are slightly longer than the others
#         min_len = min([len(e) for e in v_feat_list])
#         v_feat_list = [e[:min_len] for e in v_feat_list]
#         v_feat = np.concatenate(v_feat_list, axis=1)               # (vlen=34, 768)
#         return torch.from_numpy(v_feat)  # (Lv, D)

# def start_end_collate(batch):
#     batch_meta = [e["meta"] for e in batch]  # seems no need to collate ?

#     model_inputs_keys = batch[0]["model_inputs"].keys()
#     batched_data = dict()
#     for k in model_inputs_keys:
#         if k == "span_labels":
#             batched_data[k] = [dict(spans=e["model_inputs"]["span_labels"]) for e in batch]
#             continue
#         if k in ["saliency_pos_labels", "saliency_neg_labels"]:
#             batched_data[k] = torch.LongTensor([e["model_inputs"][k] for e in batch])
#             continue
#         if k == "saliency_all_labels":
#             pad_data, mask_data = pad_sequences_1d([e["model_inputs"][k] for e in batch], dtype=np.float32, fixed_length=None)
#             batched_data[k] = torch.tensor(pad_data, dtype=torch.float32)
#             continue
#         if k == 'qid':
#             batched_data[k] = [e["model_inputs"][k] for e in batch]
#             continue
#         if k == 'vid':
#             batched_data[k] = [e["model_inputs"][k] for e in batch]
#             continue
#         batched_data[k] = pad_sequences_1d(
#             [e["model_inputs"][k] for e in batch], dtype=torch.float32, fixed_length=None)
#     return batch_meta, batched_data

# def prepare_batch_inputs(batched_model_inputs, device, non_blocking=False):
#     model_inputs = dict(
#         src_txt=batched_model_inputs["query_feat"][0].to(device, non_blocking=non_blocking),
#         src_txt_mask=batched_model_inputs["query_feat"][1].to(device, non_blocking=non_blocking),
#         src_vid=batched_model_inputs["video_feat"][0].to(device, non_blocking=non_blocking),
#         src_vid_mask=batched_model_inputs["video_feat"][1].to(device, non_blocking=non_blocking),
#         vid=batched_model_inputs["vid"],
#         qid=batched_model_inputs["qid"],
#     )
#     targets = {}

#     # import pdb; pdb.set_trace()

#     if "span_labels" in batched_model_inputs:
#         targets["span_labels"] = [
#             dict(spans=e["spans"].to(device, non_blocking=non_blocking))
#             for e in batched_model_inputs["span_labels"]
#         ]
#     if "saliency_pos_labels" in batched_model_inputs:
#         for name in ["saliency_pos_labels", "saliency_neg_labels"]:
#             targets[name] = batched_model_inputs[name].to(device, non_blocking=non_blocking)

#     if "saliency_all_labels" in batched_model_inputs:
#         targets["saliency_all_labels"] = batched_model_inputs["saliency_all_labels"].to(device, non_blocking=non_blocking)
#         targets["relevant_clips"] = batched_model_inputs["saliency_all_labels"].to(device, non_blocking=non_blocking)
#     targets = None if len(targets) == 0 else targets
#     return model_inputs, targets

