import json
import logging
import os
from pycocoevalcap.tokenizer.ptbtokenizer import PTBTokenizer
from pycocoevalcap.meteor.meteor import Meteor
from pycocoevalcap.cider.cider import Cider
from pycocoevalcap.bleu.bleu import Bleu
from pycocoevalcap.rouge.rouge import Rouge
from pycocoevalcap.spice.spice import Spice
from src.share_utils.distributed import is_main_process

logger = logging.getLogger(__name__)


def reformat(d:dict):
    return {k:[{'caption':c}] for k,v in d.items() for c in v} # k是vid id，v是cap list

def cal_score(ref, sample):
    scorers = [
        # (Bleu(4), ["Bleu_1", "Bleu_2", "Bleu_3", "Bleu_4"]),
        # (Meteor(),"METEOR"),
        # (Rouge(), "ROUGE_L"),
        (Cider(), "CIDEr"),
        # (Spice(), "SPICE")
    ]
    final_scores = {}
    for scorer, method in scorers:
        logger.info('computing %s score' % (scorer.method()))
        assert ref.keys() == sample.keys(), f"ref.keys():{ref.keys()} sample.keys():{sample.keys()}"
        score, scores = scorer.compute_score(ref, sample)
        logger.info(score)
        if type(score) == list:
            for m, s in zip(method, score):
                final_scores[m] = s
        else:
            final_scores[method] = score
    return final_scores 



def compute_metrics_cider(eval_results, eval_dataset, output_dir, global_step):

    save_path = os.path.join(output_dir, f"{eval_dataset.label_file.split('/')[-1].split('.')[0]}_global_step{global_step}_eval_results.json")

    if is_main_process():
        # Open the file in write mode
        with open(save_path, "w") as file:
            # Dump the dictionary into the file
            json.dump(eval_results, file)
    
    logger.info(f'successfully save evaluation results to {save_path}')
    prediction_caps = {}
    # Iterate over the list of dictionaries
    for item in eval_results:
        # Access the image and caption values
        image = item['image_id']
        caption = item['caption']
        prediction_caps[image] = [caption]

    with open(eval_dataset.label_file, "r") as file:
        old_gt_data = json.load(file)
        gt_data = []
        for item in old_gt_data:
            if type(item['caption']) is list:
                for cap in item['caption']:
                    gt_data.append({eval_dataset.media_type:item[eval_dataset.media_type], "caption":cap})
            else:
                gt_data.append({eval_dataset.media_type:item[eval_dataset.media_type], "caption":item["caption"]})
                
        gt_caps = {}
        for item in gt_data:
            gt_caps[eval_dataset.data_root_prefix + os.path.join(eval_dataset.data_root, item[eval_dataset.media_type])] = [item['caption']]

    gt_caps = reformat(gt_caps)

    prediction_caps = reformat(prediction_caps)
    tokenizer = PTBTokenizer()
    gt_cap  = tokenizer.tokenize(gt_caps)
    pre_cap = tokenizer.tokenize(prediction_caps)

    score = cal_score(gt_cap, pre_cap)
    logger.info(f"anno_path of eval_dataset: {eval_dataset.label_file}, CIDER: {score}")

    return score


def compute_metrics_cider_from_file(result_path, anno_path, media_type, data_prefix='pssd:s3://MSR-VTT/MSRVTT_Videos/'):
    # Open the file in write mode
    with open(result_path, "r") as f:
        # Dump the dictionary into the file
        eval_results = json.load(f)
    
    prediction_caps = {}
    # Iterate over the list of dictionaries
    for item in eval_results:
        # Access the image and caption values
        image = item['image_id']
        caption = item['caption']
        prediction_caps[image] = [caption]

    with open(anno_path, "r") as file:
        old_gt_data = json.load(file)
    gt_data = []
    for item in old_gt_data:
        if type(item['caption']) is list:
            for cap in item['caption']:
                gt_data.append({media_type:item[media_type], "caption":cap})
        else:
            gt_data.append({media_type:item[media_type], "caption":item["caption"]})
    
    gt_caps = {}
    for item in gt_data:
        if data_prefix+item[media_type] in prediction_caps.keys():
            gt_caps[data_prefix+item[media_type]] = [item['caption']]
    
    
    # all_count = 0
    all_keys = []
    for k in prediction_caps.keys():
        if k not in gt_caps.keys():
            all_keys.append(k)
    for k in all_keys:
        prediction_caps.pop(k)
    
    all_keys = []
    for k in gt_caps.keys():
        if k not in prediction_caps.keys():
            all_keys.append(k)
    for k in all_keys:
        gt_caps.pop(k)
    
    gt_caps = reformat(gt_caps)

    prediction_caps = reformat(prediction_caps)
    tokenizer = PTBTokenizer()
    gt_cap  = tokenizer.tokenize(gt_caps)
    pre_cap = tokenizer.tokenize(prediction_caps)

    score = cal_score(gt_cap, pre_cap)
    print(score)
    return score

if __name__ == "__main__":
    compute_metrics_cider_from_file(result_path="/mnt/petrelfs/lixinhao/lxh_exp/multimodalllm-lxh/logs/scripts/eval/umt-L_qformer_vicuna7b/eval_anet_f8.sh_20240503_195031_no_prompt/anet_ret_val_global_step0_eval_results.json", media_type='video', anno_path="/mnt/petrelfs/share/videointern/annotations/anno_downstream/anet_ret_val.json",data_prefix="sssd:s3://video_pub/ANet_320p_fps30/val/")

