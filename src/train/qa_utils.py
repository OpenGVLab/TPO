import os
import json

from .vqav2_metrics_src.vqa import VQA as VQAV2_VQA
from .vqav2_metrics_src.vqaEval import VQAEval as VQAV2_VQAEval
from .vizwiz_metrics_src.vqa import VQA as Vizwiz_VQA
from .vizwiz_metrics_src.vqaEval import VQAEval as Vizwiz_VQAEval
from src.share_utils.distributed import is_main_process
import logging
logger = logging.getLogger(__name__)

def extract_answer(response):
    response = response.replace('\"', '')
    # response = response.strip().split('.')[0].split(',')[0].split('!')[0].lower()
    response = response.strip().split('\n')[0].split('.')[0].split(',')[0].split('!')[0].lower()

    if 'is ' in response:
        response = response.split('is ')[1]
    if 'are ' in response:
        response = response.split('are ')[1]
    if 'a ' in response:
        response = response.split('a ')[1]
    if 'an ' in response:
        response = response.split('an ')[1]
    if 'the ' in response:
        response = response.split('the ')[1]
    if ' of' in response:
        response = response.split(' of')[0]

    if ' or ' in response:
        response = response.split(' or ')[0]
    if ' and ' in response:
        response = response.split(' and ')[0]

    return response.strip()

def compute_metrics_vqa(
    eval_results, 
    eval_dataset, 
    output_dir, 
    global_step,
    use_extract_answer=True,
):
    save_path = os.path.join(output_dir, f"{eval_dataset.label_file.split('/')[-1].split('.')[0]}_global_step{global_step}_eval_results.json")

    if is_main_process():
        # Open the file in write mode
        with open(save_path, "w") as file:
            # Dump the dictionary into the file
            json.dump(eval_results, file)
    
    data = eval_results
    total = len(data)
    acc = 0
    for item in data:
        try:
            gt = item['original_answer'][len('Answer: (')]
            answer = item['answer'][0]
            if gt == answer:
                acc += 1
            else:
                print(gt, answer)
        except:
            continue

    # print('total: ', total, 'right:', acc , 'Accuracy: ', acc / total)
    
    return {'overall_accuracy': acc / total}
    
    # vqa = VQAV2_VQA('/mnt/petrelfs/share_data/wangweiyun/datasets/VQAv2/vqav2_val.jsonl', '/mnt/petrelfs/share_data/wangweiyun/datasets/VQAv2/v2_OpenEnded_mscoco_val2014_questions.json')
    # vqaRes = vqa.loadRes(answers, '/mnt/petrelfs/share_data/wangweiyun/datasets/VQAv2/v2_OpenEnded_mscoco_val2014_questions.json')
    # vqaEval = VQAV2_VQAEval(vqa, vqaRes, n=2)  # n is precision of accuracy (number of places after decimal), default is 2
    # vqaEval.evaluate()

    # logger.info('accuracy of VQAv2: ', vqaEval.accuracy['overall'])
    
    # return {'overall_accuracy': vqaEval.accuracy['overall']}


def compute_metrics_vizwiz(
    annotation_file,
    answers,
    use_extract_answer=True,
):
    # answers = json.load(open(results_file))
    for item in answers:
        answer = item['answer']
        
        if use_extract_answer:
            answer = extract_answer(answer)
    
        item['answer'] = answer
    
    vqa = Vizwiz_VQA(annotation_file)
    vqaRes = Vizwiz_VQA(annotation=answers)
    vqaEval = Vizwiz_VQAEval(vqa, vqaRes, n=2)  # n is precision of accuracy (number of places after decimal), default is 2
    vqaEval.evaluate()

    res = {'overall_accuracy': vqaEval.accuracy['overall']}
    res.update(vqaEval.caption_metric.items())
    return res