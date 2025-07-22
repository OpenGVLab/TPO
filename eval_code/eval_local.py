from src.train.cap_utils import compute_metrics_cider_from_file


# print(compute_metrics_cider_from_file(
#     result_path="/mnt/petrelfs/lixinhao/lxh_exp/multimodalllm-lxh/logs/scripts/eval/umt-L_qformer_vicuna7b/eval_coco_f4_no_prompt.sh_20240512_201155/coco_5k_test_final_concat_global_step0_eval_results.json",
#     anno_path="/mnt/petrelfs/share_data/wangchenting/coco_5k_test_final_concat.json",
#     media_type='image',
#     data_prefix="pssd:s3://coco_caption/"))

print(
    'HowToLink Long MSRVTT cider: ',
    compute_metrics_cider_from_file(
    result_path="/mnt/petrelfs/wangchenting/multimodalllm/logs/scripts/eval/1b_mistral/msrvtt.sh_20240613_110845/msrvtt_test1k_global_step0_eval_results.json", 
    media_type='video', 
    anno_path="/mnt/petrelfs/wangchenting/multimodalllm/msrvtt_test1k.json")
)

# print(
#     'VCAP Long MSRVTT cider: ',
#     compute_metrics_cider_from_file(
#     result_path="/mnt/petrelfs/wangchenting/multimodalllm/logs/scripts/eval/umt-L_qformer_vicuna7b/vcap.sh_20240519_221437/msrvtt_test1k_global_step0_eval_results.json", 
#     media_type='video', 
#     anno_path="/mnt/petrelfs/wangchenting/multimodalllm/sharegptvideo_msrvtt.json")
# )

