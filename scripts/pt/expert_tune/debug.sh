export OMP_NUM_THREADS=1
export DISABLE_ADDMM_CUDA_LT=1
export TORCH_CUDNN_USE_HEURISTIC_MODE_B=1
export NCCL_SOCKET_IFNAME=bond0
export NCCL_DEBUG="INFO"
export NCCL_IB_HCA=mlx5_2,mlx5_3
export TORCH_CPP_LOG_LEVEL=INFO
NNODE=1
NUM_GPUS=1
NUM_CPU=8
JOB_NAME=$(basename $0)_$(date +"%Y%m%d_%H%M%S")
OUTPUT_DIR="$(dirname $0)/$JOB_NAME"

srun -p video5 \
    --preempt \
    --job-name=${JOB_NAME} \
    --ntasks=1 \
    --gres=gpu:1 \
    --ntasks-per-node=1 \
    --cpus-per-task=8 \
    --kill-on-bad-exit=1 \
    python \
    -u src/train/train_or_eval_bbox_s2.py \
    --mode 'bbox' \
    --model configs/model/expert_umtl_f8_s2.py \
    --tokenizer /mnt/petrelfs/yanziang/trackingllm/videoagent/configs/tokenizer/special_q192.yaml \
    --train_data configs/dataset/data_config/debug.py \
    --output_dir logs/${OUTPUT_DIR} \
    --deepspeed configs/deepspeed/stage2_bf16.json \
    --evaluation_strategy "no" \
    --save_strategy "steps" \
    --save_steps 40000 \
    --num_train_epochs 3 \
    --min_lr_ratio 0.05 \
    --report_to "tensorboard" \
    --learning_rate 2e-5 \
    --weight_decay 0.02 \
    --lr_scheduler_type "cosine" \
    --per_device_train_batch_size 1 \
    --per_device_eval_batch_size 1 \
    --gradient_accumulation_steps 4 \
    --dataloader_num_workers 2 \
    --logging_steps 1 \
    --log_level 'info' \
    --bf16 \
    --adam_beta1 0.9 \
    --adam_beta2 0.999 \
    --max_grad_norm 1.0 \
    --adam_epsilon 1e-5 \
    --ignore_data_skip \
    --save_only_model True
