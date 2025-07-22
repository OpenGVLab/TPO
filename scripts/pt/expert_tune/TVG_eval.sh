export OMP_NUM_THREADS=1
export DISABLE_ADDMM_CUDA_LT=1
export TORCH_CUDNN_USE_HEURISTIC_MODE_B=1
export NCCL_SOCKET_IFNAME=bond0
# export NCCL_DEBUG="INFO"
export NCCL_IB_HCA=mlx5_2,mlx5_3
export HF_ENDPOINT=https://hf-mirror.com


# export NCCL_SOCKET_IFNAME=bond0                 # 不同集群此参数需要修改
# export NCCL_DEBUG="INFO"
# export NCCL_IB_HCA=mlx5_2,mlx5_3 

NNODE=1
NUM_GPUS=1
NUM_CPU=8
JOB_NAME=$(basename $0)_$(date +"%Y%m%d_%H%M%S")
OUTPUT_DIR="$(dirname $0)/$JOB_NAME"

srun -p video5 \
    --preempt \
    --job-name=${JOB_NAME} \
    --ntasks=8 \
    --gres=gpu:8 \
    --ntasks-per-node=8 \
    --cpus-per-task=8 \
    --kill-on-bad-exit=1 \
    python eval_TVG.py