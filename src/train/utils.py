import json
import os
import builtins
import datetime
import time
import subprocess
import logging
import torch
import torch.distributed as dist
from torch.optim.optimizer import Optimizer
from torch.optim.lr_scheduler import _LRScheduler
# from colossalai.booster import Booster
# from colossalai.cluster import DistCoordinator
from typing import Any, Dict, Tuple, Union
import random

logger = logging.getLogger(__name__)

def load_json(file_path: Union[str, os.PathLike]) -> Dict[str, Any]:
    """
    Load file in JSON format
    """
    with open(file=file_path, mode="r", encoding="utf-8") as fp:
        return json.load(fp)


def save_json(data: Dict[str, Any], file_path: Union[str, os.PathLike]) -> None:
    """
    Save as JSON format
    """
    with open(file=file_path, mode="w", encoding="utf-8") as fp:
        json.dump(data, fp=fp, ensure_ascii=False, indent=4)


# def save_checkpoint(
#     save_dir: Union[str, os.PathLike],
#     booster: Booster,
#     model: torch.nn.Module,
#     optimizer: Optimizer,
#     lr_scheduler: _LRScheduler,
#     step: int,
#     batch_size: int,
#     coordinator: DistCoordinator,
# ) -> None:
#     """
#     Save model checkpoint, optimizer, LR scheduler and intermedidate running states.
#     """

#     save_dir = os.path.join(save_dir, f"step-{step}")
#     os.makedirs(os.path.join(save_dir, "modeling"), exist_ok=True)

#     booster.save_model(model, os.path.join(save_dir, "modeling"), shard=True)

#     booster.save_optimizer(optimizer, os.path.join(save_dir, "optimizer"), shard=True)
#     booster.save_lr_scheduler(lr_scheduler, os.path.join(save_dir, "lr_scheduler"))
#     running_states = {
#         "step": step,
#         "sample_start_index": step * batch_size,
#     }
#     if coordinator.is_master():
#         save_json(running_states, os.path.join(save_dir, "running_states.json"))


# def load_checkpoint(
#     load_dir: Union[str, os.PathLike],
#     booster: Booster,
#     model: torch.nn.Module,
#     optimizer: Optimizer,
#     lr_scheduler: _LRScheduler,
# ) -> Tuple[int, int, int]:
#     """
#     Load model checkpoint, optimizer, LR scheduler and intermedidate running states.
#     """

#     # Update booster params states.
#     booster.load_model(model=model, checkpoint=os.path.join(load_dir, "modeling"))
#     booster.load_optimizer(optimizer=optimizer, checkpoint=os.path.join(load_dir, "optimizer"))
#     booster.load_lr_scheduler(lr_scheduler=lr_scheduler, checkpoint=os.path.join(load_dir, "lr_scheduler"))

#     running_states = load_json(file_path=os.path.join(load_dir, "running_states.json"))
#     return (
#         running_states["step"],
#         running_states["sample_start_index"],
#     )


def get_rank() -> int:
    if not dist.is_available():
        return 0
    if not dist.is_initialized():
        return 0
    return dist.get_rank()

def get_world_size() -> int:
    if not dist.is_available():
        return 1
    if not dist.is_initialized():
        return 1
    return dist.get_world_size()

def setup_for_distributed(is_master):
    builtin_print = builtins.print
    
    def print(*args, **kwargs):
        force = kwargs.pop("force", False)
        # force = force or (get_world_size() > 8)
        if is_master or force:
            now = datetime.datetime.now().time()
            builtin_print("[{}] ".format(now), end="")  # print with time stamp
            builtin_print(*args, **kwargs)

    builtins.print = print


def init_distributed_mode(use_dynamic_port: bool = True):
    if "SLURM_PROCID" in os.environ:
        rank = int(os.environ["SLURM_PROCID"])
        local_rank = rank % torch.cuda.device_count()

        world_size = int(os.environ["SLURM_NTASKS"])
        try:
            local_size = int(os.environ["SLURM_NTASKS_PER_NODE"])
        except:
            local_size = int(os.environ.get("LOCAL_SIZE", 1))

        if "MASTER_PORT" not in os.environ:
            port = 22111  # + random.randint(0, 20)
            if use_dynamic_port:
                for i in range(22108, 65535):
                    cmd = f"netstat -aon|grep {i}"
                    with os.popen(cmd, "r") as file:
                        if file.read() == "":
                            port = i
                            break

            print(f"MASTER_PORT = {port}")
            os.environ["MASTER_PORT"] = str(port)

            time.sleep(3)

        node_list = os.environ["SLURM_STEP_NODELIST"]
        addr = subprocess.getoutput(f"scontrol show hostname {node_list} | head -n1")
        if "MASTER_ADDR" not in os.environ:
            os.environ["MASTER_ADDR"] = addr

        os.environ["RANK"] = str(rank)
        os.environ["LOCAL_RANK"] = str(local_rank)
        os.environ["LOCAL_WORLD_SIZE"] = str(local_size)
        os.environ["WORLD_SIZE"] = str(world_size)

    else:
        rank = int(os.environ["RANK"])

    setup_for_distributed(rank == 0)

    print(
        f"Rank {os.environ['RANK']} | Local Rank {os.environ['LOCAL_RANK']} | "
        f"World Size {os.environ['WORLD_SIZE']} | Local World Size {os.environ['LOCAL_WORLD_SIZE']} |",
        force=True
    )


def setup_output_dir(output_dir, excludes=["code"]):
    """ensure not overwritting an exisiting/non-empty output dir"""
    if not os.path.exists(output_dir):
        os.makedirs(output_dir, exist_ok=False)
    else:
        existing_dirs_files = os.listdir(output_dir)  # list
        remaining = set(existing_dirs_files) - set(excludes)
        remaining = [e for e in remaining if "slurm" not in e]
        remaining = [e for e in remaining if ".out" not in e]
        # assert len(remaining) == 0, f"remaining dirs or files: {remaining}"
        logger.warn(f"remaining dirs or files: {remaining}")
