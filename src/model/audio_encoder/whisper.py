import torch
import torch.nn as nn
from .whisper_module import AudioEncoder


class BaseMLLM(nn.Module):
    def __init__(self):
        super().__init__()
        