import torch

# Hyper Parameters
device = 'cuda'

globalModelPath = "/root/autodl-tmp/SoulChat2.0-Qwen2-7B"
globalModelName = "qwq-32b"


def cuda_memory():
    return torch.cuda.memory_allocated(device) / (1024 ** 3)