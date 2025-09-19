import os, random, numpy as np, torch

def init_cuda_safely(seed=42):
    os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "expandable_segments:True"
    random.seed(seed); np.random.seed(seed); torch.manual_seed(seed); torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    return device
