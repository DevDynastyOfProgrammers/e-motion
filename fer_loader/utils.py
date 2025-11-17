import os
import random
import shutil
import numpy as np
import torch

def set_seed(seed: int):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)

def ensure_dir(path: str, remove_if_exists: bool = False):
    if os.path.exists(path):
        if remove_if_exists:
            shutil.rmtree(path)
    os.makedirs(path, exist_ok=True)
