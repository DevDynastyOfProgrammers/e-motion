import os
import random
import shutil

import numpy as np
import torch


def set_seed(seed: int) -> None:
    """
    Set random seed for Python, NumPy and PyTorch (CPU and CUDA).
    """
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def ensure_dir(path: str, remove_if_exists: bool = False) -> None:
    """
    Create directory if it does not exist.
    Optionally removes existing directory before creation.
    """
    if os.path.exists(path):
        if remove_if_exists:
            shutil.rmtree(path)
    os.makedirs(path, exist_ok=True)
