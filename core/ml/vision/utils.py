import torch
import os
from pathlib import Path

def get_device(force_device: str = None):
    if force_device:
        if force_device.lower() == "cpu":
            return torch.device("cpu")
        elif force_device.lower() == "cuda":
            return torch.device("cuda" if torch.cuda.is_available() else "cpu")
        else:
            raise ValueError("force_device должен быть 'cpu' или 'cuda'")
    return torch.device("cuda" if torch.cuda.is_available() else "cpu")

def load_checkpoint(model, path: str, device: torch.device):
    """
    Универсальная загрузка чекпоинта.
    Поддерживает state_dict или словарь с 'model_state_dict'.
    Возвращает модель (в eval режиме).
    """
    if not os.path.exists(path):
        raise FileNotFoundError(f"Модель не найдена: {path}")
    ckpt = torch.load(path, map_location=device)
    if isinstance(ckpt, dict):
        if 'model_state_dict' in ckpt:
            state = ckpt['model_state_dict']
        elif 'state_dict' in ckpt:
            state = ckpt['state_dict']
        else:
            state = ckpt
        # убираем module. если есть
        new_state = {}
        for k, v in state.items():
            new_state[k.replace("module.", "")] = v
        try:
            model.load_state_dict(new_state)
        except Exception:
            model_state = model.state_dict()
            filtered = {k: v for k, v in new_state.items() if k in model_state and v.size() == model_state[k].size()}
            model_state.update(filtered)
            model.load_state_dict(model_state)
    else:
        model.load_state_dict(ckpt)
    model.to(device)
    model.eval()
    return model
