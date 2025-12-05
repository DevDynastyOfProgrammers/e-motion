import os
from typing import Any

import torch
import torch.nn as nn
from loguru import logger


def get_device(force_device: str = 'cpu') -> torch.device:
    """
    Returns the Torch device. Defaults to CPU for game stability.
    """
    if force_device.lower() == 'cuda' and torch.cuda.is_available():
        return torch.device('cuda')
    return torch.device('cpu')


def load_checkpoint(model: nn.Module, path: str, device: torch.device) -> nn.Module:
    """
    Robustly loads model weights from a file.
    Handles 'state_dict', 'model_state_dict', and 'module.' prefixes.
    """
    if not os.path.exists(path):
        raise FileNotFoundError(f'Model file not found at: {path}')

    try:
        # Load everything to memory first
        checkpoint: dict[str, Any] | nn.Module = torch.load(path, map_location=device)

        state_dict = {}

        # Extract state_dict depending on how it was saved
        if isinstance(checkpoint, dict):
            if 'model_state_dict' in checkpoint:
                state_dict = checkpoint['model_state_dict']
            elif 'state_dict' in checkpoint:
                state_dict = checkpoint['state_dict']
            else:
                # Assume the dict itself is the state dict
                state_dict = checkpoint
        elif isinstance(checkpoint, nn.Module):
            # If the entire model object was saved
            state_dict = checkpoint.state_dict()
        else:
            raise ValueError(f'Unknown checkpoint format: {type(checkpoint)}')

        # Fix 'module.' prefix (happens when training with DataParallel)
        new_state_dict = {}
        for k, v in state_dict.items():
            name = k.replace('module.', '')
            new_state_dict[name] = v

        # Load weights
        # strict=False allows loading even if some minor keys don't match,
        # preventing crashes during development.
        keys = model.load_state_dict(new_state_dict, strict=False)

        if keys.missing_keys:
            logger.warning(f'Missing keys during load: {keys.missing_keys}')
        if keys.unexpected_keys:
            logger.debug(f'Unexpected keys in checkpoint: {keys.unexpected_keys}')

        model.to(device)
        model.eval()
        logger.info(f'Model loaded successfully from {path}')
        return model

    except Exception as e:
        logger.error(f'Failed to load checkpoint: {e}')
        raise e
