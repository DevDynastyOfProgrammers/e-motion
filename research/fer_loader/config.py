from dataclasses import dataclass


@dataclass
class Config:
    root_dir: str
    image_size: tuple[int, int] = (48, 48)
    val_ratio: float = 0.1
    random_state: int = 42
    grayscale: bool = True
    convert_to_3ch: bool = False
    batch_size: int = 64
    num_workers: int = 4
    pin_memory: bool = True
    force_rebuild_val: bool = False
    api_url: str = ''
    
