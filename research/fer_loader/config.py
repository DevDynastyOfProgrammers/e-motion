from dataclasses import dataclass


@dataclass
class Config:
    """
    Configuration object for FER-2013 data loading and preprocessing.

    Attributes:
        root_dir: Root directory containing data/ or FER-2013.zip
        image_size: Target image size (H, W)
        val_ratio: Fraction of training data used for validation split
        random_state: Seed for reproducibility
        grayscale: Load images in grayscale
        convert_to_3ch: Convert grayscale images to 3-channel format
        batch_size: Batch size for DataLoader
        num_workers: Number of DataLoader workers
        pin_memory: Enable pinned memory for faster GPU transfer
        force_rebuild_val: Force recreation of validation split
        api_url: Optional remote API endpoint (reserved)
    """
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
    
