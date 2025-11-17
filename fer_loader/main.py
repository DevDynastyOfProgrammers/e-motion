import argparse
import zipfile
import os
from .config import Config
from .dataloader import build_dataloaders
from .logger import get_logger

def main():
    parser = argparse.ArgumentParser(description="FER-2013 loader & augmentation pipeline")
    parser.add_argument("--root", required=True, help="Path to dataset root (where FER-2013.zip is or data/ folder)")
    parser.add_argument("--val-ratio", type=float, default=0.1, help="Ratio of training data used for validation split")
    parser.add_argument("--force", action="store_true", help="Force rebuild of validation split even if it exists")
    parser.add_argument("--seed", type=int, default=42, help="Random seed")
    parser.add_argument("--bs", type=int, default=64, help="Batch size")
    args = parser.parse_args()

    logger = get_logger()

    zip_path = os.path.join(args.root, "FER-2013.zip")
    data_dir = os.path.join(args.root, "data")

    if os.path.exists(zip_path):
        logger.info(f"Found archive: {zip_path}")
        if not os.path.exists(data_dir):
            logger.info("Extracting FER-2013.zip ...")
            with zipfile.ZipFile(zip_path, 'r') as zip_ref:
                zip_ref.extractall(args.root)
            logger.info("Extraction complete.")
        else:
            logger.info("Data folder already exists — skipping extraction.")
    else:
        logger.info("No archive found — assuming data/ folder already prepared.")

    root_dir = os.path.join(args.root, "data") if os.path.exists(data_dir) else args.root

    cfg = Config(
        root_dir=root_dir,
        val_ratio=args.val_ratio,
        random_state=args.seed,
        batch_size=args.bs,
        force_rebuild_val=args.force
    )

    train_loader, val_loader, test_loader = build_dataloaders(cfg)
    logger.info("Dataset prepared successfully.")
    logger.info(f"Train batches: {len(train_loader)}, Val: {len(val_loader)}, Test: {len(test_loader)}")

if __name__ == "__main__":
    import multiprocessing
    multiprocessing.freeze_support()
    main()
