import argparse
import os
import shutil
import sys
import zipfile
from pathlib import Path

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '../../')))
from research.fer_loader.config import Config
from research.fer_loader.dataloader import build_dataloaders
from research.fer_loader.logger import get_logger

logger = get_logger('fer_main')


def extract_archive(zip_path: Path, target_dir: Path) -> None:
    """Safe extraction logic."""
    logger.info(f'Extracting {zip_path} to {target_dir}...')

    with zipfile.ZipFile(zip_path, 'r') as zip_ref:
        members = zip_ref.namelist()
        root_folder = members[0].split('/')[0]
        is_nested = all(m.startswith(root_folder + '/') for m in members if '/' in m)

        zip_ref.extractall(target_dir)

    if is_nested:
        nested_dir = target_dir / root_folder
        logger.info(f"Detected nested folder '{root_folder}'. Moving files up...")
        for item in os.listdir(nested_dir):
            shutil.move(str(nested_dir / item), str(target_dir))
        os.rmdir(nested_dir)

    logger.info('Extraction complete.')


def main() -> None:
    parser = argparse.ArgumentParser(description='FER-2013 Setup using Team Loader')
    parser.add_argument('--source', required=True, help='Path to FER-2013.zip')
    parser.add_argument('--val-ratio', type=float, default=0.1, help='Validation split ratio')
    parser.add_argument('--bs', type=int, default=64, help='Batch size for check')
    args = parser.parse_args()

    # Resolve target dataset directory
    # Expected by research/vision/train.py
    base_dir = Path(__file__).resolve().parent.parent.parent
    target_data_dir = base_dir / 'research/vision/data'

    logger.info(f'Target Dataset Directory: {target_data_dir}')

    # Extract dataset
    source_path = Path(args.source)
    if not target_data_dir.exists() or not (target_data_dir / 'train').exists():
        target_data_dir.mkdir(parents=True, exist_ok=True)
        if source_path.suffix == '.zip' and source_path.exists():
            extract_archive(source_path, target_data_dir)
        else:
            logger.error(f"Archive '{source_path}' not found.")
            return
    else:
        logger.info('Data exists. Proceeding to split verification.')

    # Initialize loader configuration
    logger.info('Initializing Team Loader logic...')

    # Initialize loader configuration
    cfg = Config(
        root_dir=str(target_data_dir),
        val_ratio=args.val_ratio,
        batch_size=args.bs,
        grayscale=True,
    )

    try:
        # Validation split will be created automatically if missing
        train_loader, val_loader, test_loader = build_dataloaders(cfg)

        logger.info('Team Loader verification passed!')
        logger.info(f'   Train Batches: {len(train_loader)}')
        logger.info(f'   Val Batches:   {len(val_loader)}')
        logger.info(f'   Test Batches:  {len(test_loader)}')

        logger.info("\nData is ready for 'research/vision/train.py'")

    except Exception as e:
        logger.error(f'Loader check failed: {e}')
        import traceback

        traceback.print_exc()


if __name__ == '__main__':
    main()
