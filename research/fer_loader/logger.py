import logging


def get_logger(name='fer_dataset') -> logging.Logger:
    """
    Create or return a configured logger instance.

    Ensures that handlers are added only once to avoid
    duplicated log messages when the function is called multiple times.
    """
    logger = logging.getLogger(name)
    if not logger.handlers:
        handler = logging.StreamHandler()
        formatter = logging.Formatter('%(asctime)s [%(levelname)s] %(message)s')
        handler.setFormatter(formatter)
        logger.addHandler(handler)
        logger.setLevel(logging.INFO)
    return logger
