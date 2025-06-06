# logger.py

import logging

def get_logger(name: str) -> logging.Logger:
    """
    Creates or retrieves a logger instance with standard formatting.

    Args:
        name (str): Name of the logger, usually __name__ of the module.

    Returns:
        logging.Logger: Configured logger instance.
    """
    logger = logging.getLogger(name)
    if not logger.handlers:
        logger.setLevel(logging.INFO)
        handler = logging.StreamHandler()
        formatter = logging.Formatter('%(asctime)s - %(levelname)s - %(message)s')
        handler.setFormatter(formatter)
        logger.addHandler(handler)
    return logger
