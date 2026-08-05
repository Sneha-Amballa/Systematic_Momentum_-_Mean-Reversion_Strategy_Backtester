import logging
import os
from typing import List

def setup_logging(log_level: int = logging.INFO) -> None:
    """
    Configures the root logger for the application.
    Logs will output to both stdout (console) and a log file in the root directory.

    Args:
        log_level (int): Logging level (e.g., logging.INFO, logging.DEBUG).
    """
    logger = logging.getLogger()
    logger.setLevel(log_level)
    
    # Avoid duplicate handlers if setup_logging is called multiple times
    if logger.hasHandlers():
        logger.handlers.clear()

    # Formatter for structured and readable logs
    formatter = logging.Formatter(
        fmt="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S"
    )

    # Console handler
    console_handler = logging.StreamHandler()
    console_handler.setFormatter(formatter)
    logger.addHandler(console_handler)

    # File handler to persist logs for auditability
    log_file = "data_acquisition.log"
    file_handler = logging.FileHandler(log_file, mode="a", encoding="utf-8")
    file_handler.setFormatter(formatter)
    logger.addHandler(file_handler)

def ensure_directories(paths: List[str]) -> None:
    """
    Ensures that the specified directories exist. If not, they are created.

    Args:
        paths (List[str]): List of absolute or relative directory paths to verify/create.
    """
    logger = logging.getLogger(__name__)
    for path in paths:
        if not os.path.exists(path):
            os.makedirs(path, exist_ok=True)
            logger.info(f"Created directory: {path}")
        else:
            logger.debug(f"Directory already exists: {path}")
