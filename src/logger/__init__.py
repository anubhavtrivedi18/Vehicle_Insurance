import logging
from logging.handlers import RotatingFileHandler
from pathlib import Path
from datetime import datetime


# ============================================================
# Project paths
# ============================================================

# __file__:
# Vehicle_Insurance/src/logger/__init__.py
#
# parents[0] -> logger
# parents[1] -> src
# parents[2] -> Vehicle_Insurance (project root)

PROJECT_ROOT = Path(__file__).resolve().parents[2]

LOG_DIR = PROJECT_ROOT / "logs"

LOG_FILE = f"{datetime.now().strftime('%m_%d_%Y_%H_%M_%S')}.log"

LOG_FILE_PATH = LOG_DIR / LOG_FILE


# ============================================================
# Logging configuration
# ============================================================

MAX_LOG_SIZE = 5 * 1024 * 1024  # 5 MB
BACKUP_COUNT = 3

LOG_DIR.mkdir(parents=True, exist_ok=True)


def configure_logger():
    """
    Configure application logging.

    Logs are written to:
        project_root/logs/

    Logging is available in:
        1. Console
        2. Rotating log file
    """

    logger = logging.getLogger()

    # Prevent duplicate handlers if this module is imported
    # multiple times.
    if logger.handlers:
        return logger

    logger.setLevel(logging.DEBUG)

    # --------------------------------------------------------
    # Formatter
    # --------------------------------------------------------

    formatter = logging.Formatter(
        "[%(asctime)s] %(name)s - %(levelname)s - %(message)s"
    )

    # --------------------------------------------------------
    # File Handler
    # --------------------------------------------------------

    file_handler = RotatingFileHandler(
        LOG_FILE_PATH,
        maxBytes=MAX_LOG_SIZE,
        backupCount=BACKUP_COUNT,
        encoding="utf-8"
    )

    file_handler.setLevel(logging.DEBUG)
    file_handler.setFormatter(formatter)

    # --------------------------------------------------------
    # Console Handler
    # --------------------------------------------------------

    console_handler = logging.StreamHandler()

    console_handler.setLevel(logging.INFO)
    console_handler.setFormatter(formatter)

    # --------------------------------------------------------
    # Add handlers
    # --------------------------------------------------------

    logger.addHandler(file_handler)
    logger.addHandler(console_handler)

    return logger


# ============================================================
# Initialize logger
# ============================================================

configure_logger()