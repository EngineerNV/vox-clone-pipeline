"""Logging configuration for the application."""

import logging
import sys
from logging.handlers import RotatingFileHandler
from pathlib import Path

LOG_DIR = Path("logs")
LOG_FILE = LOG_DIR / "app.log"


def setup_logging(level: int = logging.INFO) -> None:
    """Configure console + rotating file logging for the app.

    Safe to call multiple times (e.g. on Streamlit reruns) — handlers are
    only attached once.
    """
    root = logging.getLogger()
    if root.handlers:
        return

    root.setLevel(level)
    formatter = logging.Formatter(
        "%(asctime)s [%(levelname)s] %(name)s: %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
    )

    console_handler = logging.StreamHandler(sys.stdout)
    console_handler.setFormatter(formatter)
    root.addHandler(console_handler)

    LOG_DIR.mkdir(parents=True, exist_ok=True)
    file_handler = RotatingFileHandler(LOG_FILE, maxBytes=5_000_000, backupCount=3)
    file_handler.setFormatter(formatter)
    root.addHandler(file_handler)
