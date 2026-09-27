"""Application logging setup."""
import logging
from pathlib import Path

def get_logger(name: str = "signvision") -> logging.Logger:
    logger = logging.getLogger(name)
    if not logger.handlers:
        logger.setLevel(logging.INFO)
        Path("logs").mkdir(exist_ok=True)
        handler = logging.FileHandler(Path("logs") / "signvision.log", encoding="utf-8")
        handler.setFormatter(logging.Formatter("%(asctime)s %(levelname)s %(name)s: %(message)s"))
        logger.addHandler(handler)
        logger.addHandler(logging.StreamHandler())
    return logger
