import logging
from pathlib import Path
from typing import Optional
from .config import ConfigLoader


def setup_logger(cfg: ConfigLoader, name: str = __name__) -> logging.Logger:
    log_cfg = cfg.get("logging", {})
    log_file = Path(log_cfg.get("file", "logs/pipeline.log"))
    log_file.parent.mkdir(parents=True, exist_ok=True)

    # Configure root logger
    logging.basicConfig(
        filename=log_file,
        level=getattr(logging, log_cfg.get("level", "INFO").upper()),
        format="%(asctime)s [%(levelname)s] %(name)s: %(message)s"
    )

    # Add console handler
    console = logging.StreamHandler()
    console.setLevel(getattr(logging, log_cfg.get("level", "INFO").upper()))
    console.setFormatter(logging.Formatter(
        "%(asctime)s [%(levelname)s] %(name)s: %(message)s"))
    logging.getLogger().addHandler(console)

    return logging.getLogger(name)
