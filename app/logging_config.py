"""Console and rotating-file logging for the backend."""

import logging
from logging.handlers import RotatingFileHandler

from app.config import BASE_DIR

_FORMAT = "%(asctime)s %(levelname)-7s %(name)s: %(message)s"


def configure_logging(level: str = "INFO") -> None:
    root = logging.getLogger()
    if getattr(root, "_nimbus_configured", False):
        return

    root.setLevel(level.upper())
    formatter = logging.Formatter(_FORMAT)

    console = logging.StreamHandler()
    console.setFormatter(formatter)
    root.addHandler(console)

    log_dir = BASE_DIR / "logs"
    log_dir.mkdir(exist_ok=True)
    file_handler = RotatingFileHandler(
        log_dir / "app.log", maxBytes=1_000_000, backupCount=3, encoding="utf-8"
    )
    file_handler.setFormatter(formatter)
    root.addHandler(file_handler)

    # Third-party libraries are noisy at INFO.
    for name in ("httpx", "httpcore", "chromadb", "google_genai"):
        logging.getLogger(name).setLevel(logging.WARNING)

    root._nimbus_configured = True  # type: ignore[attr-defined]
