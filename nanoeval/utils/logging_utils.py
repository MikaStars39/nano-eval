import logging
import sys

_LOGGER_CONFIGURED = False

def configure_logger(prefix: str = "", level: int = logging.INFO, stream=None):
    global _LOGGER_CONFIGURED
    if _LOGGER_CONFIGURED:
        return

    _LOGGER_CONFIGURED = True

    logging.basicConfig(
        level=level,
        format=f"[%(asctime)s{prefix}] %(filename)s:%(lineno)d - %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
        stream=stream or sys.stderr,
        force=True,
    )