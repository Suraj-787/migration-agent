import sys

from loguru import logger


def configure_logging(*, level: str = "INFO", json_logs: bool = False) -> None:
    logger.remove()
    if json_logs:
        logger.add(sys.stderr, level=level, serialize=True, enqueue=True)
    else:
        logger.add(
            sys.stderr,
            level=level,
            format=(
                "<green>{time:YYYY-MM-DD HH:mm:ss.SSS}</green> | "
                "<level>{level: <8}</level> | "
                "<cyan>{name}</cyan>:<cyan>{line}</cyan> — "
                "<level>{message}</level>"
            ),
            colorize=True,
            enqueue=True,
        )
