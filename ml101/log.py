import logging
import sys


def init() -> None:
    """Initialize the root logger and the needed logging handlers."""
    logging.root.setLevel(logging.DEBUG)

    formatter = logging.Formatter(
        fmt="%(created)s|%(levelname)s|%(message)s|%(name)s|%(filename)s|%(lineno)s"
    )

    handler = logging.StreamHandler(stream=sys.stdout)
    handler.setLevel(logging.DEBUG)
    handler.setFormatter(formatter)

    logging.root.addHandler(handler)

    logging.getLogger("websockets").setLevel(logging.WARNING)
    logging.getLogger("asyncio").setLevel(logging.WARNING)
    logging.getLogger("ccxt").setLevel(logging.WARNING)
