import importlib
from typing import Callable


def get_open_func(compression: str) -> Callable:
    """Get the function to open a file with the given compression.

    Args:
        compression: the compression algorithm to use. If None, no compression is used.

    Returns:
        Callable: the function to open a file with the given compression.
    """
    if compression is None:
        open_f = open
    else:
        try:
            module = importlib.import_module(compression)
            open_f = module.open
        except ImportError:
            raise ValueError(f"Unknown compression algorithm: {compression}")

    return open_f
