import logging
from typing import Any


def any_to_int(value: Any) -> int:
    """Convert any value to an int.
    In case of a datetime object, the year is returned.

    Args:
        value: The value to convert.

    Returns:
        The value as an int.

    Raises:
        ValueError: if the value cannot be converted to an int.
    """

    try:
        value = value.date()
    except AttributeError as e:
        logging.debug(f"'{value}' is not a datetime object: %s", e)

    try:
        return value.year
    except AttributeError as e:
        logging.debug(f"'{value}' is not a date object: %s", e)

    try:
        return int(value)
    except (TypeError, ValueError) as e:
        raise ValueError(f"Cannot convert {value} to an int") from e
