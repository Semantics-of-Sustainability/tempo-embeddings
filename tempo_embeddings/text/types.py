from collections import namedtuple
from typing import TypedDict


class Metadata(TypedDict, total=False):
    """Metadata for a text sequence."""

    id: str
    """Unique identifier for the text sequence."""
    label: str
    """Label for the text sequence."""
    source: str
    """Source for the text sequence."""
    year: int
    """Year of publication for the text sequence."""


Highlighting = namedtuple("Highlighting", ["start", "end", "passage"])
