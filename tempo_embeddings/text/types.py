from dataclasses import dataclass
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


@dataclass(eq=True, unsafe_hash=True)
class TokenInfo:
    """Data class to store information about a sub-string of a passage."""

    start: int
    end: int
