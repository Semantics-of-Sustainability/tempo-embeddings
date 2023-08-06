from dataclasses import dataclass
from dataclasses import field
from typing import Optional
from numpy.typing import ArrayLike


@dataclass
class Highlighting:
    start: int
    end: int
    token_embedding: Optional[ArrayLike] = field(default=None, repr=False)
    umap_embedding: tuple[float, float] = None
