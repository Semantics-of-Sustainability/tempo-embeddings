import csv
from dataclasses import dataclass
from dataclasses import field
from pathlib import Path
from typing import Iterable
from typing import Optional
from typing import TextIO
import numpy as np
from numpy.typing import ArrayLike
from umap import UMAP
from ..settings import DEFAULT_ENCODING
from .passage import Passage


@dataclass(eq=True, unsafe_hash=True)
class TokenInfo:
    """Data class to store information about a sub-string of a passage."""

    start: int
    end: int

    embedding: Optional[ArrayLike] = field(default=None, hash=False)
    """Embedding of the highlighted token in the passage."""

    embedding_2d: Optional[ArrayLike] = field(default=None, hash=False)


class Corpus:
    def __init__(self, passages: dict[Passage, set[TokenInfo]] = None):
        self._passages: dict[Passage, set[TokenInfo]] = passages or {}
        self._umap_reducer: Optional[UMAP] = None

    def __add__(self, other: "Corpus") -> "Corpus":
        if self._passages.keys() & other._passages.keys():
            # TODO: handle passages with multiple highlightings
            raise NotImplementedError("Passages must be unique")
        return Corpus(passages=self._passages | other._passages)

    def __contains__(self, passage: Passage) -> bool:
        return passage in self._passages

    def __len__(self) -> int:
        return len(self._passages)

    def __repr__(self) -> str:
        return f"Corpus({self._passages!r})"

    def __eq__(self, __value: object) -> bool:
        return __value._passages == self._passages

    @property
    def passages(self) -> dict[Passage, set[TokenInfo]]:
        return self._passages

    @property
    def token_infos(self) -> Iterable[TokenInfo]:
        """Returns an iterable over all TokenInfo objects, flattened from all passages."""
        return (
            token_info
            for token_infos in self.passages.values()
            for token_info in token_infos
        )

    def all_embeddings(self) -> ArrayLike:
        if not self.has_embeddings():
            raise ValueError("Corpus does not have embeddings")
        return np.array([token_info.embedding for token_info in self.token_infos])

    def has_embeddings(self) -> bool:
        """Returns True if all TokenInfo objects have an embedding."""
        # Either all or none of the TokenInfo objects have an embedding
        return any(token_info.embedding is not None for token_info in self.token_infos)

    def find(self, token: str) -> Iterable[tuple[Passage, int]]:
        for passage in self._passages:
            for match_index in passage.findall(token):
                yield (passage, match_index)

    def subcorpus(self, token: str) -> "Corpus":
        """Uses find() to generate a new Corpus object with matching passages and
        highlightings."""

        # TODO make this more efficient (using map/reduce)

        passages = {}
        for passage, match_index in self.find(token):
            passages.setdefault(passage, set()).add(
                TokenInfo(start=match_index, end=match_index + len(token))
            )

        return Corpus(passages)

    def umap(self):
        if not self.has_embeddings():
            raise ValueError("Corpus has no embeddings.")

        if self._umap_reducer is None:
            reducer = UMAP(metric="cosine")
            reducer.fit(self.all_embeddings())
            self._umap_reducer = reducer

        return self._umap_reducer

    def umap_embeddings(self):
        return self.umap().transform(self.all_embeddings())

    @classmethod
    def from_lines(cls, f: TextIO, metadata: dict = None):
        """Read input data from an open file handler, one sequence per line."""
        return Corpus.from_passages((Passage(line, metadata) for line in f))

    @classmethod
    def from_file(cls, filepath: Path, encoding=DEFAULT_ENCODING):
        """Read input data from a file, one sequence per line."""
        with open(filepath, "rt", encoding=encoding) as f:
            return Corpus.from_lines(f)

    @classmethod
    def from_csv(
        cls, filepath: Path, text_columns: list[str], encoding=DEFAULT_ENCODING
    ):
        """Read input data from a CSV file."""
        with open(filepath, "rt", encoding=encoding) as f:
            reader = csv.DictReader(f)
            if not all(column in reader.fieldnames for column in text_columns):
                raise ValueError("Not all text columns found in CSV file.")

            passages = []
            for row in reader:
                metadata = {
                    column: row[column]
                    for column in reader.fieldnames
                    if column not in text_columns
                }
                for text_column in text_columns:
                    passages.append(Passage(row[text_column], metadata))
        return Corpus.from_passages(passages)

    @classmethod
    def from_passages(cls, passages: Iterable[Passage]):
        """Create a Corpus from a list of passages."""
        return cls({passage: set() for passage in passages})
