import csv
import gzip
import logging
from pathlib import Path
from typing import Any
from typing import Iterable
from typing import Optional
from typing import TextIO
import joblib
import numpy as np
from numpy.typing import ArrayLike
from umap import UMAP
from ..settings import DEFAULT_ENCODING
from .passage import Passage
from .types import TokenInfo


class Corpus:
    def __init__(self, passages: dict[Passage, set[TokenInfo]] = None):
        self._passages: dict[Passage, set[TokenInfo]] = passages or {}
        self._umap_embeddings: Optional[np.ndarray] = None

        self._embeddings_model_name: Optional[str] = None

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
        return f"Corpus({list(self._passages.items())[:10]!r})"

    def __eq__(self, other: object) -> bool:
        return (
            self._embeddings_model_name == other._embeddings_model_name
            and other._passages == self._passages
        )

    @property
    def passages(self) -> dict[Passage, set[TokenInfo]]:
        return self._passages

    @property
    def embeddings_model_name(self) -> Optional[str]:
        return self._embeddings_model_name

    @embeddings_model_name.setter
    def embeddings_model_name(self, value: str):
        self._embeddings_model_name = value

    def _token_passages(self) -> Iterable[tuple[Passage, TokenInfo]]:
        """Returns an iterable over all passages.

        Yields one instance per token highlighting.
        If a passage has no highlightings, it is not yielded.
        If a passage has multiple highlightings, it is yielded multiple times.
        """
        for passage, token_infos in self._passages.items():
            if not token_infos:
                logging.warning("Passage %s has no highlightings", passage)

            for token_info in token_infos:
                yield passage, token_info

    @property
    def token_infos(self) -> Iterable[TokenInfo]:
        """Returns an iterable over all TokenInfo objects,
        flattened from all passages."""
        for token_infos in self.passages.values():
            yield from token_infos

    def has_metadata(self, key: str, strict=False) -> bool:
        """Returns True if the corpus has a metadata key.

        Args:
            key: The metadata key to check for.
            strict: If True, returns True only if all passages have the key.
        """
        condition = all if strict else any
        return condition(passage.has_metadata(key) for passage in self.passages)

    def get_token_metadatas(self, key: str) -> Iterable[Any]:
        """Returns an iterable over all values
        for a given metadata key for each token."""
        for passage, _ in self._token_passages():
            try:
                yield passage.get_metadata(key)
            except KeyError as e:
                raise ValueError(f"Passage missing metadata key: {passage}") from e

    def set_metadatas(self, key, value):
        """Sets a metadata key to a value for all passages.

        Args:
            key: The metadata key to set.
            value: The value to set the metadata key to.
        """

        for passage in self.passages:
            passage.set_metadata(key, value)

    def _token_embeddings(self) -> ArrayLike:
        if not self.has_embeddings():
            raise ValueError("Corpus does not have embeddings")

        return np.array(
            [
                passage.token_embedding(token_info)
                for passage, token_info in self._token_passages()
            ]
        )

    def has_embeddings(self, validate=False) -> bool:
        """Returns True embeddings have been computed for the corpus.

        Args:
            validate: If True, validates that all TokenInfo objects have an embedding.

        Returns:
            True if embeddings have been computed
        """
        if self.embeddings_model_name is not None:
            return (not validate) or all(
                passage.embedding is not None for passage in self.passages
            )
        return False

    def _find(self, token: str) -> Iterable[tuple[Passage, int]]:
        # FIXME: skip sub-strings matching within words?
        for passage in self._passages:
            for match_index in passage.findall(token):
                yield (passage, match_index)

    def subcorpus(self, token: str) -> "Corpus":
        """Generate a new Corpus object with matching passages and highlightings."""

        # TODO make this more efficient (using map/reduce)

        passages = {}
        for passage, match_index in self._find(token):
            passages.setdefault(passage, set()).add(
                TokenInfo(start=match_index, end=match_index + len(token))
            )

        return Corpus(passages)

    def umap_embeddings(self):
        if self._umap_embeddings is None:
            umap = UMAP(metric="cosine")
            self._umap_embeddings = umap.fit_transform(self._token_embeddings())

        return self._umap_embeddings

    def highlighted_texts(self, metadata_fields: Iterable[str] = None) -> list[str]:
        """Returns an iterable over all highlighted texts, flattened from all passages.

        A passage is returned multiple times if it has multiple highlightings.
        """
        return [
            passage.highlighted_text(token_info, metadata_fields)
            for passage, token_info in self._token_passages()
        ]

    def save(self, filepath: Path):
        """Save the corpus to a file."""
        with open(filepath, "wb") as f:
            joblib.dump(self, f)

    @classmethod
    def load(cls, filepath: Path):
        """Load the corpus from a file."""
        with open(filepath, "rb") as f:
            corpus = joblib.load(f)

        if not isinstance(corpus, cls):
            raise TypeError(f"Expected {cls}, got {type(corpus)}")

        return corpus

    @classmethod
    def from_lines(cls, f: TextIO, metadata: dict = None):
        """Read input data from an open file handler, one sequence per line."""
        return Corpus.from_passages((Passage(line, metadata) for line in f))

    @classmethod
    def from_lines_file(cls, filepath: Path, encoding=DEFAULT_ENCODING):
        """Read input data from a file, one sequence per line."""
        with open(filepath, "rt", encoding=encoding) as f:
            return Corpus.from_lines(f)

    @classmethod
    def from_csv_file(
        cls,
        filepath: Path,
        text_columns: list[str],
        encoding=DEFAULT_ENCODING,
        compression: Optional[str] = None,
        **kwargs,
    ):
        """Read input data from a CSV file."""
        open_func = gzip.open if compression == "gzip" else open

        with open_func(filepath, "rt", encoding=encoding) as f:
            return cls.from_csv_stream(f, text_columns, **kwargs)

    @classmethod
    def from_csv_stream(
        cls,
        file_handler,
        text_columns: list[str],
        *,
        window_size: int = 1000,
        window_overlap: int = 0,
        **kwargs,
    ):
        reader = csv.DictReader(file_handler, **kwargs)
        if not all(column in reader.fieldnames for column in text_columns):
            raise ValueError("Not all text columns found in CSV file.")

        passages = []
        for row in reader:
            # generate separate passage for each text column, sharing the same metadata
            metadata = {
                column: row[column]
                for column in reader.fieldnames
                if column not in text_columns
            }

            for text_column in text_columns:
                passages.extend(
                    Passage.from_text(
                        row[text_column],
                        metadata=metadata,
                        window_size=window_size,
                        window_overlap=window_overlap,
                    )
                )
        return Corpus.from_passages(passages)

    @classmethod
    def from_passages(cls, passages: Iterable[Passage]):
        """Create a Corpus from a list of passages."""
        return cls({passage: set() for passage in passages})
