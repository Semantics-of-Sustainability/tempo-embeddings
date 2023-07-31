import csv
import gzip
import logging
from collections import defaultdict
from itertools import groupby
from pathlib import Path
from typing import Any
from typing import Iterable
from typing import Optional
from typing import TextIO
import joblib
import numpy as np
from numpy.typing import ArrayLike
from scipy.spatial.distance import cosine
from sklearn.cluster import HDBSCAN
from umap import UMAP
from ..settings import DEFAULT_ENCODING
from .passage import Passage
from .types import Highlighting


class Corpus:
    def __init__(
        self,
        passages: list[Passage] = None,
        highlightings: list[Highlighting] = None,
        embeddings_model_name: Optional[str] = None,
    ):
        self._passages: list[Passage] = passages or []
        self._highlightings: list[Highlighting] = highlightings or []
        self._embeddings_model_name: Optional[str] = embeddings_model_name

        self._umap_embeddings: Optional[np.ndarray] = None

    def __add__(self, other: "Corpus") -> "Corpus":
        if self._embeddings_model_name != other._embeddings_model_name:
            raise ValueError("Cannot add two corpora with different embeddings models")

        # Dropping previously computed UMAP embeddings
        return Corpus(
            passages=self._passages + other._passages,
            highlightings=self._highlightings + other._highlightings,
            embeddings_model_name=self._embeddings_model_name,
        )

    def __contains__(self, passage: Passage) -> bool:
        return passage in self._passages

    def __len__(self) -> int:
        return len(self._passages)

    def __repr__(self) -> str:
        return f"Corpus({self._passages[:10]!r})"

    def __eq__(self, other: object) -> bool:
        return (
            self._embeddings_model_name == other._embeddings_model_name
            and other._passages == self._passages
            and other._highlightings == self._highlightings
        )

    @property
    def passages(self) -> list[Passage]:
        return self._passages

    def texts(self):
        return [passage.text for passage in self._passages]

    @property
    def embeddings_model_name(self) -> Optional[str]:
        return self._embeddings_model_name

    @embeddings_model_name.setter
    def embeddings_model_name(self, value: str):
        self._embeddings_model_name = value

    @property
    def highlightings(self) -> list[Highlighting]:
        return self._highlightings

    def has_metadata(self, key: str, strict=False) -> bool:
        """Returns True if the corpus has a metadata key.

        Args:
            key: The metadata key to check for.
            strict: If True, returns True only if all passages have the key.
        """
        condition = all if strict else any
        return condition(passage.has_metadata(key) for passage in self.passages)

    def get_highlighting_metadatas(self, key: str) -> Iterable[Any]:
        """Returns an iterable over all metadata values for a key
        for each _highlighted_ token."""
        for _, _, passage in self._highlightings:
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
                passage.token_embedding(start, end)
                for start, end, passage in self._highlightings
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

    def _find(self, token: str) -> Iterable[Highlighting]:
        for passage in self._passages:
            for match_index in passage.findall(token):
                yield Highlighting(
                    start=match_index, end=match_index + len(token), passage=passage
                )

    def subcorpus(self, token: str, **metadata) -> "Corpus":
        """Generate a new Corpus object with matching passages and highlightings.

        Args:
            token: The token to search for.
            metadata: Metadata fields to match against
        """

        matches: list[Highlighting] = [
            highlighting
            for highlighting in self._find(token)
            if all(
                highlighting.passage.metadata[key] == value
                for key, value in metadata.items()
            )
        ]

        passages = [passage for passage, _ in groupby(matches, lambda x: x.passage)]

        # Dropping unmatched passages
        return Corpus(passages, matches, self._embeddings_model_name)

    def context_words(self, token: str):
        """The most common words in the context of a token in all passages."""

    def mean(self) -> ArrayLike:
        """The mean for all passage embeddings."""
        return self._token_embeddings().mean(axis=0)

    def cosine(self, other: "Corpus") -> float:
        """The cosine distance between the mean of this corpus and another."""
        return cosine(self.mean(), other.mean())

    def clusters(self, **kwargs) -> Iterable["Corpus"]:
        labels = HDBSCAN(**kwargs).fit_predict(self._token_embeddings())

        # TODO: handle cluster with outliers (label -1)
        corpora = defaultdict(list)
        for label, highlighting in zip(labels, self._highlightings, strict=True):
            corpora[label].append(highlighting)

        if corpora[-1]:
            logging.warning("Found %d outliers", len(corpora[-1]))

        return [
            Corpus(
                passages=[p for p, _ in groupby(highlightings, lambda h: h.passage)],
                highlightings=highlightings,
                embeddings_model_name=self._embeddings_model_name,
            )
            for label, highlightings in corpora.items()
            if label >= 0
        ]

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
            passage.highlighted_text(start, end, metadata_fields)
            for start, end, passage in self._highlightings
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
        return Corpus([Passage(line, metadata) for line in f])

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
        return Corpus(passages)
