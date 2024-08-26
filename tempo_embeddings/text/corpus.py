import csv
import gzip
import logging
from pathlib import Path
from typing import Any, Iterable, Optional

import joblib
from sklearn.feature_extraction.text import TfidfVectorizer

from ..settings import DEFAULT_ENCODING
from .abstractcorpus import AbstractCorpus
from .passage import Passage
from .segmenter import Segmenter


class Corpus(AbstractCorpus):
    """A Corpus implementation that holds the concrecte passages and embedings."""

    def __init__(self, passages: list[Passage] = None, label: Optional[Any] = None):
        super().__init__()

        self._passages: list[Passage] = passages or []
        self._label: Optional[str] = label
        self._vectorizer: TfidfVectorizer = None

    def __add__(self, other: "Corpus") -> "Corpus":
        new_label = self.label if self.label == other.label else None
        return Corpus(self._passages + other._passages, label=new_label)

    def __len__(self) -> int:
        """Return the number of passages in the corpus.

        Returns:
            int: The number of passages in the corpus.
        """
        return len(self._passages)

    def __repr__(self) -> str:
        return f"Corpus({self._label!r}, {self._passages[:10]!r})"

    @property
    def passages(self) -> list[Passage]:
        return self._passages

    @property
    def vectorizer(self) -> TfidfVectorizer:
        if self._vectorizer is None:
            self._vectorizer = AbstractCorpus.tfidf_vectorizer(self.passages)
        return self._vectorizer

    def batches(self, batch_size: int) -> Iterable[list[Passage]]:
        if batch_size <= 1:
            yield self.passages
        else:
            for batch_start in range(0, len(self.passages), batch_size):
                yield self.passages[batch_start : batch_start + batch_size]

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
    def from_csv_files(cls, files: Iterable[Path], **kwargs):
        """Read input data from multiple CSV files in a directory."""
        return sum((cls.from_csv_file(file, **kwargs) for file in files), Corpus())

    @classmethod
    def from_csv_file(
        cls,
        filepath: Path,
        text_columns: list[str],
        *,
        segmenter: Segmenter,
        filter_terms: list[str] = None,
        encoding=DEFAULT_ENCODING,
        compression: Optional[str] = None,
        **dict_reader_kwargs,
    ):
        """Read input data from a CSV file."""

        open_func = gzip.open if compression == "gzip" else open

        with open_func(filepath, "rt", encoding=encoding) as f:
            try:
                return cls.from_csv_stream(
                    f,
                    text_columns,
                    filter_terms=filter_terms,
                    segmenter=segmenter,
                    **dict_reader_kwargs,
                )
            except EOFError as e:
                logging.error(f"Error reading file '{filepath}': {e}")
                return Corpus()

    @classmethod
    def from_csv_stream(
        cls,
        file_handler,
        text_columns: list[str],
        *,
        segmenter: Segmenter,
        filter_terms: list[str] = None,
        **dict_reader_kwargs,
    ):
        reader = csv.DictReader(file_handler, **dict_reader_kwargs)

        passages = segmenter.passages_from_dict_reader(
            reader,
            provenance=file_handler.name,
            text_columns=text_columns,
            filter_terms=filter_terms,
        )
        return Corpus(passages)
