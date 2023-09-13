import csv
import gzip
import logging
from pathlib import Path
from typing import Any
from typing import Iterable
from typing import Optional
from typing import TextIO
import joblib
from numpy.typing import ArrayLike
from ..settings import DEFAULT_ENCODING
from .abstractcorpus import AbstractCorpus
from .passage import Passage


class Corpus(AbstractCorpus):
    """A Corpus implementation that holds the concrecte passages and embedings."""

    def __init__(
        self,
        passages: list[Passage] = None,
        label: Optional[Any] = None,
        embeddings=None,
    ):
        self._passages: list[Passage] = passages or []
        self._label: Optional[str] = label
        self._embeddings = embeddings

        self._validate_embeddings()

    def __add__(self, other: "Corpus", new_label: str = None) -> "Corpus":
        match (self._embeddings, other._embeddings):
            case (None, None):
                embeddings = None
            case _:
                raise NotImplementedError()

        return Corpus(self._passages + other._passages, new_label, embeddings)

    def __repr__(self) -> str:
        return f"Corpus({self._label!r}, {self._passages[:10]!r})"

    @property
    def passages(self) -> list[Passage]:
        return self._passages

    @property
    def embeddings(self):
        return self._embeddings

    @embeddings.setter
    def embeddings(self, embeddings: ArrayLike):
        self._embeddings = embeddings
        self._validate_embeddings()

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
    def from_lines(
        cls,
        f: TextIO,
        *,
        filter_terms: list[str] = None,
        metadata: dict = None,
        window_size: Optional[int] = None,
    ):
        """Read input data from an open file handler, one sequence per line."""
        if filter_terms and len(filter_terms) > 1:
            raise NotImplementedError(
                "Highlighting/embedding multiple filter terms not yet implemented."
            )

        windows: Iterable[Passage] = (
            passage
            for line in f
            for passage in Passage.from_text(
                line, metadata=metadata, window_size=window_size
            )
            if passage.contains_any(filter_terms)
        )

        if filter_terms:
            passages = [
                highlighted
                for window in windows
                for term in filter_terms
                for highlighted in window.highlight(term, exact_match=False)
            ]
        else:
            logging.warning("No filter terms defined, hence no highlighting.")
            passages = list(windows)

        return Corpus(passages, label="; ".join(filter_terms) if filter_terms else None)

    @classmethod
    def from_lines_file(
        cls,
        filepath: Path,
        *,
        filter_terms: list[str] = None,
        encoding=DEFAULT_ENCODING,
    ):
        """Read input data from a file, one sequence per line."""

        with open(filepath, "rt", encoding=encoding) as f:
            return Corpus.from_lines(f, filter_terms=filter_terms)

    @classmethod
    def from_csv_file(
        cls,
        filepath: Path,
        text_columns: list[str],
        *,
        filter_terms: list[str] = None,
        encoding=DEFAULT_ENCODING,
        compression: Optional[str] = None,
        **kwargs,
    ):
        """Read input data from a CSV file."""

        open_func = gzip.open if compression == "gzip" else open

        with open_func(filepath, "rt", encoding=encoding) as f:
            return cls.from_csv_stream(
                f, text_columns, filter_terms=filter_terms, **kwargs
            )

    @classmethod
    def from_csv_stream(
        cls,
        file_handler,
        text_columns: list[str],
        *,
        filter_terms: list[str] = None,
        window_size: int = 1000,
        window_overlap: int = 0,
        **kwargs,
    ):
        if filter_terms and len(filter_terms) > 1:
            raise NotImplementedError(
                "Highlighting/embedding multiple filter terms not yet implemented."
            )

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
                if filter_terms and not any(
                    term.casefold() in row[text_column].casefold()
                    for term in filter_terms
                ):
                    # skip document early, before creating Passage objects
                    continue

                for window in Passage.from_text(
                    text=row[text_column],
                    metadata=metadata,
                    window_size=window_size,
                    window_overlap=window_overlap,
                ):
                    if filter_terms and window.contains_any(filter_terms):
                        for term in filter_terms:
                            passages.extend(window.highlight(term, exact_match=False))

        return Corpus(passages)
