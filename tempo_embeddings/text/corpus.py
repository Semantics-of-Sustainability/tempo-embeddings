import csv
import gzip
import logging
from pathlib import Path
from typing import Any, Iterable, Optional, TextIO

import joblib
from sklearn.feature_extraction.text import TfidfVectorizer
from tqdm import tqdm

from ..settings import DEFAULT_ENCODING
from .abstractcorpus import AbstractCorpus
from .passage import Passage


class Corpus(AbstractCorpus):
    """A Corpus implementation that holds the concrecte passages and embedings."""

    def __init__(self, passages: list[Passage] = None, label: Optional[Any] = None):
        self._passages: list[Passage] = passages or []
        self._label: Optional[str] = label
        self._vectorizer: TfidfVectorizer = None

    def __add__(self, other: "Corpus", new_label: str = None) -> "Corpus":
        if self.has_embeddings() or other.has_embeddings():
            logging.warning(
                "Dropping existing embeddings to avoid inconsistent vector spaces."
            )

        return Corpus(self._passages + other._passages, new_label)

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
            for batch_start in tqdm(
                range(0, len(self.passages), batch_size),
                desc="Embeddings",
                unit="batch",
                total=len(self.passages) // batch_size + 1,
            ):
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
    def from_lines(
        cls,
        f: TextIO,
        *,
        filter_terms: list[str] = None,
        metadata: dict = None,
        window_size: Optional[int] = None,
        nlp_pipeline=None,
    ):
        """Read input data from an open file handler, one sequence per line."""

        windows: Iterable[Passage] = (
            passage
            for line in f
            for passage in Passage.from_text(
                line,
                metadata=metadata,
                window_size=window_size,
                nlp_pipeline=nlp_pipeline,
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
    def from_csv_files(cls, files: Iterable[Path], desc: str = None, **kwargs):
        """Read input data from multiple CSV files in a directory."""
        return sum(
            (
                cls.from_csv_file(file, **kwargs)
                for file in tqdm(files, desc=desc, unit="file")
            ),
            Corpus(),
        )

    @classmethod
    def from_csv_file(
        cls,
        filepath: Path,
        text_columns: list[str],
        *,
        filter_terms: list[str] = None,
        encoding=DEFAULT_ENCODING,
        compression: Optional[str] = None,
        nlp_pipeline=None,
        **kwargs,
    ):
        """Read input data from a CSV file."""

        open_func = gzip.open if compression == "gzip" else open

        with open_func(filepath, "rt", encoding=encoding) as f:
            return cls.from_csv_stream(
                f,
                text_columns,
                filter_terms=filter_terms,
                nlp_pipeline=nlp_pipeline,
                **kwargs,
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
        nlp_pipeline=None,
        **kwargs,
    ):
        reader = csv.DictReader(file_handler, **kwargs)
        for column in text_columns:
            if column not in reader.fieldnames:
                raise ValueError(
                    f"Text column(s) {text_columns} not found in CSV file '{file_handler.name}'."
                )

        passages = []
        for row in reader:
            # generate separate passage for each text column, sharing the same metadata
            metadata = {
                column: row[column]
                for column in reader.fieldnames
                # skip blank column names and text columns:
                if column and column not in text_columns
            }

            for text_column in text_columns:
                if filter_terms and not any(
                    term.casefold() in row[text_column].casefold()
                    for term in filter_terms
                ):
                    # skip document early, before creating Passage objects
                    continue

                windows: Iterable[Passage] = Passage.from_text(
                    text=row[text_column],
                    metadata=metadata,
                    window_size=window_size,
                    window_overlap=window_overlap,
                    nlp_pipeline=nlp_pipeline,
                )

                if filter_terms:
                    # Highlight terms in passages
                    passages.extend(
                        [
                            passage
                            for window in windows
                            if window.contains_any(filter_terms)
                            for term in filter_terms
                            for passage in window.highlight(term, exact_match=False)
                        ]
                    )

                else:
                    passages.extend(windows)

        return Corpus(passages)
