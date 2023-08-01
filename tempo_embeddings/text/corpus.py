import csv
import gzip
import logging
from collections import Counter
from collections import defaultdict
from itertools import groupby
from pathlib import Path
from typing import Any
from typing import Iterable
from typing import Optional
from typing import TextIO
import joblib
import numpy as np
import pandas as pd
import umap.plot
from numpy.typing import ArrayLike
from scipy.spatial.distance import cosine
from sklearn.cluster import HDBSCAN
from umap import UMAP
from ..embeddings.model import TransformerModelWrapper
from ..settings import DEFAULT_ENCODING
from .highlighting import Highlighting
from .passage import Passage


class Corpus:
    def __init__(
        self,
        passages: list[Passage] = None,
        highlightings: list[Highlighting] = None,
        model: Optional[TransformerModelWrapper] = None,
        umap: Optional[UMAP] = None,
    ):
        self._passages: list[Passage] = passages or []
        self._highlightings: list[Highlighting] = highlightings or []
        self._model: Optional[TransformerModelWrapper] = model

        self._umap: Optional[UMAP] = umap

    def __add__(self, other: "Corpus") -> "Corpus":
        # TODO: does model.__eq__() work?
        if self._model != other._model:
            raise ValueError("Cannot add two corpora with different embeddings models")

        # Dropping previously computed UMAP embeddings
        return Corpus(
            passages=self._passages + other._passages,
            highlightings=self._highlightings + other._highlightings,
            model=self._model,
        )

    def __contains__(self, passage: Passage) -> bool:
        return passage in self._passages

    def __len__(self) -> int:
        return len(self._passages)

    def __repr__(self) -> str:
        return f"Corpus({self._passages[:10]!r})"

    def __eq__(self, other: object) -> bool:
        return (
            self._model == other._model
            and other._passages == self._passages
            and other._highlightings == self._highlightings
        )

    @property
    def passages(self) -> list[Passage]:
        return self._passages

    def texts(self):
        return [passage.text for passage in self._passages]

    def passages_untokenized(self) -> list[Passage]:
        return [passage for passage in self._passages if passage.tokenization is None]

    def passages_unembeddened(self) -> list[Passage]:
        return [passage for passage in self._passages if passage.embeddings is None]

    @property
    def embeddings_model(self) -> Optional[TransformerModelWrapper]:
        return self._model

    @embeddings_model.setter
    def embeddings_model(self, value: TransformerModelWrapper):
        self._model = value

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
        for highlighting in self._highlightings:
            try:
                yield highlighting.passage.get_metadata(key)
            except KeyError as e:
                raise ValueError(
                    f"Passage missing metadata key: {highlighting.passage}"
                ) from e

    def set_metadatas(self, key, value):
        """Sets a metadata key to a value for all passages.

        Args:
            key: The metadata key to set.
            value: The value to set the metadata key to.
        """

        for passage in self.passages:
            passage.set_metadata(key, value)

    def compute_embeddings(self):
        if self._model is None:
            raise ValueError("Corpus does not have a model.")
        self._model.compute_embeddings(self)

    def _token_embeddings(self) -> ArrayLike:
        if not self._highlightings:
            logging.warning("Corpus has no highlightings")
            return np.array([])

        if not self.has_embeddings():
            # batch-compute embeddings for all passages in corpus.
            self.compute_embeddings()

        return [
            highlighting.get_token_embedding() for highlighting in self._highlightings
        ]

    def has_embeddings(self, validate=False) -> bool:
        """Returns True embeddings have been computed for the corpus.

        Args:
            validate: If True, validates that all Passage objects have an embedding.

        Returns:
            True if embeddings have been computed
        """
        func = all if validate else any
        return func(passage.embeddings is not None for passage in self.passages)

    def has_tokenizations(self, validate: bool = False) -> bool:
        func = all if validate else any
        return func(passage.tokenization is not None for passage in self.passages)

    def subcorpus(self, token: str, **metadata) -> "Corpus":
        """Generate a new Corpus object with matching passages and highlightings.

        Args:
            token: The token to search for.
            metadata: Metadata fields to match against
        """

        matches: list[Highlighting] = [
            highlighting
            for passage in self._passages
            for highlighting in passage.findall(token)
            if all(
                highlighting.passage.metadata[key] == value
                for key, value in metadata.items()
            )
        ]

        passages = [passage for passage, _ in groupby(matches, lambda x: x.passage)]

        # Dropping unmatched passages and Umap
        return Corpus(passages, matches, self._model)

    def frequent_words(
        self,
        stop_words: set[str] = None,
        n: int = 10,
        *,
        use_tokenizer: bool = False,
    ) -> list[tuple[str, int]]:
        """The most common words in the context of a token in all passages."""

        if stop_words is None:
            stop_words = set()

        if any(word.casefold() != word for word in stop_words):
            raise ValueError("Stop words should be lowercase")

        return Counter(
            (
                word
                for passage in self._passages
                for word in passage.words(use_tokenizer)
                if word.casefold() not in stop_words
            )
        ).most_common(n)

    def mean(self) -> ArrayLike:
        """The mean for all passage embeddings."""
        return np.array(self._token_embeddings()).mean(axis=0)

    def cosine(self, other: "Corpus") -> float:
        """The cosine distance between the mean of this corpus and another."""
        return cosine(self.mean(), other.mean())

    def cluster(self, *, overwrite: bool = False, **kwargs) -> None:
        """Clusters the corpus using HDBSCAN and assigns labels to highlightings.
        
        Outliers are assigend the label -1.
        """

        labels = HDBSCAN(**kwargs).fit_predict(self.umap_embeddings())
        for label, highlighting in zip(labels, self._highlightings, strict=True):
            if highlighting.label is not None and not overwrite:
                raise ValueError(f"Highlight already has label: {highlighting}")

            highlighting.label = label

    def clusters(self, **kwargs) -> Iterable["Corpus"]:
        """Clusters the corpus using HDBSCAN and returns a list of subcorpora.

        Outliers are dropped.
        """
        labels = HDBSCAN(**kwargs).fit_predict(self.umap_embeddings())

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
                model=self._model,
                umap=self._umap,
            )
            for label, highlightings in corpora.items()
            if label >= 0
        ]

    def _labels(self) -> Optional[list]:
        labels = [highlighting.label for highlighting in self._highlightings]
        if not any(labels):
            return None
        return labels

    def plot(self, **kwargs):
        labels = kwargs.get("labels", self._labels())
        if labels is not None:
            labels = np.array(labels)

        umap.plot.points(self.umap, labels=labels, **kwargs)

    def hover_datas(self, metadata_keys=None) -> list[dict[str, Any]]:
        return [
            highlighting.hover_data(metadata_keys=metadata_keys)
            for highlighting in self._highlightings
        ]

    def interactive_plot(self, **kwargs):
        hover_data = pd.DataFrame(self.hover_datas())

        labels = kwargs.get("labels", self._labels())
        if labels is not None:
            labels = np.array(labels)

        return umap.plot.interactive(
            self.umap, labels=labels, hover_data=hover_data, **kwargs
        )

    @property
    def umap(self):
        if self._umap is None:
            umap = UMAP(metric="cosine")
            umap.fit(np.array(self._token_embeddings()))
            self._umap = umap
        return self._umap

    def umap_embeddings(self):
        return self.umap.embedding_

    def highlighted_texts(self, metadata_fields: Iterable[str] = None) -> list[str]:
        """Returns an iterable over all highlightings."""
        texts = [
            highlighting.text(metadata_fields) for highlighting in self._highlightings
        ]
        assert all(text.strip() for text in texts), "Empty text."
        return texts

    def save(self, filepath: Path):
        """Save the corpus to a file."""
        # TODO: (optionally) do not save model, just the name for reloading on-demand
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
        metadata: dict = None,
        model: Optional[TransformerModelWrapper] = None,
    ):
        """Read input data from an open file handler, one sequence per line."""
        return Corpus([Passage(line, metadata, model) for line in f], model=model)

    @classmethod
    def from_lines_file(
        cls,
        filepath: Path,
        encoding=DEFAULT_ENCODING,
        model: Optional[TransformerModelWrapper] = None,
    ):
        """Read input data from a file, one sequence per line."""
        with open(filepath, "rt", encoding=encoding) as f:
            return Corpus.from_lines(f, model=model)

    @classmethod
    def from_csv_file(
        cls,
        filepath: Path,
        text_columns: list[str],
        model: Optional[TransformerModelWrapper] = None,
        *,
        encoding=DEFAULT_ENCODING,
        compression: Optional[str] = None,
        **kwargs,
    ):
        """Read input data from a CSV file."""
        open_func = gzip.open if compression == "gzip" else open

        with open_func(filepath, "rt", encoding=encoding) as f:
            return cls.from_csv_stream(f, text_columns, model, **kwargs)

    @classmethod
    def from_csv_stream(
        cls,
        file_handler,
        text_columns: list[str],
        model: Optional[TransformerModelWrapper] = None,
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
                        text=row[text_column],
                        model=model,
                        metadata=metadata,
                        window_size=window_size,
                        window_overlap=window_overlap,
                    )
                )
        return Corpus(passages, model=model)
