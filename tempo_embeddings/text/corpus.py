import csv
import gzip
import logging
from collections import Counter
from collections import defaultdict
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
        model: Optional[TransformerModelWrapper] = None,
        umap: Optional[UMAP] = None,
        label: Optional[Any] = None,
    ):
        self._passages: list[Passage] = passages or []
        self._model: Optional[TransformerModelWrapper] = model

        self._umap: Optional[UMAP] = umap
        self._label: str = str(label) or ""

    def __add__(self, other: "Corpus") -> "Corpus":
        # TODO: does model.__eq__() work?
        if self._model != other._model:
            raise ValueError("Cannot add two corpora with different embeddings models")

        # Dropping previously computed UMAP embeddings
        return Corpus(passages=self._passages + other._passages, model=self._model)

    def __contains__(self, passage: Passage) -> bool:
        return passage in self._passages

    def __len__(self) -> int:
        return len(self._passages)

    def __repr__(self) -> str:
        return f"Corpus({self._label!r} {self._passages[:10]!r})"

    def __eq__(self, other: object) -> bool:
        return self._model == other._model and other._passages == self._passages

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
        return [
            highlighting
            for passage in self.passages
            for highlighting in passage.highlightings
        ]

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
        for each highlighted token."""

        return [
            passage.get_metadata(key)
            for passage in self.passages
            for _ in passage.highlightings
        ]

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
            logging.warning("Corpus does not have a model.")
        self._model.compute_embeddings(self)

    def _token_embeddings(self) -> list[ArrayLike]:
        if not self.highlightings:
            logging.warning("Corpus has no highlightings")
            return []

        if not self.has_embeddings():
            # batch-compute embeddings for all passages in corpus.
            self.compute_embeddings()

        return [
            embedding
            for passage in self.passages
            for embedding in passage.token_embeddings()
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

        highlighted_passages = [
            passage
            for passage in self.passages
            if all(
                passage.metadata.get(key) == value for key, value in metadata.items()
            )
            and passage.add_highlightings(token)
        ]

        # Dropping unmatched passages
        return Corpus(highlighted_passages, self._model, self._umap)

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

    def clusters(self, **kwargs) -> Iterable["Corpus"]:
        """Clusters the corpus using HDBSCAN and returns a list of subcorpora."""
        labels: list[int] = (
            HDBSCAN(**kwargs).fit_predict(self.umap_embeddings()).astype(int).tolist()
        )

        clusters: dict[int, list[Passage]] = defaultdict(list)

        for passage in self.passages:
            highlightings = passage.highlightings
            assert highlightings, "No highlightings in passage"

            passage_labels: list[int] = []
            unique_labels: list[int] = []

            for _ in highlightings:
                label = labels.pop(0)
                passage_labels.append(label)
                if label not in unique_labels:
                    unique_labels.append(label)

            for label, passage in zip(
                unique_labels, passage.split_highlightings(passage_labels), strict=True
            ):
                clusters[label].append(passage)

        return [
            Corpus(passages=passages, model=self._model, umap=self._umap, label=label)
            for label, passages in clusters.items()
        ]

    def hover_datas(self, metadata_keys=None) -> list[dict[str, Any]]:
        return [
            hover_data | {"label": self._label}
            for passage in self.passages
            for hover_data in passage.hover_datas(metadata_keys=metadata_keys)
        ]

    def document_frequencies(self, **kwargs) -> Counter[str]:
        """Returns the document frequency of each word in the corpus."""
        return Counter(
            word
            for passage in self.passages
            for word in passage.term_frequencies(**kwargs).keys()
        )

    def interactive_plot(self, **kwargs):
        return umap.plot.interactive(
            self.umap, hover_data=pd.DataFrame(self.hover_datas()), **kwargs
        )

    @property
    def umap(self):
        if self._umap is None:
            self._compute_umap()
        return self._umap

    def _compute_umap(self) -> list[ArrayLike]:
        assert self._umap is None, "UMAP has already been computed."

        logging.info("Computing UMAP...")
        umap = UMAP(metric="cosine")
        embeddings = umap.fit_transform(np.array(self._token_embeddings()))

        # TODO: this becomes invalid when highlightings are changes
        for embedding, highlighting in zip(embeddings, self.highlightings):
            highlighting.umap_embedding = embedding

        self._umap = umap
        return embeddings

    def umap_embeddings(self) -> list[ArrayLike]:
        embeddings = [
            highlighting.umap_embedding for highlighting in self.highlightings
        ]
        if any(embedding is None for embedding in embeddings):
            logging.warning("UMAP embeddings have not been computed.")
            embeddings = self._compute_umap()

        assert len(embeddings) == len(self.highlightings)
        return embeddings

    def highlighted_texts(self, metadata_fields: Iterable[str] = None) -> list[str]:
        """Returns an iterable over all highlightings."""
        texts = [
            text
            for passage in self.passages
            for text in passage.highlighted_texts(metadata_fields)
        ]
        # TODO: remove assertion
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
