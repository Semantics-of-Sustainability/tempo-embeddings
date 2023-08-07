import csv
import gzip
import logging
from collections import defaultdict
from pathlib import Path
from typing import Any
from typing import Iterable
from typing import Optional
from typing import TextIO
import joblib
import numpy as np
from numpy.typing import ArrayLike
from scipy.sparse import csr_matrix
from sklearn.cluster import HDBSCAN
from sklearn.feature_extraction.text import TfidfVectorizer
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

    @property
    def label(self) -> Optional[str]:
        return self._label

    @label.setter
    def label(self, value: str):
        self._label = value

    def texts(self):
        return [passage.text for passage in self._passages]

    def passages_untokenized(self) -> list[Passage]:
        return [passage for passage in self._passages if passage.tokenization is None]

    def passages_unembeddened(self) -> list[Passage]:
        return [passage for passage in self._passages if passage.embedding is None]

    @property
    def embeddings_model(self) -> Optional[TransformerModelWrapper]:
        return self._model

    @embeddings_model.setter
    def embeddings_model(self, value: TransformerModelWrapper):
        self._model = value

    @property
    def highlightings(self) -> list[Highlighting]:
        return [passage.highlighting for passage in self.passages]

    def has_metadata(self, key: str, strict=False) -> bool:
        """Returns True if the corpus has a metadata key.

        Args:
            key: The metadata key to check for.
            strict: If True, returns True only if all passages have the key.
        """
        condition = all if strict else any
        return condition(passage.has_metadata(key) for passage in self.passages)

    def get_metadatas(self, key: str) -> list[Any]:
        """Returns all metadata values for a key."""

        return [passage.metadata.get(key) for passage in self.passages]

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

        return [passage.token_embedding() for passage in self.passages]

    def has_embeddings(self, validate=False) -> bool:
        """Returns True embeddings have been computed for the corpus.

        Args:
            validate: If True, validates that all Passage objects have an embedding.

        Returns:
            True if embeddings have been computed
        """
        func = all if validate else any
        return func(passage.embedding is not None for passage in self.passages)

    def has_tokenizations(self, validate: bool = False) -> bool:
        func = all if validate else any
        return func(passage.tokenization is not None for passage in self.passages)

    def subcorpus(self, token: str, exact_match: bool = True, **metadata) -> "Corpus":
        """Generate a new Corpus object with matching passages and highlightings.

        Args:
            token: The token to search for.
            metadata: Metadata fields to match against
        """
        passages = []
        for passage in self.passages:
            passages.extend(
                [
                    passage
                    for passage in passage.highlight(token, exact_match=exact_match)
                    if all(
                        passage.metadata.get(key) == value
                        for key, value in metadata.items()
                    )
                ]
            )

        # Dropping unmatched passages
        return Corpus(passages, self._model, self._umap, label=token)

    def nearest_neighbours(self, n: int = 5) -> Iterable[tuple[Passage, float]]:
        centroid = self.umap_mean()

        distances: ArrayLike = np.array(
            [passage.distance(centroid) for passage in self.passages]
        )
        if n > len(distances) - 1:
            logging.warning(
                "n (%d) is greater than the number of passages (%d).", n, len(distances)
            )
            n = len(distances) - 1

        nearest_highlighting_indices: ArrayLike = np.argpartition(-distances, n)
        for i in nearest_highlighting_indices[:n]:
            yield self.passages[i], distances[i]

    def topic_words(self, vectorizer: TfidfVectorizer, n: int = 5) -> list[str]:
        """The most important words in the corpus according to a vectorizer."""
        tf_idfs: csr_matrix = self.tf_idf(vectorizer)

        assert all(
            passage.highlighting for passage in self.passages
        ), "Passages must have highlightings"

        assert tf_idfs.shape == (
            len(self.passages),
            len(vectorizer.get_feature_names_out()),
        ), f"tf_idfs shape ({tf_idfs.shape}) does not match expected shape."

        ### Weigh in vector distances
        centroid = self.umap_mean()
        distances: ArrayLike = np.array(
            [passage.distance(centroid) for passage in self.passages]
        )
        weights = np.ones(distances.shape[0]) - (distances / np.linalg.norm(distances))

        assert weights.argmin() == distances.argmax()
        assert weights.argmax() == distances.argmin()
        assert (
            weights.shape[0] == tf_idfs.shape[0]
        ), f"distances shape ({weights.shape}) does not match expected shape."

        weighted_tf_idfs = np.average(tf_idfs.toarray(), weights=weights, axis=0)
        assert weighted_tf_idfs.shape[0] == len(vectorizer.get_feature_names_out())

        # pylint: disable=invalid-unary-operand-type
        top_indices = np.argpartition(-weighted_tf_idfs, n)

        return [vectorizer.get_feature_names_out()[i] for i in top_indices[:n]]

    def umap_mean(self) -> ArrayLike:
        """The mean for all passage embeddings."""
        return np.array(self.umap_embeddings()).mean(axis=0)

    def clusters(self, **kwargs) -> Iterable["Corpus"]:
        """Clusters the corpus using HDBSCAN and returns a list of subcorpora."""
        labels: list[int] = (
            HDBSCAN(**kwargs).fit_predict(self.umap_embeddings()).astype(int).tolist()
        )

        clusters: dict[int, list[Passage]] = defaultdict(list)

        for label, passage in zip(labels, self.passages):
            if label == -1:
                label = "outliers"

            clusters[label].append(passage)
            clusters[label].extend(passage.split_highlightings([label]))

        return [
            Corpus(passages=passages, model=self._model, umap=self._umap, label=label)
            for label, passages in clusters.items()
        ]

    def hover_datas(self, metadata_keys=None) -> list[dict[str, Any]]:
        return [
            hover_data | {"label": self._label}
            for passage in self.passages
            for hover_data in passage.hover_data(metadata_keys=metadata_keys)
        ]

    def tfidf_vectorizer(self, **kwargs) -> TfidfVectorizer:
        def tokenizer(passage: Passage) -> list[str]:
            return [word.casefold() for word in passage.words()]

        vectorizer = TfidfVectorizer(
            tokenizer=tokenizer, preprocessor=lambda x: x, **kwargs
        )
        vectorizer.fit(self.passages)
        return vectorizer

    def tf_idf(self, vectorizer: TfidfVectorizer) -> csr_matrix:
        """Returns a sparse matrix for the TF-IDF of all passages in the corpus.

        Args:
            vectorizer (TfidfVectorizer): a vectorizer generated by count_vectorizer()

        Returns:
            ArrayLike: a sparse matrix of n_passages x n_words
        """
        return vectorizer.transform(self.passages)

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
        for embedding, passage in zip(embeddings, self.passages):
            passage.highlighting.umap_embedding = embedding

        self._umap = umap
        return embeddings

    def umap_embeddings(self) -> list[ArrayLike]:
        embeddings = [passage.highlighting.umap_embedding for passage in self.passages]

        if any(embedding is None for embedding in embeddings):
            logging.warning("UMAP embeddings have not been computed.")
            # FIXME: this will raise an error for newly added passages
            embeddings = self._compute_umap()

        assert len(embeddings) == len(self.highlightings)
        return embeddings

    def highlighted_texts(self, metadata_fields: Iterable[str] = None) -> list[str]:
        """Returns an iterable over all highlightings."""
        texts = [passage.highlighted_text(metadata_fields) for passage in self.passages]

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
        window_size: Optional[int] = None,
    ):
        """Read input data from an open file handler, one sequence per line."""
        return Corpus(
            passages=[
                passage
                for line in f
                for passage in Passage.from_text(
                    line, model, metadata=metadata, window_size=window_size
                )
            ],
            model=model,
        )

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
