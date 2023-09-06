import csv
import gzip
import logging
from collections import Counter
from functools import lru_cache
from pathlib import Path
from typing import Any
from typing import Iterable
from typing import Optional
from typing import TextIO
import joblib
import numpy as np
from numpy.typing import ArrayLike
from scipy.sparse import csr_matrix
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
        if not model:
            logging.warning(
                "No model provided, will not be able to compute embeddings."
            )

        self._umap: Optional[UMAP] = umap
        self._label: Optional[str] = label

    def __add__(self, other: "Corpus") -> "Corpus":
        if self._model != other._model:
            raise ValueError("Cannot add two corpora with different embeddings models")

        passages = self._passages + other._passages

        return Corpus(passages, self._model)

    def __contains__(self, passage: Passage) -> bool:
        return passage in self._passages

    def __len__(self) -> int:
        return len(self._passages)

    def __repr__(self) -> str:
        return f"Corpus({self._label!r}, {self._passages[:10]!r})"

    def __eq__(self, other: object) -> bool:
        return self._model == other._model and other._passages == self._passages

    @property
    def passages(self) -> list[Passage]:
        return self._passages

    def passages_with_highlighting(self) -> list[Passage]:
        return [
            passage for passage in self._passages if passage.highlighting is not None
        ]

    @property
    def model(self) -> Optional[TransformerModelWrapper]:
        return self._model

    @property
    def umap(self):
        if self._umap is None:
            self._compute_umap()
        return self._umap

    @property
    def label(self) -> Optional[str]:
        return self._label

    @label.setter
    def label(self, value: str):
        self._label = value

    def texts(self):
        return [passage.text for passage in self._passages]

    @property
    def embeddings_model(self) -> Optional[TransformerModelWrapper]:
        return self._model

    @embeddings_model.setter
    def embeddings_model(self, value: TransformerModelWrapper):
        self._model = value

    @property
    def highlightings(self) -> list[Highlighting]:
        return [passage.highlighting for passage in self.passages]

    def metadata_fields(self) -> set[str]:
        """Returns all metadata fields in the corpus."""

        return {key for passage in self.passages for key in passage.metadata}

    def has_metadata(self, key: str, strict=False) -> bool:
        """Returns True if the corpus has a metadata key.

        Args:
            key: The metadata key to check for.
            strict: If True, returns True only if all passages have the key.
        """
        condition = all if strict else any
        return condition(key in passage.metadata for passage in self.passages)

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
        """Computes embeddings and tokenizations for all passages in the corpus.

        If the passages in the corpus have highlightings,
        embeddings for the highlighted tokens are also computed.
        """

        if self._model is None:
            logging.error("Corpus does not have a model.")
        else:
            self._model.compute_embeddings(self)

            if any(passage.has_umap_embedding() for passage in self.passages):
                logging.warning("Removing existing UMAP embeddings.")
                for passage in self.passages:
                    passage.highlighting.umap_embedding = None

    def _token_embeddings(self) -> list[ArrayLike]:
        """Returns a list of token embeddings for all passages in the corpus.

        Converts the tensors into Numpy arrays and copies to the CPU.

        Returns:
            a list of Numpy arrays for all passages in the corpus.
        """
        if not self.highlightings:
            logging.warning("Corpus has no highlightings")
            return []

        if not self.has_embeddings():
            # batch-compute embeddings for all passages in corpus.
            self.compute_embeddings()

        return [
            passage.token_embedding().cpu().numpy()
            for passage in self.passages_with_highlighting()
        ]

    def has_embeddings(self, validate=False) -> bool:
        """Returns True embeddings have been computed for the corpus.

        Args:
            validate: If True, validates that all Passage objects have an embedding.

        Returns:
            True if embeddings have been computed
        """
        func = all if validate else any
        return func(passage.has_embedding() is not None for passage in self.passages)

    def has_tokenizations(self, validate: bool = False) -> bool:
        func = all if validate else any
        return func(passage.tokenization is not None for passage in self.passages)

    def subcorpus(self, token: str, exact_match: bool = True, **metadata) -> "Corpus":
        """Generate a new Corpus object with matching passages and highlightings.

        Args:
            token: The token to search for.
            exact_match: if True (default), match substring matches in a token too.
            metadata: Metadata fields to match against

        Returns:
            a new Corpus object with matching passages and highlightings.
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

        if self._label is not None:
            logging.warning(
                "Parent corpus had label '%s', new subcorpus will have '%s'.",
                self.label,
                token,
            )
        return Corpus(passages, self._model, self._umap, label=token)

    def distances(self, normalize: bool) -> ArrayLike:
        """Compute the distances between the passages and the centroid of the corpus.

        Uses Euclidian distance and UMAP embeddings.

        Args:
            normalize (bool): Normalize the distances vector.

        Returns: an array with the distances per passage in this corpus.
        """
        centroid = self.umap_mean()

        distances: ArrayLike = np.array(
            [passage.distance(centroid) for passage in self.passages]
        )
        return (distances / np.linalg.norm(distances)) if normalize else distances

    def _nearest_neighbours(
        self, n: int = 5, normalize: bool = False
    ) -> Iterable[tuple[Passage, float]]:
        """Get the passages that are closest to the centroid of the corpus.

        Args:
            n (int): The number of neighbours to return. Defaults to 5.
            normalize (bool, optional): Normalize the distances. Defaults to False.

        Returns:
            Iterable[tuple[Passage, float]]: n tuples of Passage and distance
        """
        distances: ArrayLike[float] = self.distances(normalize)

        if n > len(distances):
            logging.warning(
                "n (%d) is greater than the number of passages (%d).", n, len(distances)
            )
            n = len(distances)

        top_indices: ArrayLike[int] = np.argsort(distances)[:n]

        for i in top_indices:
            yield self.passages[i], distances[i]

    def set_topic_label(
        self,
        vectorizer: Optional[TfidfVectorizer],
        *,
        exclude_word: str = "",
        n: int = 1,
        exact_match: bool,
        stopwords: set[str] = None,
    ) -> None:
        """Set the label of the corpus to the top word(s) in the corpus.

        Args:
            vectorizer: The vectorizer to use for tf-idf scoring;
                see Corpus.tfidf_vectorizer() and Corpus.topic_words().
                If None, uses most words with largest document frequency.
            exclude_word: The word to exclude from the label,
                e.g. the search term used for composing this corpus
            n: concatenate the top n words in the corpus as the label.
            exact_match: if False, exclude all words that contain the `exclude_word`.
            stopwords: if given, exclude these words
        """
        # TODO: merge exclude_word and stopwords?

        if stopwords is None:
            stopwords = set()

        def filter_word(word: str) -> bool:
            return (
                word != exclude_word and word.casefold() not in stopwords
                if exact_match
                else exclude_word.casefold() not in word.casefold()
            )

        _n = n * 1000

        if vectorizer:
            words = self._tf_idf_words(vectorizer, n=_n)
        else:
            words = self._document_frequencies(_n)

        top_words: list[str] = [word for word in words if filter_word(word)]

        self._label = "; ".join(top_words[:n])

        return self._label

    def _tf_idf_words(self, vectorizer: TfidfVectorizer, n: int = None) -> list[str]:
        """The most significant words in the corpus by <tf-idf> x <distance to centroid>.

        Each word is scored by multiplying its tf-idf score for each passage
        with the passage's distance to the centroid, using UMAP embeddings.

        Args:
            vectorizer: The vectorizer to use for tf-idf scoring;
                typically generated by calling Corpus.tfidf_vectorizer()
                on a super-corpus (ie. a superset of this collection).
            n: The number of words to return. Defaults to None, returning all words

        Returns:
            the n highest scoring words in a sorted list
        """
        tf_idfs: csr_matrix = self.tf_idf(vectorizer)

        assert all(
            passage.highlighting for passage in self.passages
        ), "Passages must have highlightings"

        assert tf_idfs.shape == (
            len(self.passages),
            len(Corpus.get_vocabulary(vectorizer)),
        ), f"tf_idfs shape ({tf_idfs.shape}) does not match expected shape."

        ### Weigh in vector distances
        distances: ArrayLike = self.distances(normalize=True)
        weights = np.ones(distances.shape[0]) - distances

        assert weights.argmin() == distances.argmax()
        assert weights.argmax() == distances.argmin()
        assert (
            weights.shape[0] == tf_idfs.shape[0]
        ), f"distances shape ({weights.shape}) does not match expected shape."

        weighted_tf_idfs = np.average(tf_idfs.toarray(), weights=weights, axis=0)
        assert weighted_tf_idfs.shape[0] == len(Corpus.get_vocabulary(vectorizer))

        # pylint: disable=invalid-unary-operand-type
        top_indices = np.argsort(-weighted_tf_idfs)[:n]

        return [Corpus.get_vocabulary(vectorizer)[i] for i in top_indices]

    def umap_mean(self) -> ArrayLike:
        """The mean for all passage embeddings."""
        return np.array(self.umap_embeddings()).mean(axis=0)

    def hover_datas(self, metadata_fields=None) -> list[dict[str, str]]:
        """A dictionary for each passage in this corpus to be used for visualization.

        Args:
            metadata_fields: The metadata fields to include in the hover data.
                If None (default), all metadata fields are included.

        Returns:
            A list of dictionaries, one for each passage in this corpus.
        """
        return [
            passage.hover_data(metadata_fields=metadata_fields)
            | {"corpus": str(self.label)}
            for passage in self.passages
        ]

    def tfidf_vectorizer(self, **kwargs) -> TfidfVectorizer:
        """Returns a vectorizer for the TF-IDF of all passages in the corpus.

        The resulting vectorizer is used for computing topic words in (sub-)corpora.

        kwargs are passed to TfidfVectorizer.

        Returns:
            TfidfVectorizer: a vectorizer for the TF-IDF of all passages in the corpus.
        """

        def tokenizer(passage: Passage) -> list[str]:
            return [word.casefold() for word in passage.words(use_tokenizer=False)]

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

    def _document_frequencies(self, n: int) -> list[str]:
        counter = Counter()
        for passage in self.passages:
            counter.update({word.casefold().strip(".,;") for word in passage.words()})
        return [word for word, freq in counter.most_common(n)]

    def _compute_umap(self) -> list[ArrayLike]:
        assert self._umap is None, "UMAP has already been computed."

        logging.info("Computing UMAP...")
        umap = UMAP(metric="cosine")
        umap_embeddings = umap.fit_transform(np.array(self._token_embeddings()))
        passages = self.passages_with_highlighting()

        # TODO: this becomes invalid when highlightings are changes
        for embedding, passage in zip(umap_embeddings, passages):
            passage.highlighting.umap_embedding = embedding

        self._umap = umap
        return umap_embeddings

    def umap_embeddings(self) -> list[ArrayLike]:
        """Returns a list of UMAP embeddings for all passages in the corpus."""

        embeddings = [
            passage.highlighting.umap_embedding
            for passage in self.passages_with_highlighting()
        ]

        if any(embedding is None for embedding in embeddings):
            logging.warning("UMAP embeddings have not been computed. Computing now...")
            # FIXME: this will raise an error for newly added passages
            embeddings = self._compute_umap()

        assert len(embeddings) == len(self.highlightings)
        return embeddings

    def highlighted_texts(self, metadata_fields: list[str] = None) -> list[str]:
        """Generates texts with formattting for all passages in the corpus.

        The resulting text is formatted with HTML tags for highlighting.

        Args:
            metadata_fields: The metadata fields to include in the highlighted text.

        Returns:
            A list of texts, one for each passage in this corpus.
        """

        texts = [
            passage.highlighted_text(metadata_fields=metadata_fields)
            for passage in self.passages
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
        *,
        filter_terms: list[str] = None,
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
                    line, metadata=metadata, window_size=window_size
                )
                if passage.contains_any(filter_terms)
            ],
            model=model,
        )

    @classmethod
    def from_lines_file(
        cls,
        filepath: Path,
        *,
        filter_terms: list[str] = None,
        encoding=DEFAULT_ENCODING,
        model: Optional[TransformerModelWrapper] = None,
    ):
        """Read input data from a file, one sequence per line."""

        with open(filepath, "rt", encoding=encoding) as f:
            return Corpus.from_lines(f, filter_terms=filter_terms, model=model)

    @classmethod
    def from_csv_file(
        cls,
        filepath: Path,
        text_columns: list[str],
        model: Optional[TransformerModelWrapper] = None,
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
                f, text_columns, model, filter_terms=filter_terms, **kwargs
            )

    @classmethod
    def from_csv_stream(
        cls,
        file_handler,
        text_columns: list[str],
        model: Optional[TransformerModelWrapper] = None,
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
                    else:
                        logging.error("No filter terms defined, hence no highlighting.")

        return Corpus(passages, model=model)

    @staticmethod
    @lru_cache
    def get_vocabulary(vectorizer: TfidfVectorizer) -> list[str]:
        """Caching wrapper for getting the vocabulary of a vectorizer.

        Args:
            vectorizer (TfidfVectorizer): a vectorizer generated by count_vectorizer()

        Returns:
            list[str]: the vocabulary of the vectorizer.
        """

        return vectorizer.get_feature_names_out()
