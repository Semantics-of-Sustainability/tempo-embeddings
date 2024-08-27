import logging
import random
import string
from abc import ABC, abstractmethod
from collections import defaultdict
from functools import lru_cache
from typing import Any, Iterable, Optional

import numpy as np
import pandas as pd
from scipy.sparse import csr_matrix
from scipy.spatial.distance import cosine
from sklearn.cluster import HDBSCAN
from sklearn.feature_extraction.text import TfidfVectorizer
from umap.umap_ import UMAP

from ..settings import OUTLIERS_LABEL, STRICT
from .passage import Passage


class AbstractCorpus(ABC):
    def __init__(self) -> None:
        super().__init__()

        self._embeddings_2d = None
        """Stores the 2d embeddings of the corpus."""

        self._umap: UMAP = None
        """The UMAP model that can be re-used to compress embeddings that are added later."""

    @property
    @abstractmethod
    def passages(self) -> list[Passage]:
        return NotImplemented

    @property
    def embeddings(self) -> np.ndarray:
        return np.array([p.embedding for p in self.passages])

    @embeddings.setter
    def embeddings(self, embeddings: np.ndarray):
        for row, passage in zip(embeddings, self._passages, **STRICT):
            passage.embedding = row

    def has_embeddings(self) -> bool:
        """Check if the corpus has embeddings.

        For performance, this assumes consistency across all passages in the corpus, hence stops if any passage has an embedding.

        Returns:
            bool: True if the corpus has embeddings, False otherwise, including empty corpora.
        """

        return not any(passage.embedding is None for passage in self.passages) and any(
            any(passage.embedding) for passage in self.passages
        )

    def _select_embeddings(self, use_2d_embeddings: bool, recompute: bool = False):
        if use_2d_embeddings:
            self.compress_embeddings(recompute=recompute)
            embeddings = self._embeddings_2d
        else:
            embeddings = self.embeddings
        return embeddings

    @property
    @abstractmethod
    def vectorizer(self) -> TfidfVectorizer:
        return NotImplemented

    @property
    def label(self) -> Optional[str]:
        return self._label

    @label.setter
    def label(self, value: str):
        self._label = value

    def __eq__(self, other: object) -> bool:
        return other.passages == self.passages

    def __hash__(self) -> int:
        return hash(tuple(self.passages))

    def __len__(self) -> int:
        return len(self._passages)

    def __contains__(self, passage: Passage) -> bool:
        return passage in self._passages

    def centroid(self, use_2d_embeddings: bool = True) -> np.ndarray:
        """The mean for all passage embeddings."""
        return np.array(self._select_embeddings(use_2d_embeddings)).mean(axis=0)

    @property
    def embeddings_2d(self) -> Optional[np.ndarray]:
        return self._embeddings_2d

    @embeddings_2d.setter
    def embeddings_2d(self, value: np.ndarray):
        self._embeddings_2d = value

    @property
    def umap(self) -> UMAP:
        return self._umap

    def _compute_umap(self, **umap_args):
        if self.umap:
            logging.warning("UMAP model already exists. Overwriting.")
        self._umap = UMAP(**umap_args).fit(self.embeddings)

    def compress_embeddings(self, *, recompute: bool = False, **umap_args):
        """Compress the embeddings of the corpus using UMAP and stores them in the corpus

        Args:
            recompute: If True, recomputes the UMAP model even if it has already been computed.
            umap_args: Additional arguments to pass to UMAP.

        Returns:
            np.ndarray: the compressed embeddings.

        Raises:
            ValueError: If the corpus has zero or exactly one embeddings.
        """
        if self._umap is None or recompute:
            self._compute_umap(**umap_args)
            self.embeddings_2d = self._umap.transform(self.embeddings)

        assert self.embeddings_2d is not None, "UMAP embeddings have not been computed."

        return self.embeddings_2d

    def texts(self) -> Iterable[str]:
        return (passage.text for passage in self._passages)

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

    def to_dataframe(
        self, sample_size=None, centroid_based_sample=False
    ) -> pd.DataFrame:
        """Transforms the Key Cluster information in a pandas Dataframe.

        Args:
            sample_size (int, optional): If provided it only returns the 'sample_size' otherwise all points are returned.
              Defaults to None.
            centroid_based_sample (bool): If True, the sample comprises the closest points to the centroid
                If False the sample is taken randomly.

        Returns:
            a Pandas Dataframe

        Raises:
            ValueError: If centroid_based_sample is True but sample_size is not provided.
        """
        if centroid_based_sample and not sample_size:
            raise ValueError(
                "centroid_based_sample cannot be True if sample_size is not provided."
            )

        if centroid_based_sample:
            distances = self.distances(normalize=False)
            sample_indices = np.argsort(distances)[:sample_size]
        elif sample_size:
            sample_indices = random.sample(range(len(self.passages)), sample_size)
        else:
            sample_indices = range(len(self.passages))

        rows = []
        for i in sample_indices:
            passage = self.passages[i]
            row = passage.metadata | {
                "ID_DB": passage.get_unique_id(),
                "text": passage.text,
            }
            if centroid_based_sample:
                row["distance_to_centroid"] = distances[i]

            if self._embeddings_2d is not None:
                row["x"] = self._embeddings_2d[i, 0]
                row["y"] = self._embeddings_2d[i, 1]

            rows.append(row)

        return pd.DataFrame(rows)

    def coordinates(self) -> pd.DataFrame:
        """Returns the coordinates of the corpus.

        Returns:
            pd.DataFrame: The coordinates of the corpus.
        """

        if self._embeddings_2d is None:
            logging.warning("No 2D embeddings available.")
            df = pd.DataFrame()
        else:
            df = pd.DataFrame(
                {
                    "x": [e[0] for e in self._embeddings_2d],
                    "y": [e[1] for e in self._embeddings_2d],
                }
            )
        return df

    def hover_datas(
        self, metadata_fields: Optional[list[str]] = None
    ) -> list[dict[str, str]]:
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

    def distances(self, normalize: bool, use_2d_embeddings: bool = True) -> np.ndarray:
        """Compute the distances between the passages and the centroid of the corpus.

        Args:
            normalize (bool): Normalize the distances vector.
            use_2d_embeddings (bool): If True, use the 2d embeddings for computing the distances.

        Returns: an array with the distances per passage in this corpus.
        """
        centroid = self.centroid(use_2d_embeddings=use_2d_embeddings)

        # TODO: can this be vectorized?

        embeddings = self._select_embeddings(use_2d_embeddings)
        distances: np.ndarray = np.array(
            [cosine(centroid, embedding) for embedding in embeddings]
        )

        if normalize:
            distances = distances / np.linalg.norm(distances)

        return distances

    @lru_cache(maxsize=8)
    def top_words(
        self,
        *,
        exclude_words: list[str] = None,
        min_word_length: int = 3,
        n: int = 5,
    ):
        """
        Extract the top words from the corpus.

        Args:
            exclude_words: The word to exclude from the label,
                e.g. stopwords and the search term used for composing this corpus
            stopwords: if given, exclude these words
            n:  the number of words to return. Defaults to 5
        """
        if exclude_words is None:
            exclude_words = set()
        exclude_words = {word.casefold() for word in exclude_words}

        # account for word filtering:
        _n = n + len(exclude_words)

        if self.label in (-1, OUTLIERS_LABEL):
            words = [OUTLIERS_LABEL]
        else:
            words = self._tf_idf_words(n=_n)

        # filter words
        return [
            word
            for word in words
            if len(word.strip()) >= min_word_length
            and not any(char in string.punctuation for char in word)
            and word.casefold() not in exclude_words
        ][:n]

    def set_topic_label(self, **kwargs) -> None:
        """Set the label of the corpus to the top word(s) in the corpus.

        Kwargs:
            exclude_words: The word to exclude from the label,
                e.g. stopwords and the search term used for composing this corpus
            n: concatenate the top n words in the corpus as the label.
            stopwords: if given, exclude these words
        """

        self._label = "; ".join(sorted(self.top_words(**kwargs)))

        return self._label

    def _tf_idf_words(self, *, n: int = None) -> list[str]:
        """The most words in the corpus with maximum <tf-idf> x <distance to centroid>.

        Each word is scored by multiplying its tf-idf score for each passage
        with the passage's embedding distance to the centroid.

        Args:
            n (int): The number of words to return. Defaults to None, returning all words

        Returns:
            the n highest scoring words in a sorted list
        """
        tf_idfs: csr_matrix = self.tf_idf()

        # assert all(
        #     passage.highlighting for passage in self.passages
        # ), "Passages must have highlightings"

        assert tf_idfs.shape == (
            len(self.passages),
            len(get_vocabulary(self.vectorizer)),
        ), f"tf_idfs shape ({tf_idfs.shape}) does not match expected shape."

        ### Weigh in vector distances
        distances: np.ndarray = self.distances(normalize=True)
        weights = np.ones(distances.shape[0]) - distances

        assert (
            weights.shape[0] == tf_idfs.shape[0]
        ), f"distances shape ({weights.shape}) does not match expected shape."

        weighted_tf_idfs = np.average(tf_idfs.toarray(), weights=weights, axis=0)
        assert weighted_tf_idfs.shape[0] == len(get_vocabulary(self.vectorizer))

        # pylint: disable=invalid-unary-operand-type
        top_indices = np.argsort(-weighted_tf_idfs)[:n]

        return [get_vocabulary(self.vectorizer)[i] for i in top_indices]

    def tf_idf(self) -> csr_matrix:
        """Returns a sparse matrix for the TF-IDF of all passages in the corpus.

        Returns:
            np.ndarray: a sparse matrix of n_passages x n_words
        """
        return self.vectorizer.transform(self.passages)

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

    def cluster(self, use_2d_embeddings: bool = True, **kwargs):
        embeddings = self._select_embeddings(use_2d_embeddings)
        cluster_labels: list[int] = (
            HDBSCAN(**kwargs).fit_predict(embeddings).astype(int).tolist()
        )
        clusters: dict[int, int] = defaultdict(list)
        for i, label in enumerate(cluster_labels):
            clusters[label].append(i)

        # pylint: disable=import-outside-toplevel,cyclic-import
        from .subcorpus import Subcorpus

        return [Subcorpus(self, indices, label) for label, indices in clusters.items()]

    @staticmethod
    def tfidf_vectorizer(passages: list[Passage], **kwargs) -> TfidfVectorizer:
        """Returns a vectorizer for the TF-IDF of all passages in the corpus.

        The resulting vectorizer is used for computing topic words in (sub-)corpora.

        kwargs are passed to TfidfVectorizer.

        Returns:
            TfidfVectorizer: a vectorizer for the TF-IDF of all passages in the corpus.
        """

        def tokenizer(passage: Passage) -> list[str]:
            return [word.casefold() for word in passage.words()]

        vectorizer = TfidfVectorizer(
            tokenizer=tokenizer, preprocessor=lambda x: x, **kwargs
        )
        vectorizer.fit(passages)
        return vectorizer


@lru_cache
def get_vocabulary(vectorizer: TfidfVectorizer) -> list[str]:
    """Caching wrapper for getting the vocabulary of a vectorizer.

    Args:
        vectorizer (TfidfVectorizer): a vectorizer generated by count_vectorizer()

    Returns:
        list[str]: the vocabulary of the vectorizer.
    """

    return vectorizer.get_feature_names_out()
