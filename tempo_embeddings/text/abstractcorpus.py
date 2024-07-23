import string
import sys
from abc import ABC
from abc import abstractmethod
from collections import defaultdict
from functools import lru_cache
from typing import Any
from typing import Iterable
from typing import Optional
import numpy as np
import pandas as pd
from numpy.typing import ArrayLike
from scipy.sparse import csr_matrix
from scipy.spatial.distance import cosine
from sklearn.cluster import HDBSCAN
from sklearn.feature_extraction.text import TfidfVectorizer
from ..settings import OUTLIERS_LABEL
from .passage import Passage


class AbstractCorpus(ABC):
    @property
    @abstractmethod
    def passages(self) -> list[Passage]:
        return NotImplemented

    @property
    def embeddings(self) -> ArrayLike:
        return np.array([p.embedding for p in self.passages])

    @embeddings.setter
    def embeddings(self, embeddings: ArrayLike):
        try:
            for row, passage in zip(embeddings, self._passages, strict=True):
                passage.embedding = row
        except TypeError as e:
            # TODO: remove this block once we drop support for Python < 3.10
            if sys.version_info.minor < 10:
                if len(embeddings) == len(self._passages):
                    for row, passage in zip(embeddings, self._passages):
                        passage.embedding = row
                else:
                    raise ValueError(
                        f"embeddings must have the same length as passages: {len(embeddings)} != {len(self._passages)}"
                    ) from e
            else:
                raise e

    @property
    @abstractmethod
    def vectorizer(self) -> TfidfVectorizer:
        return NotImplemented

    def embeddings_as_df(self) -> pd.DataFrame:
        return pd.DataFrame(
            {"x": [e[0] for e in self.embeddings], "y": [e[1] for e in self.embeddings]}
        )

    @property
    def label(self) -> Optional[str]:
        return self._label

    @label.setter
    def label(self, value: str):
        self._label = value

    def __eq__(self, other: object) -> bool:
        return other._passages == self._passages

    def __hash__(self) -> int:
        return hash(tuple(self.passages))

    def __len__(self) -> int:
        return len(self._passages)

    def __contains__(self, passage: Passage) -> bool:
        return passage in self._passages

    def centroid(self) -> ArrayLike:
        """The mean for all passage embeddings."""
        return np.array(self.embeddings).mean(axis=0)

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

    def to_dataframe(self):
        rows = []
        for p in self.passages:
            row = {"ID_DB": p.get_unique_id(), "text": p.text}
            for key in p.metadata.keys():
                row[key] = p.metadata[key]
            rows.append(row)
        return pd.DataFrame(rows)

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

    def distances(self, normalize: bool) -> ArrayLike:
        """Compute the distances between the passages and the centroid of the corpus.

        Args:
            normalize (bool): Normalize the distances vector.

        Returns: an array with the distances per passage in this corpus.
        """
        centroid = self.centroid()

        # TODO: can this be vectorized?
        distances: ArrayLike = np.array(
            [cosine(centroid, embedding) for embedding in self.embeddings]
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
        distances: ArrayLike = self.distances(normalize=True)
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
            ArrayLike: a sparse matrix of n_passages x n_words
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

    def cluster(self, **kwargs):
        cluster_labels: list[int] = (
            HDBSCAN(**kwargs).fit_predict(self.embeddings).astype(int).tolist()
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
