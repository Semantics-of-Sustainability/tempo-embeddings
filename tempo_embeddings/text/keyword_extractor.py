import logging
import string
from typing import Iterable

import numpy as np
from scipy.sparse import csr_matrix
from sklearn.base import check_is_fitted
from sklearn.exceptions import NotFittedError
from sklearn.feature_extraction.text import TfidfVectorizer

from ..settings import OUTLIERS_LABEL
from .corpus import Corpus


class KeywordExtractor:
    """A class to extract keywords from a corpus using (c)TF-IDF.

    Call the fit() method to fit the vectorizer on the corpus.
    The object is immutable, and linked to the corpus it was created with.
    """

    def __init__(self, corpus: Corpus) -> None:
        self._corpus = corpus
        self._vectorizer = TfidfVectorizer(
            tokenizer=lambda passage: [word.casefold() for word in passage.words()],
            preprocessor=lambda x: x,
        )

        self._feature_names = None
        """Store the feature names once the vectorizer has been fitted."""

    def _is_fitted(self) -> bool:
        """Check if the vectorizer has been fitted.

        Returns:
            True if the vectorizer has been fitted, False otherwise.
        """
        try:
            check_is_fitted(self._vectorizer)
            return True
        except NotFittedError:
            return False

    def fit(self) -> "KeywordExtractor":
        """Fit the vectorizer on the corpus.

        Returns:
            The fitted KeywordExtractor instance.
        Raises:
            RuntimeError: If the vectorizer has already been fitted.
        """

        if self._is_fitted():
            raise RuntimeError("Vectorizer has already been fitted.")

        logging.info(f"Fitting vectorizer on {self._corpus}.")
        self._vectorizer.fit(self._corpus.passages)
        self._feature_names = self._vectorizer.get_feature_names_out()

        return self

    def _tf_idf_words(
        self, corpus: Corpus, *, use_2d_embeddings: bool
    ) -> Iterable[str]:
        """The top n words in the corpus with maximum score by <tf-idf> x <distance to centroid>.

        Each word is scored by multiplying its tf-idf score for each passage
        with the passage's embedding distance to the centroid.

        The term frequency (TF) is computed on the input corpus.
        The (inverse) document frequency (IDF) depends on the corpus on which the vectorizer was fitted.

        Args:
            corpus (Corpus): The corpus to extract the top words from
            use_2d_embeddings (bool): Whether to use 2D embeddings for distance calculation.

        Yields:
            (str) the words in the corpus, descending by score.

        Raises:
            RuntimeError: If the vectorizer has not been fitted.
        """

        try:
            self.fit()
            logging.debug("Vectorizer fitted.")
        except RuntimeError as e:
            logging.debug(str(e))

        tf_idfs: csr_matrix = self._vectorizer.transform(corpus.passages)

        assert tf_idfs.shape == (
            len(corpus.passages),
            len(self._feature_names),
        ), f"tf_idfs shape ({tf_idfs.shape}) does not match expected shape."

        ### Weigh in vector distances
        distances: np.ndarray = corpus.distances(
            normalize=True, use_2d_embeddings=use_2d_embeddings
        )
        weights = np.ones(distances.shape[0]) - distances

        assert (
            weights.shape[0] == tf_idfs.shape[0]
        ), f"distances shape ({weights.shape}) does not match expected shape."

        weighted_tf_idfs = np.average(tf_idfs.toarray(), weights=weights, axis=0)
        assert weighted_tf_idfs.shape[0] == len(self._feature_names)

        # TODO: no need to sort the whole array, just get the indices of the top n
        top_indices = np.argsort(-weighted_tf_idfs)
        for i in top_indices:
            yield self._feature_names[i]

    def top_words(
        self,
        corpus: Corpus,
        *,
        exclude_words: Iterable[str] = None,
        min_word_length: int = 3,
        use_2d_embeddings: bool = False,
    ) -> Iterable[str]:
        """
        Extract the top words for the given corpus, relative to the extractor's corpus data.

        If necessary, the vectorizer is fitted first.

        Short words, and words containing punctuation are filtered out, as well as words that are in the exclude_words set.

        Args:
            exclude_words: The word to exclude from the label (case-insensitive),
                e.g. stopwords and the search term used for composing this corpus. Defaults to None
            min_word_length: the minimum length of a word to be included in the label. Defaults to 3
            use_2d_embeddings: Whether to use 2D embeddings for distance calculation. Defaults to False

        Yield:
            the words in the corpus, descending by score.
        """

        exclude_words: set[str] = {word.casefold() for word in (exclude_words or [])}

        if corpus.label in (-1, OUTLIERS_LABEL):
            words = [OUTLIERS_LABEL]
        else:
            words = self._tf_idf_words(corpus, use_2d_embeddings=use_2d_embeddings)

        for word in words:
            if (
                len(word.strip()) >= min_word_length
                and not any(char in string.punctuation for char in word)
                and word.casefold() not in exclude_words
            ):
                yield word
