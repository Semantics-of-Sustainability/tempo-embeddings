import csv
import gzip
import logging
import random
import string
from collections import Counter, defaultdict
from functools import lru_cache
from itertools import groupby
from pathlib import Path
from typing import Any, Iterable, Optional

import joblib
import numpy as np
import pandas as pd
from scipy.sparse import csr_matrix
from scipy.spatial.distance import cosine
from sklearn.base import check_is_fitted
from sklearn.cluster import HDBSCAN
from sklearn.exceptions import NotFittedError
from sklearn.feature_extraction.text import TfidfVectorizer
from umap.umap_ import UMAP

from ..settings import DEFAULT_ENCODING, OUTLIERS_LABEL, STRICT
from .abstractcorpus import AbstractCorpus
from .passage import Passage
from .segmenter import Segmenter
from .subcorpus import Subcorpus


class Corpus(AbstractCorpus):
    """A Corpus implementation that holds the concrecte passages and embeddings."""

    def __init__(
        self,
        passages: Iterable[Passage] = None,
        label: Optional[Any] = None,
        *,
        umap_model: Optional["UMAP"] = None,
        vectorizer: Optional[TfidfVectorizer] = None,
    ):
        """Create a new corpus.

        Args:
            passages: The passages to add to the corpus. Defaults to None. Type is not enforced, but should be immutable.
            label: The label of the corpus. Defaults to None.
            umap_model: The UMAP model to use for compressing embeddings.
                Defaults to None, hence a new model will be initialized.
            vectorizer: The TfidfVectorizer model to use for vectorizing passages.
                Defaults to None, hence a new model will be initialized.
        """
        super().__init__()

        self._passages: tuple[Passage, ...] = passages or tuple()
        self._label: Optional[str] = label
        self._umap = umap_model or UMAP()
        self._vectorizer: TfidfVectorizer = vectorizer or TfidfVectorizer(
            tokenizer=lambda passage: [word.casefold() for word in passage.words()],
            preprocessor=lambda x: x,
        )

    def __add__(self, other: "Corpus") -> "Corpus":
        if self.umap is other.umap:
            umap = self.umap
        else:
            logging.warning(
                "Dropping UMAP model while merging corpora with different models."
            )
            umap = None

        if self.vectorizer is other.vectorizer:
            vectorizer = self.vectorizer
        else:
            logging.warning(
                "Dropping TfidfVectorizer model while merging corpora with different models."
            )
            vectorizer = None

        return Corpus(
            self._passages + other._passages,
            label=" + ".join((str(self.label), str(other.label))),
            umap_model=umap,
            vectorizer=vectorizer,
        )

    def __eq__(self, other: object) -> bool:
        return other.passages == self.passages

    def __hash__(self) -> int:
        return hash(tuple(self.passages))

    def __contains__(self, passage: Passage) -> bool:
        # TODO: a (frozen)set would be more efficient
        return passage in self._passages

    def __len__(self) -> int:
        """Return the number of passages in the corpus.

        Returns:
            int: The number of passages in the corpus.
        """
        return len(self.passages)

    def __repr__(self) -> str:
        return f"Corpus({self._label!r}, {len(self._passages)} passages)"

    def _select_embeddings(self, use_2d_embeddings: bool, recompute: bool = False):
        if use_2d_embeddings:
            self.compress_embeddings(recompute=recompute)
            embeddings = self.embeddings_2d
        else:
            embeddings = self.embeddings
        return embeddings

    @property
    def embeddings(self) -> np.ndarray:
        return np.array([p.embedding for p in self.passages])

    @embeddings.setter
    def embeddings(self, embeddings: np.ndarray):
        for row, passage in zip(embeddings, self.passages, **STRICT):
            passage.embedding = row

    @property
    def embeddings_2d(self) -> Optional[np.ndarray]:
        if any(p.embedding_compressed for p in self.passages):
            return np.array([p.embedding_compressed for p in self.passages])
        else:
            return None

    @embeddings_2d.setter
    def embeddings_2d(self, value: np.ndarray):
        for row, passage in zip(value, self.passages, **STRICT):
            # TODO: test
            passage.embedding_2d = row.tolist()

    @property
    def label(self) -> Optional[str]:
        return self._label

    @label.setter
    def label(self, value: str):
        self._label = value

    @property
    def passages(self) -> list[Passage]:
        return self._passages

    @property
    def vectorizer(self) -> TfidfVectorizer:
        if not self.__is_fitted(self._vectorizer):
            self.fit_vectorizer()

        return self._vectorizer

    @property
    def umap(self) -> UMAP:
        return self._umap

    @staticmethod
    def __is_fitted(model) -> bool:
        try:
            check_is_fitted(model)
            return True
        except NotFittedError:
            return False

    def centroid(self, use_2d_embeddings: bool = True) -> np.ndarray:
        """The mean for all passage embeddings."""
        embeddings = self._select_embeddings(use_2d_embeddings)
        return np.array(embeddings).mean(axis=0)

    def coordinates(self) -> pd.DataFrame:
        """Returns the coordinates of the corpus.

        Returns:
            pd.DataFrame: The coordinates of the corpus.
        """

        if self.embeddings_2d is None:
            logging.warning("No 2D embeddings available.")
            df = pd.DataFrame()
        else:
            df = pd.DataFrame(
                {
                    "x": [e[0] for e in self.embeddings_2d],
                    "y": [e[1] for e in self.embeddings_2d],
                }
            )
        return df

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

    def groupby(
        self, key, *, default_value: Any = None
    ) -> Iterable[tuple[Any, list[Passage]]]:
        """Group the passages in the corpus by a metadata key.

        See Also:
            `get_metadatas()` to get all the metadata values
        Args:
            key: The metadata key to group by.
            default_value: The default value to use if the key is missing. Defaults to None.
        Returns:
            Iterable[tuple[Any, list[Passage]]]: An iterable of tuples with the key and the passages that have that key.
        Raises:
            TypeError: if the sorting fails due to incomparable types, e.g. None and int.

        """
        # TODO: do we still need this method?

        def key_func(p):
            return p.metadata.get(key, default_value)

        return groupby(sorted(self._passages, key=key_func), key_func)

    ##### FITTING methods #####
    def fit_vectorizer(self):
        self._vectorizer.fit(self.passages)

    def _fit_umap(self, *, recompute: bool = False):
        if recompute or not self.__is_fitted(self._umap):
            self._umap.fit(self.embeddings)
        else:
            logging.warning("UMAP model already fitted.")

    ##### Clustering and Embedding methods #####
    def _clusters(self, embeddings, max_clusters: int, **hdbscan_args) -> list[int]:
        """Cluster the embeddings using HDBSCAN.

        Recursively adapts the HDBSCAN parameters if the number of clusters exceeds `max_clusters`.

        Args:
            embeddings: The embeddings to cluster.
            max_clusters: The maximum number of clusters to create.
            **hdbscan_args: Additional keyword arguments to pass to HDBSCAN.
        Returns:
            list[int]: The cluster labels per passage from HDBSCAN.
        """
        passage_clusters: list[int] = (
            HDBSCAN(**hdbscan_args).fit_predict(embeddings).astype(int).tolist()
        )
        """A list of cluster labels assigned to each passage by HDBSCAN."""

        clusters = Counter(passage_clusters)

        if max_clusters and len(clusters) > max_clusters:
            logging.warning(
                f"Clustering with {hdbscan_args} resulted in >{max_clusters} ({len(clusters)}) clusters."
            )

            # set minimum cluster size to half the size of the largest cluster (except outliers)
            _min_cluster_size: int = next(
                size // 2 for cluster, size in clusters.most_common() if cluster != -1
            )
            if (
                _min_cluster_size != hdbscan_args["min_cluster_size"]
                and _min_cluster_size > 1
            ):
                hdbscan_args["min_cluster_size"] = _min_cluster_size
                logging.info(
                    "Setting 'min_cluster_size' to %d", hdbscan_args["min_cluster_size"]
                )

                passage_clusters = self._clusters(
                    embeddings, max_clusters, **hdbscan_args
                )
            else:
                logging.warning("Could not reduce number of clusters.")

        return passage_clusters

    def cluster(
        self, max_clusters: Optional[int] = 50, use_2d_embeddings: bool = True, **kwargs
    ) -> Iterable[Subcorpus]:
        """Cluster the passages in the corpus.

        Args:
            max_clusters: The maximum number of clusters to create. If necessary, the epsilon HDBSCAN parameter is increased iteratively.
            use_2d_embeddings: Whether to use 2D embeddings instead of full embeddings for clustering. If necessary, they are computed. Defaults to True.
            **kwargs: Additional keyword arguments to pass to HDBSCAN.
        Yields:
            Corpus: A generator of Corpus objects, each representing a cluster. Labels are integers as assigned by HDBSCAN, with -1 indicating outliers.

        """
        embeddings = self._select_embeddings(use_2d_embeddings)

        hdbscan_args = {
            "min_cluster_size": 5,
            "cluster_selection_method": "leaf",
            "cluster_selection_epsilon": 0.0,
            "min_samples": kwargs.get("min_samples", 5),
        } | kwargs

        if (
            hdbscan_args.get("cluster_selection_method") == "leaf"
            and "max_cluster_size" in hdbscan_args
        ):
            logging.warning(
                f"'max_cluster_size' ({hdbscan_args['max_cluster_size']}) has no effect with cluster selection method '{hdbscan_args.get('cluster_selection_method')}'."
            )

        passage_clusters: list[int] = self._clusters(
            embeddings, max_clusters or 0, **hdbscan_args
        )

        cluster_passages: dict[int, int] = defaultdict(list)
        """A dictionary mapping cluster labels/indices to passage indices in the corpus."""
        for passage, cluster in zip(self.passages, passage_clusters, **STRICT):
            cluster_passages[cluster].append(passage)

        for cluster, passages in cluster_passages.items():
            yield Corpus(
                tuple(passages),
                cluster,
                umap_model=self._umap,
                vectorizer=self._vectorizer,
            )

    def compress_embeddings(self, *, recompute: bool = False):
        """Compress the embeddings of the corpus using UMAP and stores them in the corpus

        Args:
            recompute: If True, recomputes the UMAP model even if it has already been computed.

        Returns:
            np.ndarray: the compressed embeddings.

        Raises:
            ValueError: If the corpus has zero or exactly one embeddings.
        """
        self._fit_umap(recompute=recompute)
        self.embeddings_2d = self.umap.transform(self.embeddings)

        assert (
            self.embeddings_2d.shape[0] == len(self)
        ), f"{self.embeddings_2d.shape[0]} UMAP embeddings have been computed, but there are {len(self)} passages."

        return self.embeddings_2d

    ##### METADATA methods #####
    def get_metadatas(self, key: str, *, default_value: Any = None) -> Iterable[Any]:
        """Returns all metadata values for a key.

        Args:
            key: The metadata key to get.
            default_value: The default value to use if the key is missing. Defaults to None.
        Yields:
            Any: each passage's the metadata values for the key.
        """

        for passage in self.passages:
            yield passage.metadata.get(key, default_value)

    def set_metadatas(self, key, value):
        """Sets a metadata key to a value for all passages.

        Args:
            key: The metadata key to set.
            value: The value to set the metadata key to.
        """

        for passage in self.passages:
            passage.set_metadata(key, value)

    def has_metadata(self, key: str, strict=False) -> bool:
        """Returns True if the corpus has a metadata key.

        Args:
            key: The metadata key to check for.
            strict: If True, returns True only if all passages have the key.
        """
        condition = all if strict else any
        return condition(key in passage.metadata for passage in self.passages)

    def metadata_fields(self) -> set[str]:
        """Returns all metadata fields in the corpus."""

        return {key for passage in self.passages for key in passage.metadata}

    def texts(self) -> Iterable[str]:
        for passage in self._passages:
            yield passage.text

    ##### TF-IDF methods #####
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

    @lru_cache(maxsize=256)
    def top_words(
        self,
        *,
        exclude_words: Iterable[str] = None,
        min_word_length: int = 3,
        n: int = 5,
    ):
        """
        Extract the top words from the corpus.

        Args:
            exclude_words: The word to exclude from the label,
                e.g. stopwords and the search term used for composing this corpus
            min_word_length: the minimum length of a word to be included in the label. Defaults to 3
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

        # TODO: no need to filter all words if we only return n
        return [
            word
            for word in words
            # filter words:
            if len(word.strip()) >= min_word_length
            and not any(char in string.punctuation for char in word)
            and word.casefold() not in exclude_words
        ][:n]

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

    def highlighted_texts(self, metadata_fields: list[str] = None) -> list[str]:
        """Generates texts with formatting for all passages in the corpus.

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

    def batches(self, batch_size: int) -> Iterable[tuple[Passage, ...]]:
        """Split the passages into batches of a given size.

        Args:
            batch_size: The maximum number of passages per batch.
        Yields:
            Iterable[tuple[Passage, ...]]: A generator of tuples of passages.
        """
        if batch_size <= 1:
            yield self.passages
        else:
            for batch_start in range(0, len(self.passages), batch_size):
                yield self.passages[batch_start : batch_start + batch_size]

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

            if self.embeddings_2d is not None:
                row["x"] = self.embeddings_2d[i, 0]
                row["y"] = self.embeddings_2d[i, 1]

            if passage.highlighting is not None:
                row["highlight_start"] = passage.highlighting.start
                row["highlight_end"] = passage.highlighting.end
            else:
                row["highlight_start"] = None
                row["highlight_end"] = None

            rows.append(row)

        return pd.DataFrame(rows)

    def save(self, filepath: Path):
        """Save the corpus to a file."""

        # FIXME: TfIdfVectorizer cannot be saved, see  https://stackoverflow.com/questions/32764991/how-do-i-store-a-tfidfvectorizer-for-future-use-in-scikit-learn
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


@lru_cache
def get_vocabulary(vectorizer: TfidfVectorizer) -> list[str]:
    """Caching wrapper for getting the vocabulary of a vectorizer.

    Args:
        vectorizer (TfidfVectorizer): a vectorizer generated by count_vectorizer()

    Returns:
        list[str]: the vocabulary of the vectorizer.
    """

    return vectorizer.get_feature_names_out()
