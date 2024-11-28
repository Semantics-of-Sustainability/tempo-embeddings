import csv
import gzip
import logging
import random
from collections import Counter, defaultdict
from itertools import groupby, islice
from pathlib import Path
from typing import Any, Iterable, Optional

import joblib
import numpy as np
import pandas as pd
from scipy.spatial.distance import cosine
from sklearn.base import check_is_fitted
from sklearn.cluster import HDBSCAN
from sklearn.exceptions import NotFittedError
from umap.umap_ import UMAP

from ..settings import DEFAULT_ENCODING, OUTLIERS_LABEL, STRICT
from .passage import Passage
from .segmenter import Segmenter
from .util import any_to_int


class Corpus:
    """A Corpus implementation that holds the concrecte passages and embeddings."""

    def __init__(
        self,
        passages: Iterable[Passage] = None,
        label: Optional[Any] = None,
        *,
        umap_model: Optional["UMAP"] = None,
    ):
        """Create a new corpus.

        Args:
            passages: The passages to add to the corpus. Defaults to None. Type is not enforced, but should be immutable.
            label: The label of the corpus. Defaults to None.
            umap_model: The UMAP model to use for compressing embeddings.
                Defaults to None, hence a new model will be initialized.
        """
        super().__init__()

        self._passages: tuple[Passage, ...] = tuple(passages or [])
        self._label: Optional[str] = label
        self._umap = umap_model or UMAP()

        self._top_words: list[str] = []
        """A list of significant words in this corpus."""

    def __add__(self, other: "Corpus") -> "Corpus":
        if self.umap is other.umap:
            logging.info("Reusing commong UMAP model.")
            umap = self.umap
        elif self._is_fitted(self.umap) and self._is_fitted(other.umap):
            raise RuntimeError("Corpora have conflicting UMAP models.")
        elif self._is_fitted(self.umap):
            logging.info("Using UMAP model from first corpus.")
            umap = self.umap
        elif self._is_fitted(other.umap):
            logging.info("Using UMAP model from second corpus.")
            umap = other.umap
        else:
            logging.info("No UMAP model has been computed.")
            umap = None

        if self.top_words or other.top_words:
            logging.warning(
                "Dropping existing top words: %s, %s", self.top_words, other.top_words
            )

        label = " + ".join(
            (label for label in (self.label, other.label) if label != str(None))
        )
        return Corpus(
            self._passages + other._passages, label=label or None, umap_model=umap
        )

    def __eq__(self, other: object) -> bool:
        return (
            isinstance(other, Corpus)
            and self.label == other.label
            and other.passages == self.passages
        )

    def __hash__(self) -> int:
        return hash((self.label, tuple(self.passages)))

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
        return f"Corpus({self._label!r}, {len(self._passages)} passages, top words={self.top_words})"

    def _select_embeddings(self, use_2d_embeddings: bool):
        if use_2d_embeddings:
            if self.embeddings_2d is None and not self._is_fitted(self.umap):
                self.compress_embeddings()
            embeddings = self.embeddings_2d
        else:
            embeddings = self.embeddings

        return embeddings

    @property
    def embeddings(self) -> Optional[np.ndarray]:
        if not any(p.embedding is None for p in self.passages):
            return np.array([p.embedding for p in self.passages])
        else:
            logging.warning("No embeddings available.")
            return None

    @embeddings.setter
    def embeddings(self, embeddings: np.ndarray):
        for row, passage in zip(embeddings, self.passages, **STRICT):
            passage.embedding = row

    @property
    def embeddings_2d(self) -> Optional[np.ndarray]:
        if any(p.embedding_compressed for p in self.passages):
            # assuming all passages have embeddings_compressed if one has:
            return np.array([p.embedding_compressed for p in self.passages])
        else:
            logging.warning("No 2D embeddings available.")
            return None

    @embeddings_2d.setter
    def embeddings_2d(self, value: np.ndarray):
        for row, passage in zip(value, self.passages, **STRICT):
            passage.embedding_compressed = row.tolist()

    @property
    def label(self) -> Optional[str]:
        return str(self._label)

    @label.setter
    def label(self, value: str):
        self._label = value

    @property
    def top_words(self) -> list[str]:
        return self._top_words

    @top_words.setter
    def top_words(self, value: list[str]):
        if self._top_words:
            logging.warning(f"Overwriting top words: {self._top_words}")
        self._top_words = value

    def top_words_string(self, *, delimiter=";") -> str:
        if self.is_outliers():
            return OUTLIERS_LABEL
        elif self.top_words:
            return delimiter.join(self.top_words)
        else:
            logging.debug(f"No top words available for {self}.")
            return self.label

    @property
    def passages(self) -> list[Passage]:
        return self._passages

    @property
    def umap(self) -> UMAP:
        return self._umap

    @staticmethod
    def _is_fitted(model) -> bool:
        try:
            check_is_fitted(model)
            return True
        except NotFittedError:
            return False

    def is_outliers(self) -> bool:
        return OUTLIERS_LABEL in self.label

    def centroid(self, use_2d_embeddings: bool = True) -> np.ndarray:
        """The mean for all passage embeddings."""
        embeddings = self._select_embeddings(use_2d_embeddings)

        if embeddings is None:
            raise RuntimeError("No embeddings available.")
        return embeddings.mean(axis=0)

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

        if normalize and len(self) > 1:
            if norm := np.linalg.norm(distances):  # prevent division by zero
                distances /= norm

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
    def _fit_umap(self):
        if self._is_fitted(self._umap):
            raise RuntimeError("UMAP model has already been fitted.")
        elif self.embeddings is None:
            raise RuntimeError("Cannot fit UMAP, no embeddings available.")

        self._umap.fit(self.embeddings)

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
    ) -> Iterable["Corpus"]:
        """Cluster the passages in the corpus.

        Args:
            max_clusters: The maximum number of clusters to create. If necessary, the epsilon HDBSCAN parameter is increased iteratively.
            use_2d_embeddings: Whether to use 2D embeddings instead of full embeddings for clustering. If necessary, they are computed. Defaults to True.
            **kwargs: Additional keyword arguments to pass to HDBSCAN.
        Yields:
            Iterable[Corpus]: A generator of Corpus objects, each representing a cluster. Labels are integers as assigned by HDBSCAN, with -1 indicating outliers.

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
            label = OUTLIERS_LABEL if cluster == -1 else f"cluster {cluster}"
            yield Corpus(
                tuple(passages), label=f"{self.label}; {label}", umap_model=self._umap
            )

    def compress_embeddings(self) -> np.ndarray:
        """Compress the embeddings of the corpus using UMAP and stores them in the corpus

        Returns:
            np.ndarray: the compressed embeddings.

        Raises:
            ValueError: If the corpus has zero or exactly one embeddings.
        """
        self._fit_umap()
        self.embeddings_2d = self.umap.transform(self.embeddings)

        assert (
            self.embeddings_2d.shape[0] == len(self)
        ), f"{self.embeddings_2d.shape[0]} UMAP embeddings have been computed, but there are {len(self)} passages."

        return self.embeddings_2d

    def sample(self, sample_size: int, centroid_based: bool = False) -> "Corpus":
        """Sample a subset of the corpus, randomly or based on the distance to the centroid.

        Args:
            sample_size: The number of passages to sample.
            centroid_based: If True, the sample comprises the closest points to the centroid
                If False the sample is taken randomly.

        Returns:
            Corpus: A new corpus with the sampled passages.
        """
        if sample_size > len(self):
            raise ValueError(
                (f"Sample size ({sample_size}) exceeds corpus size ({len(self)})")
            )
        elif sample_size == len(self):
            logging.info("Sample size equals corpus size, returning the entire corpus.")
            return self

        if centroid_based:
            distances = self.distances(normalize=False)
            sample_indices = np.argsort(distances)[:sample_size]
        else:
            sample_indices = random.sample(range(len(self.passages)), sample_size)

        passages = [self.passages[i] for i in sample_indices]

        return Corpus(passages, label=self.label, umap_model=self.umap)

    def windows(
        self,
        step_size: int,
        *,
        start: Optional[int] = None,
        stop: Optional[int] = None,
        date_field: str = "date",
    ) -> Iterable["Corpus"]:
        """Split the corpus into windows of a given size.

        Assumes that the values in the given metadata field are integers or convertible to integers.

        Args:
            step_size: The step size between windows.
            start (Optional[int]): The start of the first window. Defaults to the minimum value in the metadata field.
            stop (Optional[int]): The end of the last window (excluded). Defaults to the maximum value in the metadata field.
            date_field (str): The metadata field to use for splitting the corpus. Defaults to 'date'
        Yields:
            Corpus: A new corpus with the passages for each non-empty window.
        """

        dates: list[int] = sorted(
            (any_to_int(date) for date in self.get_metadatas(date_field))
        )
        if start is None:
            start = dates[0]
        if stop is None:
            stop = dates[-1] + 1

        bin_starts = range(start, stop, step_size)
        passages = defaultdict(list)
        for passage in self.passages:
            try:
                bin: int = next(
                    bin_start
                    for bin_start in bin_starts
                    if any_to_int(passage.metadata.get(date_field))
                    in range(bin_start, min(bin_start + step_size, stop))
                )
                passages[bin].append(passage)
            except StopIteration:
                logging.debug(f"Passage {passage} is out of range {start}-{stop}.")

        for bin, bin_passages in passages.items():
            label = f"{self.label} {bin}-{min(bin+step_size, stop)}"
            if bin_passages:
                yield Corpus(tuple(bin_passages), label=label, umap_model=self.umap)
            else:
                logging.warning(f"No passages found for{label}")

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
        label = self.label[:50] + "..." if len(self.label) > 50 else self.label
        return [
            passage.hover_data(metadata_fields=metadata_fields)
            | {"corpus": label, "top words": self.top_words_string()}
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

    def to_dataframe(self) -> pd.DataFrame:
        """Transforms the Key Cluster information in a pandas Dataframe.

        Returns:
            a Pandas Dataframe

        """
        # TODO: add option for including compressed or full embedding or no centroid distances
        # TODO: merge with hover_datas()

        corpus_properties = {"corpus": self.label}
        if self.top_words:
            corpus_properties["top words"] = self.top_words_string()

        return pd.DataFrame(
            (
                passage.to_dict()
                | corpus_properties
                | {"distance_to_centroid": distance}
                for passage, distance in zip(
                    self.passages,
                    self.distances(normalize=True, use_2d_embeddings=True),
                )
            )
        )

    @classmethod
    def from_dataframe(
        cls,
        df: pd.DataFrame,
        *,
        label: str = "",
        umap_model: Optional[UMAP] = None,
        text_field: str = "text",
    ) -> "Corpus":
        """Create a corpus from a Pandas DataFrame.

        Args:
            df: The DataFrame to create the corpus from.
            label: The label of the corpus. Defaults to an empty string.
            umap_model (optional): The UMAP model to use for compressing embeddings. Defaults to None (new model).
            text_field: The name of the column containing the text. Defaults to 'text'.
        Returns:
            Corpus: The corpus created from the DataFrame.
        """

        return cls(
            [
                Passage.from_df_row(row, text_field=text_field)
                for _, row in df.iterrows()
            ],
            label=label,
            umap_model=umap_model,
        )

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
    def from_csv_files(cls, files: Iterable[Path], **kwargs):
        """Read input data from multiple CSV files in a directory."""
        return sum(
            (corpus for file in files for corpus in cls.from_csv_file(file, **kwargs)),
            Corpus(),
        )

    @classmethod
    def from_csv_file(
        cls,
        filepath: Path,
        text_columns: list[str],
        *,
        max_corpus_size: Optional[int] = None,
        segmenter: Segmenter,
        filter_terms: list[str] = None,
        encoding=DEFAULT_ENCODING,
        compression: Optional[str] = None,
        **dict_reader_kwargs,
    ) -> Iterable["Corpus"]:
        """Read input data from a CSV file."""

        open_func = gzip.open if compression == "gzip" else open

        with open_func(filepath, "rt", encoding=encoding) as f:
            try:
                yield from cls.from_csv_stream(
                    f,
                    text_columns,
                    filter_terms=filter_terms,
                    max_corpus_size=max_corpus_size,
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
        max_corpus_size: Optional[int] = None,
        segmenter: Segmenter,
        filter_terms: list[str] = None,
        **dict_reader_kwargs,
    ) -> Iterable["Corpus"]:
        reader = csv.DictReader(file_handler, **dict_reader_kwargs)

        passages = iter(
            segmenter.passages_from_dict_reader(
                reader,
                provenance=file_handler.name,
                text_columns=text_columns,
                filter_terms=filter_terms,
            )
        )
        if max_corpus_size:
            while batch := tuple(islice(passages, max_corpus_size)):
                yield Corpus(batch)
        else:
            yield Corpus(tuple(passages))
