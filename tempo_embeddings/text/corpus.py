import csv
import gzip
import logging
from collections import Counter, defaultdict
from itertools import groupby
from pathlib import Path
from typing import Any, Iterable, Optional

import joblib
import numpy as np
from sklearn.cluster import HDBSCAN
from sklearn.feature_extraction.text import TfidfVectorizer

from ..settings import DEFAULT_ENCODING
from .abstractcorpus import AbstractCorpus
from .passage import Passage
from .segmenter import Segmenter
from .subcorpus import Subcorpus


class Corpus(AbstractCorpus):
    """A Corpus implementation that holds the concrecte passages and embedings."""

    def __init__(self, passages: list[Passage] = None, label: Optional[Any] = None):
        super().__init__()

        self._passages: list[Passage] = passages or []
        self._label: Optional[str] = label
        self._vectorizer: TfidfVectorizer = None

    def __add__(self, other: "Corpus") -> "Corpus":
        new_label = self.label if self.label == other.label else None
        return Corpus(self._passages + other._passages, label=new_label)

    def __len__(self) -> int:
        """Return the number of passages in the corpus.

        Returns:
            int: The number of passages in the corpus.
        """
        return len(self.passages)

    def __repr__(self) -> str:
        return f"Corpus({self._label!r}, {len(self._passages)} passages)"

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

        def key_func(p):
            return p.metadata.get(key, default_value)

        return groupby(sorted(self._passages, key=key_func), key_func)

    @property
    def passages(self) -> list[Passage]:
        return self._passages

    @property
    def vectorizer(self) -> TfidfVectorizer:
        if self._vectorizer is None:
            self._vectorizer = AbstractCorpus.tfidf_vectorizer(self.passages)
        return self._vectorizer

    def _clusters(self, embeddings, max_clusters: int, **hdbscan_args) -> list[int]:
        """Cluster the embeddings using HDBSCAN.

        Recursively adapts the HDBSCAN parameters if the number of clusters exceeds `max_clusters`.

        Args:
            embeddings: The embeddings to cluster.
            max_clusters: The maximum number of clusters to create.
            **hdbscan_args: Additional keyword arguments to pass to HDBSCAN.
        Returns:
            list[int]: The cluster labels assigned by HDBSCAN.
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
    ) -> list[Subcorpus]:
        """Cluster the passages in the corpus.

        Args:
            max_clusters: The maximum number of clusters to create. If necessary, the epsilon HDBSCAN parameter is increased iteratively.
            use_2d_embeddings: Whether to use 2D embeddings instead of full embeddings for clustering. If necessary, they are computed. Defaults to True.
            **kwargs: Additional keyword arguments to pass to HDBSCAN.
        Returns:
            list[Subcorpus]: A list of Subcorpus objects, each representing a cluster. Labels are integers as assigned by HDBSCAN, with -1 indicating outliers.

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
        """A list of cluster labels assigned to each passage by HDBSCAN."""
        assert len(passage_clusters) == len(self)

        cluster_passages: dict[int, int] = defaultdict(list)
        """A dictionary mapping cluster labels/indices to passage indices in the corpus."""

        for passage_index, cluster in enumerate(passage_clusters):
            cluster_passages[cluster].append(passage_index)

        return [
            Subcorpus(self, passage_indices, cluster)
            for cluster, passage_indices in cluster_passages.items()
        ]

    def extend(self, passages: list[Passage]) -> list[int]:
        """Add multiple passages to the corpus.

        If a UMAP model is already present, 2D embeddings are computed for the new passages.

        Note: The UMAP model is *not* retrained with the added passages.
        In order to do so, call the corpus object's `compress_embeddings(recompute=True)` after adding new passages.

        Args:
            passages: The passages to add to the corpus.
        Returns:
            the indices in the corpus where the new passages were added, to be used in SubCorpus objects.
        """
        start_index = len(self._passages)
        self._passages.extend(passages)
        end_index = len(self._passages)

        if self._umap:
            self.embeddings_2d = np.append(
                self.embeddings_2d,
                self._umap.transform(self.embeddings[start_index:end_index]),
                axis=0,
            )

        return range(start_index, end_index)

    def batches(self, batch_size: int) -> Iterable[list[Passage]]:
        if batch_size <= 1:
            yield self.passages
        else:
            for batch_start in range(0, len(self.passages), batch_size):
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
