import logging
from collections import defaultdict
from functools import reduce
from operator import add
from typing import Iterable
from typing import Optional
import numpy as np
from sklearn.cluster import HDBSCAN
from sklearn.feature_extraction.text import TfidfVectorizer
from ..settings import OUTLIERS_LABEL
from ..visualization.clusters import ClusterVisualizer
from ..visualization.plotly import PlotlyVisualizer
from .corpus import Corpus
from .passage import Passage


class Cluster:
    """A container with helper functions for multiple corpora."""

    def __init__(
        self,
        parent_corpus: Corpus,
        *,
        vectorizer: Optional[TfidfVectorizer] = None,
        n_topic_words: int = 1,
    ) -> None:
        """Create a new cluster.

        Args:
            parent_corpus: The parent corpus of the cluster
            vectorizer: The Tf-IDF vectorizer to use for the cluster;
                used for extracting topic words and sub-corpus labels
            n_topic_words: The number of topic words used for setting sub-corpus labels.
                Defaults to 1
        """
        self._parent = parent_corpus
        self._vectorizer = vectorizer
        self._n_topic_words = n_topic_words

        self._subcorpora: list[Corpus] = []

    def __repr__(self) -> str:
        return f"Cluster({self._parent!r}, {self._subcorpora!r})"

    def labels(self) -> list[str]:
        """Returns the labels of the sub-corpora."""
        return [corpus.label for corpus in self._subcorpora]

    @property
    def vectorizer(self) -> TfidfVectorizer:
        return self._vectorizer

    def _get_corpora_by_label(self, label: str) -> list[Corpus]:
        """Returns the corpora with the given label.

        Args:
            label: The label of the corpora to return.

        Returns:
            A list of corpora with the given label.
            Should be a list of length 0 or 1; a warning is issued if there are multiple
        """
        if self._subcorpora is None:
            raise ValueError("No subcorpora available")

        matches = [corpus for corpus in self._subcorpora if corpus.label == label]

        if len(matches) > 1:
            logging.warning("Found multiple corpora with label '%s'.", label)

        return matches

    def select_subcorpora(self, *labels: str) -> Iterable[Corpus]:
        """Selects the corpora with the given labels.

        Args:
            *labels: The labels of the corpora to select.

        Returns:
            The corpora with the given labels.

        Raises:
            ValueError: If no corpus with the given label is found.
            RuntimeError: If multiple corpora with the given label are found.

        """
        for label in labels:
            match self._get_corpora_by_label(label):
                case []:
                    raise ValueError(f"No corpus with label '{label}'")
                case [corpus]:
                    yield corpus
                case [*corpora]:
                    raise RuntimeError(
                        f"{len(corpora)} corpora found  with label '{label}'."
                    )

    def set_topic_labels(
        self,
        *corpora: Iterable[Corpus],
        exclude_word: Optional[str] = None,
        exact_match: bool = True,
        stopwords: set[str] = None,
    ) -> Iterable[str]:
        """Sets the topic labels for all subcorpora.

        Args:
            *corpora: The corpora to set the topic labels for.
            exclude_word: a word to exclude from the topic labels.
            exact_match: whether to use exact matching for the exclude word.
                Defaults to True
            stopwords: words to exclude

        Returns:
            The topic labels of the corpora.

        """
        if not self._n_topic_words:
            raise RuntimeError(f"Number of topic words set to {self._n_topic_words}")

        for corpus in corpora:
            if corpus.label == -1:
                corpus.label = OUTLIERS_LABEL
            else:
                corpus.set_topic_label(
                    self.vectorizer,
                    n=self._n_topic_words,
                    exclude_word=exclude_word or self._parent.label,
                    exact_match=exact_match,
                    stopwords=stopwords,
                )
            yield corpus.label

    def umap_embeddings(self, *child_labels) -> np.ndarray:
        """Returns the UMAP embeddings of the parent corpus or the given sub-corpora.

        Args:
            *child_labels: The labels of the sub-corpora to return the embeddings from.
                If empty, the embeddings of the parent corpus are returned.

        Returns:
            The UMAP embeddings of the parent corpus or the given sub-corpora.
        """
        if child_labels:
            embeddings = [
                embeddings
                for corpus in self.select_subcorpora(*child_labels)
                for embeddings in corpus.umap_embeddings()
            ]
        else:
            embeddings = self._parent.umap_embeddings()
        return np.array(embeddings)

    def passages(self, *child_labels) -> list[Passage]:
        """Returns the passages of the parent corpus or the given sub-corpora.

        Args:
            *child_labels: The labels of the sub-corpora to return the passages from.
                If empty, the passages of the parent corpus are returned.

        Returns:
            The passages of the parent corpus or the given sub-corpora.
        """
        if child_labels:
            passages = [
                passage
                for corpus in self.select_subcorpora(*child_labels)
                for passage in corpus.passages
            ]
        else:
            passages = self._parent.passages
        return passages

    def _cluster(self, *corpus_labels, **kwargs) -> list[Corpus]:
        cluster_labels: list[int] = (
            HDBSCAN(**kwargs)
            .fit_predict(self.umap_embeddings(*corpus_labels))
            .astype(int)
            .tolist()
        )

        clusters: dict[int, list[Passage]] = defaultdict(list)
        for label, passage in zip(
            cluster_labels, self.passages(*corpus_labels), strict=True
        ):
            clusters[label].append(passage)

        return [
            Corpus(
                passages=passages,
                model=self._parent.model,
                umap=self._parent.umap,
                label=label,
            )
            for label, passages in clusters.items()
        ]

    def cluster(self, stopwords: set[str] = None, **kwargs) -> list[str]:
        """Clusters the parent corpus and creates initial subcorpora.

        Args:
            stopwords: if given, use stopwords for filtering topic labels.
            **kwargs: Keyword arguments passed to the clustering algorithm.

        Returns:
            the labels of the new sub-corpora

        Raises:
            RuntimeError: If the parent corpus is already clustered.
        """

        if self._subcorpora:
            raise RuntimeError("Parent corpus is already clustered")

        self._subcorpora = self._cluster(**kwargs)

        if self._n_topic_words:
            labels = list(self.set_topic_labels(*self._subcorpora, stopwords=stopwords))

            assert len(labels) == len(self._subcorpora)
            if len(set(labels)) != len(self._subcorpora):
                logging.warning("Labels are not unique: %s", str(labels))

        return self.labels()

    def cluster_subcorpus(self, label, **kwargs) -> list[str]:
        """Split a sub-corpus into clusters.

        Args:
            label: The label of the sub-corpus to split.

        Returns:
            the labels of the new sub-corpora
        """

        subcorpora = self._get_corpora_by_label(label)
        if len(subcorpora) != 1:
            raise ValueError(
                f"Found {len(subcorpora)} subcorpora with label '{label}'."
            )

        # FIXME: this calls self._get_corpus_by_label again downstream
        sub_clusters: list[Corpus] = self._cluster(label, **kwargs)

        if self._n_topic_words:
            new_labels = list(self.set_topic_labels(*sub_clusters))
        else:
            new_labels = [corpus.label for corpus in sub_clusters]

        if len(set(self.labels())) < len(self.labels()):
            # TODO: merge subclusters with same label; only for outliers?
            logging.warning("Duplicate labels in subcorpora")

        self._subcorpora.remove(subcorpora[0])
        self._subcorpora.extend(sub_clusters)

        return new_labels

    def merge(self, label1: str, label2: str, *labels) -> str:
        """Merge sub-corpora with the given labels.

        Args:
            label1: The label of the first corpus.
            label2: The label of the second corpus.
            *labels: The labels of the remaining corpora.

        Returns: the label of the merged sub-corpus

        """
        corpora = list(self.select_subcorpora(label1, label2, *labels))

        merged: Corpus = reduce(add, corpora)

        if self._n_topic_words:
            self.set_topic_labels(merged)

        for corpus in corpora:
            self._subcorpora.remove(corpus)
        self._subcorpora.append(merged)

        return merged.label

    def visualize(self, metadata_fields: Optional[list[str]] = None):
        corpora = self._subcorpora or [self._parent]
        visualizer = PlotlyVisualizer(*corpora)
        visualizer.visualize(
            metadata_fields=metadata_fields or list(self._parent.metadata_fields())
        )

    def scatter_plot(self):
        visualizer = ClusterVisualizer(self._subcorpora or [self._parent])
        visualizer.visualize()
