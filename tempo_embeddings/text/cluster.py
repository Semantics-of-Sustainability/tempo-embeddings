import logging
from collections import defaultdict
from typing import Iterable
from typing import Optional
import numpy as np
from sklearn.cluster import HDBSCAN
from sklearn.feature_extraction.text import TfidfVectorizer
from ..settings import OUTLIERS_LABEL
from .corpus import Corpus
from .passage import Passage


class Cluster:
    """A container with helper functions for multiple corpora."""

    def __init__(
        self, parent_corpus: Corpus, *, vectorizer: Optional[TfidfVectorizer] = None
    ) -> None:
        self._parent = parent_corpus
        self._vectorizer = vectorizer

        self._subcorpora: list[Corpus] = []

    def __repr__(self) -> str:
        return f"Cluster({self._parent!r}, {self._subcorpora!r})"

    def labels(self) -> list[str]:
        """Returns the labels of the corpora."""
        return [corpus.label for corpus in self._subcorpora]

    @property
    def vectorizer(self) -> TfidfVectorizer:
        """The vectorizer used to create the corpus."""
        if self._vectorizer is None:
            if self._stopwords:
                stop_words = self._stopwords
            else:
                stop_words = "english"
                logging.warning(
                    "No stopwords provided, using default '(%s').", stop_words
                )

            self._vectorizer = self._parent.tfidf_vectorizer(stop_words=stop_words)

        return self._vectorizer

    def _get_corpus_by_label(self, label: str) -> Optional[Corpus]:
        """Returns the corpus with the given label."""
        if self._subcorpora is None:
            raise ValueError("No subcorpora available")

        return [corpus for corpus in self._subcorpora if corpus.label == label]

    def select_corpora(self, *labels: str) -> Iterable[Corpus]:
        """Selects the corpora with the given labels."""
        if labels:
            for label in labels:
                match self._get_corpus_by_label(label):
                    case []:
                        raise ValueError(f"No corpus with label '{label}'")
                    case [corpus]:
                        yield corpus
                    case [*corpora]:
                        raise ValueError(
                            f"{len(corpora)} corpora found  with label '{label}'."
                        )
        else:
            yield self._parent

    def set_topic_labels(
        self, *corpora, n: int = 1, exclude_word: Optional[str] = None
    ) -> None:
        """Sets the topic labels for all subcorpora."""

        for corpus in corpora:
            if corpus.label == -1:
                corpus.label = OUTLIERS_LABEL
            else:
                corpus.set_topic_label(
                    self.vectorizer,
                    n=n,
                    exclude_word=exclude_word or self._parent.label,
                )

    def umap_embeddings(self, *child_labels) -> np.ndarray:
        return np.array(
            [
                embeddings
                for corpus in self.select_corpora(*child_labels)
                for embeddings in corpus.umap_embeddings()
            ]
        )

    def passages(self, *child_labels) -> list[Passage]:
        return [
            passage
            for corpus in self.select_corpora(*child_labels)
            for passage in corpus.passages
        ]

    def _cluster(self, *corpora, **kwargs) -> list[Corpus]:
        labels: list[int] = (
            HDBSCAN(**kwargs)
            .fit_predict(self.umap_embeddings(*corpora))
            .astype(int)
            .tolist()
        )

        clusters: dict[int, list[Passage]] = defaultdict(list)
        for label, passage in zip(labels, self.passages(*corpora), strict=True):
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

    def cluster(self, set_topic_labels: int = 1, **kwargs) -> None:
        """Clusters the parent corpus and creates initial subcorpora."""

        if self._subcorpora:
            raise ValueError("Parent corpus already clustered")

        self._subcorpora = self._cluster(**kwargs)

        if set_topic_labels > 0:
            self.set_topic_labels(*self._subcorpora, n=set_topic_labels)

    def cluster_child(self, label, set_topic_labels: int = 1, **kwargs):
        child = self._get_corpus_by_label(label)
        if child is None:
            raise ValueError(f"No child corpus with label '{label}'")

        sub_clusters: list[Corpus] = self._cluster(child.umap_embeddings(), **kwargs)

        if set_topic_labels > 0:
            self.set_topic_labels(*sub_clusters, n=set_topic_labels)

        if len(set(self.labels()) < len(self.labels())):
            # TODO: merge subclusters with same label; only for outliers?
            raise ValueError("Duplicate labels in subcorpora")

        self._subcorpora.remove(child)
        self._subcorpora.extend(sub_clusters)
