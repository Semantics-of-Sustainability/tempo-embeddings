from typing import TYPE_CHECKING, Optional

import numpy as np

from .abstractcorpus import AbstractCorpus
from .passage import Passage

if TYPE_CHECKING:
    from .corpus import Corpus


class Subcorpus(AbstractCorpus):
    """A partial corpus referencing to a parent corpus."""

    def __init__(
        self, parent_corpus: "Corpus", indices: list[int], label: Optional[str] = None
    ) -> None:
        super().__init__()

        self._parent_corpus = parent_corpus
        self._indices = indices
        self._label = label

    @property
    def embeddings_2d(self) -> Optional[np.ndarray]:
        return self._parent_corpus.embeddings_2d[self._indices]

    def __repr__(self) -> str:
        return f"Subcorpus({self._label!r}, {self._indices[:10]!r})"

    def __add__(self, other: "Subcorpus", new_label: Optional[str] = None):
        if self._parent_corpus == other._parent_corpus:
            indices = set(self._indices + other._indices)
            label = new_label or "+".join([self._label, other._label])
        else:
            raise ValueError("Cannot merge sub-corpora with different parent corpora.")

        return Subcorpus(self._parent_corpus, list(indices), label)

    @property
    def passages(self) -> list[Passage]:
        return [self._parent_corpus.passages[i] for i in self._indices]

    def compress_embeddings(self) -> np.ndarray:
        # UMAP should be fitted on all embeddings of the parent corpus
        raise NotImplementedError("Subcorpus does not support compress_embeddings")

    def extend(self, passages: list[Passage]) -> list[int]:
        """Add multiple passages to the corpus.

        Args:
            passages: The passages to add to the corpus.
        Returns:
            the indices in the corpus where the new passages were added, to be used in SubCorpus objects.
        """

        new_indices = self._parent_corpus.extend(passages)
        self._indices.extend(new_indices)
        return new_indices

    @property
    def vectorizer(self):
        return self._parent_corpus.vectorizer
