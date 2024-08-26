from typing import TYPE_CHECKING, Optional

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

    def _embeddings_2d(self):
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

    @property
    def vectorizer(self):
        return self._parent_corpus.vectorizer
