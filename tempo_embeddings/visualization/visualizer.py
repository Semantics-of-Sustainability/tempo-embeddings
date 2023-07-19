import abc
from ..text.corpus import Corpus


class Visualizer(abc.ABC):
    def __init__(self, corpus: Corpus):
        self._corpus = corpus

    @abc.abstractmethod
    def visualize(self):
        return NotImplemented