import abc

from ..text.abstractcorpus import AbstractCorpus


class Visualizer(abc.ABC):
    def __init__(self, corpus: AbstractCorpus):
        self._corpus = corpus

    @abc.abstractmethod
    def visualize(self):
        return NotImplemented
