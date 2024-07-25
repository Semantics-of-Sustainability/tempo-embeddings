import numpy as np
from numpy.typing import ArrayLike

from ..text.corpus import Corpus


class Analyser:
    def __init__(self, corpus: Corpus):
        self._corpus = corpus

    def std_variation(self):
        distances: ArrayLike = self._corpus.distances()
        return np.std(distances)
