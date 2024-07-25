from contextlib import nullcontext as does_not_raise

import numpy as np
import pytest
from numpy.testing import assert_equal

from tempo_embeddings.embeddings.vector_database import VectorDatabaseManagerWrapper
from tempo_embeddings.text.corpus import Corpus
from tempo_embeddings.text.passage import Passage


class TestVectorDatabaseManagerWrapper:
    @pytest.mark.parametrize(
        "corpus, expected_exception",
        [
            (Corpus(), pytest.raises(ValueError)),
            (Corpus(list(Passage.from_text("test text"))), does_not_raise()),
            (Corpus(list(Passage.from_text("test text")) * 10), does_not_raise()),
        ],
    )
    def test_compress_embeddings(self, corpus: Corpus, expected_exception):
        # generate random embedding vectors
        for passage in corpus.passages:
            passage.embedding = np.random.rand(10).tolist()

        with expected_exception:
            compressed = VectorDatabaseManagerWrapper.compress_embeddings(corpus)

            assert compressed.shape == (len(corpus), 2)

            if len(corpus) == 1:
                # For single samples, the output should be zeros
                assert_equal(compressed, np.zeros((len(corpus), 2)))
