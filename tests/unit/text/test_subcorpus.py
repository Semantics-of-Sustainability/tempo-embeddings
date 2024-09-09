import numpy as np
import pytest

from tempo_embeddings.text.corpus import Corpus
from tempo_embeddings.text.passage import Passage
from tempo_embeddings.text.subcorpus import Subcorpus


class TestSubCorpus:
    def test_len(self):
        subcorpus = Subcorpus(Corpus([Passage("text 1"), Passage("text 2")]), [0])
        assert len(subcorpus) == 1

    def test_passages(self):
        parent_corpus = Corpus([Passage("text 1"), Passage("text 2")])

        subcorpus = Subcorpus(parent_corpus, [0], label="test")
        assert subcorpus.passages == [parent_corpus.passages[0]]
        assert subcorpus.passages[0] is parent_corpus.passages[0]

    def test_extend(self):
        parent_corpus = Corpus([Passage("text 1"), Passage("text 2")])

        subcorpus = Subcorpus(parent_corpus, [0], label="test")

        assert subcorpus.extend([Passage("text 3")]) == range(2, 3)
        assert subcorpus.passages == [Passage("text 1"), Passage("text 3")]
        assert subcorpus._indices == [0, 2]

    def test_add(self):
        parent_corpus = Corpus([Passage("text 1"), Passage("text 2")])

        subcorpus1 = Subcorpus(parent_corpus, [0], label="test1")
        subcorpus2 = Subcorpus(parent_corpus, [1], label="test2")
        assert subcorpus1 + subcorpus2 == Subcorpus(
            parent_corpus, [0, 1], label="test1+test2"
        )

        with pytest.raises(ValueError) as exc_info:
            subcorpus1 + Subcorpus(Corpus(), [0], label="test1")  # noqa: expression-not-assigned
            assert exc_info == "Cannot merge sub-corpora with different parent corpora."

    def test_compress_embeddings(self, corpus):
        corpus.embeddings = np.random.rand(len(corpus), 768)

        subcorpus = Subcorpus(corpus, [0])

        compressed = subcorpus.compress_embeddings()
        assert compressed.shape == (1, 2)
        np.testing.assert_array_equal(subcorpus.embeddings_2d, compressed)

        assert corpus.embeddings_2d.shape == (len(corpus), 2)

    def test_umap(self, corpus):
        subcorpus = Subcorpus(corpus, [0])
        assert subcorpus.umap is corpus.umap

        corpus.compress_embeddings()
        assert subcorpus.umap is corpus.umap
