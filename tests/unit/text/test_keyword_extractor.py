import logging

import numpy as np
import pytest
from sklearn.exceptions import NotFittedError

from tempo_embeddings.text.corpus import Corpus
from tempo_embeddings.text.keyword_extractor import KeywordExtractor
from tempo_embeddings.text.passage import Passage


class TestKeywordExtractor:
    def test_fit(self, corpus):
        test_passages = [Passage("test text", embedding=np.random.rand(768).tolist())]

        extractor = KeywordExtractor(corpus)
        with pytest.raises(NotFittedError):
            extractor._vectorizer.transform(test_passages)

        extractor.fit()

        np.testing.assert_equal(
            extractor._vectorizer.transform(test_passages).toarray(),
            np.array([[0.7071067811865475, 0.7071067811865475]]),
        )
        assert extractor._feature_names.tolist() == ["test", "text"]

        with pytest.raises(RuntimeError):
            extractor.fit()

    @pytest.mark.parametrize(
        "n,exclude_words,expected",
        [
            (None, None, ["test", "text"]),
            (None, ["test"], ["text"]),
            (5, None, ["test", "text"]),
            (1, None, ["test"]),
            (1, ["test"], ["text"]),
            (0, None, []),
        ],
    )
    def test_top_words(self, corpus, n, exclude_words, expected, caplog):
        test_corpus = Corpus(
            [
                Passage(
                    "test a punct.uation text.", embedding=np.random.rand(768).tolist()
                )
            ]
        )
        extractor = KeywordExtractor(corpus)

        with caplog.at_level(logging.DEBUG):
            top_words = extractor.top_words(
                test_corpus, exclude_words=exclude_words, n=n
            )
            assert top_words == expected

        assert caplog.record_tuples == [
            ("root", logging.INFO, f"Fitting vectorizer on {corpus}."),
            ("root", logging.DEBUG, "Vectorizer fitted."),
        ]

        caplog.clear()
        with caplog.at_level(logging.DEBUG):
            list(extractor.top_words(test_corpus, exclude_words=exclude_words, n=n))
        assert "Vectorizer has already been fitted." in caplog.text
