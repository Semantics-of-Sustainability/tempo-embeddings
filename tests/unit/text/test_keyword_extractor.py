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
        "min_words,max_words,exclude_words,expected",
        [
            (1, 10, None, ["test", "text"]),
            (1, 10, ["test"], ["text"]),
            (1, 5, None, ["test", "text"]),
            (1, 1, None, ["test"]),
            (2, 2, None, ["test", "text"]),
            (1, 1, ["test"], ["text"]),
            (0, 0, None, []),
        ],
    )
    def test_top_words(
        self, corpus, min_words, max_words, exclude_words, expected, caplog
    ):
        # FIXME: no knee is ever detected in the test corpus (too small)

        test_corpus = Corpus(
            [
                Passage(
                    "test a punct.uation text.", embedding=np.random.rand(768).tolist()
                )
            ]
        )
        extractor = KeywordExtractor(corpus, exclude_words=exclude_words)

        with caplog.at_level(logging.DEBUG):
            top_words = extractor.top_words(
                test_corpus, min_words=min_words, max_words=max_words
            )
            assert top_words == expected

        assert f"Fitting vectorizer on {corpus}." in caplog.text
        assert "Vectorizer fitted." in caplog.text
        if max_words == min_words:
            assert "No need for knee detection." in caplog.text
        else:
            assert (
                "Knee detected at " in caplog.text or "No knee detected." in caplog.text
            )

        caplog.clear()
        with caplog.at_level(logging.DEBUG):
            extractor.top_words(test_corpus, min_words=min_words, max_words=max_words)
        assert "Vectorizer has already been fitted." in caplog.text
