import numpy as np
import pandas as pd
import pytest
from numpy.testing import assert_equal
from tempo_embeddings.text.corpus import Corpus
from tempo_embeddings.text.highlighting import Highlighting
from tempo_embeddings.text.passage import Passage
from tempo_embeddings.text.subcorpus import Subcorpus


class TestCorpus:
    @pytest.mark.parametrize(
        "lines,expected",
        [
            ([], Corpus()),
            (["line1", "line2"], Corpus([Passage("line1"), Passage("line2")])),
        ],
    )
    def test_from_lines(self, lines, expected):
        assert Corpus.from_lines(lines) == expected

    def test_add(self):
        assert Corpus.from_lines(["test1"]) + Corpus.from_lines(["test2"]) == Corpus(
            [Passage("test1"), Passage("test2")]
        )

    @pytest.mark.parametrize(
        "corpus,metadata_fields,expected",
        [
            (Corpus(), None, []),
            (Corpus([Passage("test")]), None, [{"text": "test", "corpus": "None"}]),
            (
                Corpus([Passage("test")], label="test label"),
                None,
                [{"text": "test", "corpus": "test label"}],
            ),
        ],
    )
    def test_hover_datas(self, corpus, metadata_fields, expected):
        assert corpus.hover_datas(metadata_fields=metadata_fields) == expected

    @pytest.mark.parametrize(
        "corpus,key,expected",
        [
            (Corpus(), "test key", []),
            (Corpus(passages=[Passage("text", metadata={"key": 1})]), "key", [1]),
            # FIXME: redundant test cases since passages are not filtered by highlighting
            (
                Corpus(
                    [
                        Passage(
                            "text",
                            metadata={"key": 1, "other": 3},
                            highlighting=Highlighting(0, 4),
                        )
                    ],
                ),
                "key",
                [1],
            ),
            (
                Corpus(
                    [
                        Passage(
                            "text 1",
                            metadata={"key": 1, "other": 3},
                            highlighting=Highlighting(0, 4),
                        ),
                        Passage(
                            "text 2",
                            metadata={"key": 2, "other": 2},
                            highlighting=Highlighting(5, 6),
                        ),
                    ],
                ),
                "key",
                [1, 2],
            ),
            (
                Corpus(
                    [
                        Passage(
                            "text 1",
                            metadata={"key": 1, "other": 3},
                            highlighting=Highlighting(0, 4),
                        ),
                        Passage(
                            "text 2",
                            metadata={"other": 2},
                            highlighting=Highlighting(5, 6),
                        ),
                    ],
                ),
                "key",
                [1, None],
            ),
            (
                Corpus(
                    [
                        Passage(
                            "text 1",
                            metadata={"key": 1, "other": 3},
                            highlighting=Highlighting(0, 4),
                        ),
                        Passage("text 2", metadata={"key": 2, "other": 2}),
                    ],
                ),
                "key",
                [1, 2],
            ),
        ],
    )
    def test_get_token_metadatas(self, corpus: Corpus, key, expected):
        assert list(corpus.get_metadatas(key)) == expected

    def test_load_save(self, tmp_path):
        # TODO: test/handle model (de)serialization
        filepath = tmp_path / "corpus"
        corpus = Corpus(
            [
                Passage("text 1", metadata={"key": 1, "other": 3}),
                Passage("text 2", metadata={"key": 2}),
            ],
            [Highlighting(0, 4)],
        )
        corpus.save(filepath)

        assert filepath.is_file()
        assert Corpus.load(filepath) == corpus

    @pytest.mark.parametrize(
        "corpus,expected",
        [
            (Corpus([Passage("test")]), np.array([None])),
            (Corpus([Passage("test", embedding=np.ones(10))]), np.ones((1, 10))),
            (
                Corpus(
                    [
                        Passage("test 1", embedding=np.ones(10)),
                        Passage("test 2", embedding=np.ones(10)),
                    ]
                ),
                np.ones((2, 10)),
            ),
        ],
    )
    def test_embeddings(self, corpus, expected):
        assert_equal(corpus.embeddings, expected)

    @pytest.mark.parametrize(
        "corpus",
        [
            (Corpus([Passage("test")])),
            (Corpus([Passage("test" + str(i)) for i in range(5)])),
        ],
    )
    def test_embeddings_setter(self, corpus):
        embeddings = [np.random.rand(10) for passage in corpus.passages]
        corpus.embeddings = np.array(embeddings)
        for passage, embedding in zip(corpus.passages, embeddings):
            assert_equal(passage.embedding, embedding)

    @pytest.mark.parametrize(
        "embeddings,expected",
        [
            (np.ones((1, 2)), pd.DataFrame({"x": [1], "y": [1]}, dtype=np.float64)),
            (
                np.array([[1, 2], [3, 4], [5, 6]], dtype=np.float64),
                pd.DataFrame({"x": [1, 3, 5], "y": [2, 4, 6]}, dtype=np.float64),
            ),
        ],
    )
    def test_embeddings_as_df(self, embeddings, expected):
        corpus = Corpus([Passage("test" + str(i)) for i in range(embeddings.shape[0])])
        corpus.embeddings = embeddings
        pd.testing.assert_frame_equal(corpus.embeddings_as_df(), expected)

    @pytest.mark.parametrize(
        "passages,expected",
        [
            ([], False),
            ([Passage("test")], False),
            ([Passage("test", embedding=np.random.rand(10))], True),
            (
                [
                    Passage("test 1", embedding=np.random.rand(10)),
                    Passage("test 2", embedding=np.random.rand(10)),
                ],
                True,
            ),
        ],
    )
    def test_has_embeddings(self, passages, expected):
        assert Corpus(passages).has_embeddings() == expected


class TestSubCorpus:
    def test_passages(self):
        parent_corpus = Corpus([Passage("text 1"), Passage("text 2")])

        subcorpus = Subcorpus(parent_corpus, [0], label="test")
        assert subcorpus.passages == [parent_corpus.passages[0]]
        assert subcorpus.passages[0] is parent_corpus.passages[0]

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
