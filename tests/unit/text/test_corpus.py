from contextlib import nullcontext as does_not_raise
import pytest
from tempo_embeddings.text.corpus import Corpus
from tempo_embeddings.text.corpus import TokenInfo
from tempo_embeddings.text.corpus import TokenInfoPassage
from tempo_embeddings.text.passage import Passage


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
        "lines,token,expected",
        [
            ([], "test", []),
            (
                ["test token"],
                "test",
                [(Passage("test token"), 0)],
            ),
            (
                ["test token", "no match"],
                "test",
                [(Passage("test token"), 0)],
            ),
        ],
    )
    # pylint: disable=protected-access
    def test_find(self, lines, token, expected):
        assert list(Corpus.from_lines(lines)._find(token)) == expected

    @pytest.mark.parametrize(
        "corpus,token,metadata,expected",
        [
            (Corpus(), "test", {}, Corpus()),
            (
                Corpus([Passage("test line")]),
                "test",
                {},
                Corpus(
                    [Passage("test line")],
                    [(TokenInfo(start=0, end=4), Passage("test line"))],
                ),
            ),
            (
                Corpus(
                    [Passage("test line1"), Passage("test line2")],
                ),
                "line1",
                {},
                Corpus(
                    [Passage("test line1"), Passage("test line2")],
                    [(TokenInfo(start=5, end=10), Passage("test line1"))],
                ),
            ),
            (
                Corpus(
                    [
                        Passage("test line1", {"test": "value1"}),
                        Passage("test line2", {"test": "value2"}),
                    ],
                ),
                "test",
                {"test": "value1"},
                Corpus(
                    [
                        Passage("test line1", {"test": "value1"}),
                        Passage("test line2", {"test": "value2"}),
                    ],
                    [
                        (
                            TokenInfo(start=0, end=4),
                            Passage("test line1", {"test": "value1"}),
                        )
                    ],
                ),
            ),
            (
                Corpus(
                    [
                        Passage("test line1", {"key1": "value1", "key2": "value2"}),
                        Passage("test line2", {"key1": "value1", "key2": "value3"}),
                    ]
                ),
                "test",
                {"key1": "value1", "key2": "value2"},
                Corpus(
                    [
                        Passage("test line1", {"key1": "value1", "key2": "value2"}),
                        Passage("test line2", {"key1": "value1", "key2": "value3"}),
                    ],
                    [
                        (
                            TokenInfo(start=0, end=4),
                            Passage("test line1", {"key1": "value1", "key2": "value2"}),
                        ),
                    ],
                ),
            ),
            (
                Corpus(
                    [
                        Passage("test line1", {"test": "value1"}),
                        Passage("test line2", {"test": "value2"}),
                    ]
                ),
                "test",
                {"test": "value3"},
                Corpus(
                    [
                        Passage("test line1", {"test": "value1"}),
                        Passage("test line2", {"test": "value2"}),
                    ]
                ),
            ),
        ],
    )
    def test_subcorpus(self, corpus, token, metadata, expected):
        assert corpus.subcorpus(token, **metadata) == expected

    @pytest.mark.parametrize(
        "corpus,key,expected,expected_exception",
        [
            (Corpus(), "test key", [], does_not_raise()),
            (
                Corpus(passages=[Passage("text", metadata={"key": 1})]),
                "key",
                [],
                does_not_raise(),
            ),
            (
                Corpus(
                    [Passage("text", metadata={"key": 1, "other": 3})],
                    [
                        TokenInfoPassage(
                            TokenInfo(0, 4),
                            Passage("text", metadata={"key": 1, "other": 3}),
                        )
                    ],
                ),
                "key",
                [1],
                does_not_raise(),
            ),
            (
                Corpus(
                    [
                        Passage("text 1", metadata={"key": 1, "other": 3}),
                        Passage("text 2", metadata={"key": 2, "other": 2}),
                    ],
                    [
                        TokenInfoPassage(
                            TokenInfo(0, 4),
                            Passage("text 1", metadata={"key": 1, "other": 3}),
                        ),
                        TokenInfoPassage(
                            TokenInfo(5, 6),
                            Passage("text 2", metadata={"key": 2, "other": 2}),
                        ),
                    ],
                ),
                "key",
                [1, 2],
                does_not_raise(),
            ),
            (
                Corpus(
                    [
                        Passage("text 1", metadata={"key": 1, "other": 3}),
                        Passage("text 2", metadata={"other": 2}),
                    ],
                    [
                        TokenInfoPassage(
                            TokenInfo(0, 4),
                            Passage("text 1", metadata={"key": 1, "other": 3}),
                        ),
                        TokenInfoPassage(
                            TokenInfo(5, 6), Passage("text 2", metadata={"other": 2})
                        ),
                    ],
                ),
                "key",
                [1],
                pytest.raises(ValueError),
            ),
            (
                Corpus(
                    [
                        Passage("text 1", metadata={"key": 1, "other": 3}),
                        Passage("text 2", metadata={"key": 2, "other": 2}),
                    ],
                    [
                        TokenInfoPassage(
                            TokenInfo(0, 4),
                            Passage("text 1", metadata={"key": 1, "other": 3}),
                        )
                    ],
                ),
                "key",
                [1],
                does_not_raise(),
            ),
        ],
    )
    def test_get_token_metadatas(
        self, corpus: Corpus, key, expected, expected_exception
    ):
        with expected_exception:
            assert list(corpus.get_token_metadatas(key)) == expected

    def test_load_save(self, tmp_path):
        filepath = tmp_path / "corpus"
        corpus = Corpus(
            [
                Passage("text 1", metadata={"key": 1, "other": 3}),
                Passage("text 2", metadata={"key": 2}),
            ],
            [
                TokenInfoPassage(
                    TokenInfo(0, 4), Passage("text 1", metadata={"key": 1, "other": 3})
                )
            ],
        )
        corpus.embeddings_model_name = "test model"
        corpus.save(filepath)

        assert filepath.is_file()
        assert Corpus.load(filepath) == corpus
