import pytest
from tempo_embeddings.text.corpus import Corpus
from tempo_embeddings.text.highlighting import Highlighting
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
        "corpus,token,metadata,expected",
        [
            (Corpus(), "test", {}, Corpus()),
            (
                Corpus([Passage("test line")]),
                "test",
                {},
                Corpus([Passage("test line", highlighting=Highlighting(0, 4))]),
            ),
            (
                Corpus(
                    [Passage("test line1"), Passage("test line2")],
                ),
                "line1",
                {},
                Corpus([Passage("test line1", highlighting=Highlighting(5, 10))]),
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
                        Passage(
                            "test line1",
                            {"test": "value1"},
                            highlighting=Highlighting(0, 4),
                        )
                    ]
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
                        Passage(
                            "test line1",
                            {"key1": "value1", "key2": "value2"},
                            highlighting=Highlighting(0, 4),
                        )
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
                Corpus(),
            ),
        ],
    )
    def test_subcorpus(self, corpus, token, metadata, expected):
        # TODO: test with exact_match=True
        assert corpus.subcorpus(token, exact_match=False, **metadata) == expected

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
        filepath = tmp_path / "corpus"
        corpus = Corpus(
            [
                Passage("text 1", metadata={"key": 1, "other": 3}),
                Passage("text 2", metadata={"key": 2}),
            ],
            [Highlighting(0, 4)],
        )
        corpus.model = None
        corpus.save(filepath)

        assert filepath.is_file()
        assert Corpus.load(filepath) == corpus
