import pytest
from tempo_embeddings.text.corpus import Corpus
from tempo_embeddings.text.corpus import TokenInfo
from tempo_embeddings.text.passage import Passage


class TestCorpus:
    @pytest.mark.parametrize(
        "lines,expected",
        [
            ([], Corpus()),
            (
                ["line1", "line2"],
                Corpus({Passage("line1"): set(), Passage("line2"): set()}),
            ),
        ],
    )
    def test_from_lines(self, lines, expected):
        assert Corpus.from_lines(lines) == expected

    @pytest.mark.parametrize(
        "passages,expected",
        [
            ([], Corpus()),
            ([Passage("text")], Corpus({Passage("text"): set()})),
            (
                [Passage("text 1"), Passage("text 2")],
                Corpus(passages={Passage("text 1"): set(), Passage("text 2"): set()}),
            ),
            (
                [
                    Passage("text", metadata={"key": 1}),
                    Passage("text", metadata={"key": 2}),
                ],
                Corpus(
                    passages={
                        Passage("text", metadata={"key": 1}): set(),
                        Passage("text", metadata={"key": 2}): set(),
                    }
                ),
            ),
        ],
    )
    def test_from_passages(self, passages, expected):
        assert Corpus.from_passages(passages) == expected

    def test_add(self):
        assert Corpus.from_lines(["test1"]) + Corpus.from_lines(["test2"]) == Corpus(
            {Passage("test1"): set(), Passage("test2"): set()}
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
    def test_find(self, lines, token, expected):
        assert list(Corpus.from_lines(lines).find(token)) == expected

    @pytest.mark.parametrize(
        "lines,token,expected",
        [
            ([], "test", Corpus()),
            (
                ["test line"],
                "test",
                Corpus({Passage("test line"): {TokenInfo(start=0, end=4)}}),
            ),
        ],
    )
    def test_subcorpus(self, lines, token, expected):
        assert Corpus.from_lines(lines).subcorpus(token) == expected

    @pytest.mark.parametrize(
        "corpus, expected",
        [
            (Corpus(), False),
            (Corpus({Passage("test"): set()}), False),
            (Corpus({Passage("test"): {TokenInfo(0, 4)}}), False),
            (Corpus({Passage("test"): {TokenInfo(0, 4, embedding=[0.1, 0.2])}}), True),
        ],
    )
    def test_has_embeddings(self, corpus, expected):
        assert corpus.has_embeddings() == expected

    @pytest.mark.parametrize(
        "corpus, expected",
        [
            (Corpus(), []),
            (Corpus.from_passages([Passage("text")]), []),
            (Corpus({Passage("text"): {TokenInfo(0, 4)}}), [Passage("text")]),
            (
                Corpus(
                    {Passage("text 1"): {TokenInfo(0, 4)}, Passage("text 2"): set()}
                ),
                [Passage("text 1")],
            ),
            (
                Corpus({Passage("text"): {TokenInfo(0, 4), TokenInfo(3, 4)}}),
                [Passage("text"), Passage("text")],
            ),
        ],
    )
    def test_token_passages(self, corpus, expected):
        assert list(corpus.token_passages()) == expected

    @pytest.mark.parametrize(
        "corpus,key,expected,expected_exception",
        [
            (Corpus(), "test key", [], None),
            (
                Corpus(passages={Passage("text", metadata={"key": 1}): set()}),
                "key",
                [],
                None,
            ),
            (
                Corpus(
                    {
                        Passage("text", metadata={"key": 1, "other": 3}): {
                            TokenInfo(0, 4)
                        }
                    }
                ),
                "key",
                [1],
                None,
            ),
            (
                Corpus(
                    {
                        Passage("text 1", metadata={"key": 1, "other": 3}): {
                            TokenInfo(0, 4)
                        },
                        Passage("text 2", metadata={"key": 2}): {TokenInfo(5, 6)},
                    }
                ),
                "key",
                [1, 2],
                None,
            ),
            (
                Corpus(
                    {
                        Passage("text 1", metadata={"key": 1, "other": 3}): {
                            TokenInfo(0, 4)
                        },
                        Passage("text 2", metadata={"other": 2}): {TokenInfo(5, 6)},
                    }
                ),
                "key",
                [1],
                ValueError,
            ),
            (
                Corpus(
                    {
                        Passage("text 1", metadata={"key": 1, "other": 3}): {
                            TokenInfo(0, 4)
                        },
                        Passage("text 2", metadata={"key": 2}): set(),
                    }
                ),
                "key",
                [1],
                None,
            ),
        ],
    )
    def test_get_metadatas(self, corpus, key, expected, expected_exception):
        if expected_exception is None:
            assert list(corpus.get_metadatas(key)) == expected
        else:
            with pytest.raises(expected_exception):
                list(corpus.get_metadatas(key))
