import pytest
from transformers.tokenization_utils_base import CharSpan
from tempo_embeddings.text.corpus import Corpus
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
                Corpus({Passage("test line"): set([CharSpan(0, 4)])}),
            ),
        ],
    )
    def test_subcorpus(self, lines, token, expected):
        assert Corpus.from_lines(lines).subcorpus(token) == expected
