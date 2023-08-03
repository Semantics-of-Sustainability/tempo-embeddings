from contextlib import nullcontext as does_not_raise
import pytest
from tempo_embeddings.text.highlighting import Highlighting
from tempo_embeddings.text.passage import Passage


class TestPassage:
    @pytest.mark.parametrize(
        "text,metadata,expected",
        [
            ("", None, Passage("", {})),
            ("", {}, Passage("", {})),
            ("test", None, Passage("test", {})),
            (" test ", None, Passage("test", {})),
            ("test", {"key": "value"}, Passage("test", {"key": "value"})),
        ],
    )
    def test_init(self, text, metadata, expected):
        assert Passage(text, metadata) == expected

    @pytest.mark.parametrize(
        "passage, token, expected_passage, expected_result",
        [
            (Passage(""), "test", Passage(""), False),
            (
                Passage("test"),
                "test",
                Passage("test", highlightings=[Highlighting(0, 4)]),
                True,
            ),
            (Passage("no match"), "test", Passage("no match"), False),
            (
                Passage("one test match"),
                "test",
                Passage("one test match", highlightings=[Highlighting(0, 4)]),
                True,
            ),
            (
                Passage("one test another test match"),
                "test",
                Passage(
                    "one test another test match",
                    highlightings=[Highlighting(4, 8), Highlighting(17, 21)],
                ),
                True,
            ),
        ],
    )
    def test_add_highlightings(self, passage, token, expected_passage, expected_result):
        assert passage.add_highlightings(token) == expected_result
        assert passage == expected_passage

    @pytest.mark.parametrize(
        "text,window_size,window_overlap,expected",
        [
            ("test text", None, None, [Passage("test text")]),
            ("test text", 5, None, [Passage("test"), Passage("text")]),
            ("test text", 5, 0, [Passage("test"), Passage("text")]),
            ("test text", 5, 2, [Passage("test"), Passage("t tex"), Passage("ext")]),
        ],
    )
    def test_from_text(self, text, window_size, window_overlap, expected):
        assert (
            list(
                Passage.from_text(
                    text, window_size=window_size, window_overlap=window_overlap
                )
            )
            == expected
        )

    @pytest.mark.parametrize(
        "passage,labels,expected,exception_context",
        [
            (Passage("test text"), [], [Passage("test text")], does_not_raise()),
            (
                Passage("test text"),
                [1],
                [Passage("test text")],
                pytest.raises(ValueError),
            ),
            (
                Passage("test text", highlightings=[Highlighting(0, 4)]),
                [1],
                [Passage("test text", highlightings=[Highlighting(0, 4)])],
                does_not_raise(),
            ),
            (
                Passage(
                    "test text", highlightings=[Highlighting(0, 4), Highlighting(6, 10)]
                ),
                [1, 2],
                [
                    Passage("test text", highlightings=[Highlighting(0, 4)]),
                    Passage("test text", highlightings=[Highlighting(6, 10)]),
                ],
                does_not_raise(),
            ),
        ],
    )
    def test_split_highlightings(self, passage, labels, expected, exception_context):
        with exception_context:
            assert list(passage.split_highlightings(labels)) == expected

    @pytest.mark.parametrize(
        "passage,key,strict,expected,expected_exception",
        [
            (Passage("test text", metadata={}), "test key", False, None, None),
            (Passage("test text", metadata={}), "test key", True, None, KeyError),
            (Passage("text", metadata={"key": 1}), "key", False, 1, None),
        ],
    )
    def test_get_metadata(
        self, passage, key, strict, expected, expected_exception
    ):  # pylint: disable=too-many-arguments
        if expected_exception is None:
            assert passage.get_metadata(key, strict=strict) == expected
        else:
            with pytest.raises(expected_exception):
                passage.get_metadata(key, strict=strict)

    @pytest.mark.parametrize("passage,expected", [(Passage("test"), ["test"])])
    @pytest.mark.skip(reason="Not implemented")
    def test_words(self, passage, expected):
        assert list(passage.words()) == expected
