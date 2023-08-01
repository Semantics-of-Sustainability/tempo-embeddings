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
        "passage, token, expected",
        [
            (Passage(""), "test", []),
            (Passage("test"), "test", [Highlighting(0, 4, Passage("test"))]),
            (Passage("no match"), "test", []),
            (
                Passage("one test match"),
                "test",
                [Highlighting(4, 8, Passage("one test match"))],
            ),
            (
                Passage("one test another test match"),
                "test",
                [
                    Highlighting(4, 8, Passage("one test another test match")),
                    Highlighting(17, 21, Passage("one test another test match")),
                ],
            ),
        ],
    )
    def test_findall(self, passage, token, expected):
        assert list(passage.findall(token)) == expected

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
