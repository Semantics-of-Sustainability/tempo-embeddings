import pytest
from tempo_embeddings.text.passage import Passage
from tempo_embeddings.text.types import TokenInfo


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
            (Passage("test"), "test", [0]),
            (Passage("no match"), "test", []),
            (Passage("one test match"), "test", [4]),
            (Passage("one test another test match"), "test", [4, 17]),
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

    @pytest.mark.parametrize(
        "passage,token_info,expected",
        [
            (Passage("test token"), TokenInfo(0, 4), 0),
            (Passage("test token"), TokenInfo(1, 4), 0),
            (Passage("test token"), TokenInfo(5, 10), 5),
            (Passage("test token"), TokenInfo(6, 10), 5),
            (Passage("test token"), TokenInfo(10, 10), 5),
            (Passage("test,token"), TokenInfo(6, 10), 5),
        ],
    )
    def test_word_begin(self, passage, token_info, expected):
        assert passage.word_begin(token_info) == expected

    @pytest.mark.parametrize(
        "passage,token_info,expected",
        [
            (Passage("test token"), TokenInfo(0, 4), 4),
            (Passage("test token"), TokenInfo(0, 3), 4),
            (Passage("test token"), TokenInfo(5, 10), 10),
            (Passage("test token"), TokenInfo(5, 9), 10),
            (Passage("test token"), TokenInfo(5, 5), 10),
        ],
    )
    def test_word_end(self, passage, token_info, expected):
        assert passage.word_end(token_info) == expected
