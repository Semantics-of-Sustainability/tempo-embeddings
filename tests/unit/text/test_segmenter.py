from contextlib import nullcontext as does_not_raise

import pytest

from tempo_embeddings.text.passage import Passage
from tempo_embeddings.text.segmenter import (
    Segmenter,
    SentenceSplitterSegmenter,
    StanzaSegmenter,
    WindowSegmenter,
    WtpSegmenter,
)


class TestSegmenter:
    @pytest.mark.parametrize(
        "segmenter, language, kwargs, expected_type, expected_exception",
        [
            ("wtp", "en", {}, WtpSegmenter, does_not_raise()),
            ("wtp", "nl", {}, WtpSegmenter, does_not_raise()),
            pytest.param(
                "wtp",
                "invalid",
                {},
                WtpSegmenter,
                pytest.raises(ValueError),
                marks=pytest.mark.skip(reason="WtpSegmenter falls back to basic model"),
            ),
            (
                "sentence_splitter",
                "nl",
                {},
                SentenceSplitterSegmenter,
                does_not_raise(),
            ),
            ("stanza", "en", {}, StanzaSegmenter, does_not_raise()),
            ("window", None, {"window_size": 5}, WindowSegmenter, does_not_raise()),
            ("window", None, {}, WindowSegmenter, pytest.raises(TypeError)),
            (None, "en", {}, None, does_not_raise()),
            ("invalid", "en", {}, None, pytest.raises(ValueError)),
        ],
    )
    def test_segmenter(
        self, segmenter: str, language, kwargs, expected_type, expected_exception
    ):
        with expected_exception:
            _segmenter = Segmenter.segmenter(segmenter, language, **kwargs)
            if expected_type is None:
                # FIXME: this is for compatibility with Python 3.9. Python >=3.10 can use NoneType as `expected_type`
                assert _segmenter is None
            else:
                assert isinstance(_segmenter, expected_type)

    @pytest.mark.parametrize(
        "segmenter, text, expected",
        [
            (
                SentenceSplitterSegmenter("en"),
                "This is a test. This is another test.",
                ["This is a test.", "This is another test."],
            ),
            (
                WtpSegmenter("en"),
                "This is a test This is another test.",
                ["This is a test ", "This is another test."],
            ),
            (
                StanzaSegmenter("en"),
                "This is a test This is another test.",
                ["This is a test", "This is another test."],
            ),
        ],
    )
    def test_split(self, segmenter, text, expected):
        assert list(segmenter.split(text)) == expected

    def test_get_backend(self):
        assert Segmenter.get_backend() in {"cuda", "mps", "cpu"}


class TestSentenceSplitter:
    @pytest.mark.parametrize(
        "text,expected",
        [
            (
                "This is a test. This is another test.",
                [
                    Passage("This is a test.", metadata={"sentence_index": 0}),
                    Passage("This is another test.", metadata={"sentence_index": 1}),
                ],
            )
        ],
    )
    def test_passages(self, text, expected):
        assert list(SentenceSplitterSegmenter("en").passages(text)) == expected


class TestWindowSegmenter:
    @pytest.mark.parametrize(
        "text,window_size,window_overlap,expected,expected_exception",
        [
            (
                "test text",
                None,
                None,
                [Passage("test text")],
                pytest.raises(ValueError),
            ),
            (
                "test text",
                5,
                None,
                [Passage("test"), Passage("text")],
                does_not_raise(),
            ),
            ("test text", 5, 0, [Passage("test"), Passage("text")], does_not_raise()),
            (
                "test text",
                5,
                2,
                [Passage("test"), Passage("t tex"), Passage("text")],
                does_not_raise(),
            ),
        ],
    )
    def test_passages(self, text, window_size, window_overlap, expected):
        segmenter = WindowSegmenter(
            None, window_size=window_size, window_overlap=window_overlap
        )
        assert list(segmenter.passages(text)) == expected
