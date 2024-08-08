from contextlib import nullcontext as does_not_raise

import pytest

from tempo_embeddings.text.segmenter import (
    Segmenter,
    SentenceSplitterSegmenter,
    StanzaSegmenter,
    WtpSegmenter,
)


class TestSegmenter:
    @pytest.mark.parametrize(
        "segmenter, language, expected_type, expected_exception",
        [
            ("wtp", "en", WtpSegmenter, does_not_raise()),
            ("wtp", "nl", WtpSegmenter, does_not_raise()),
            pytest.param(
                "wtp",
                "invalid",
                WtpSegmenter,
                pytest.raises(ValueError),
                marks=pytest.mark.skip(reason="WtpSegmenter falls back to basic model"),
            ),
            ("sentence_splitter", "nl", SentenceSplitterSegmenter, does_not_raise()),
            ("stanza", "en", StanzaSegmenter, does_not_raise()),
            (None, "en", None, does_not_raise()),
            ("invalid", "en", None, pytest.raises(ValueError)),
        ],
    )
    def test_segmenter(
        self, segmenter: str, language, expected_type, expected_exception
    ):
        with expected_exception:
            _segmenter = Segmenter.segmenter(segmenter, language)
            if expected_type is None:
                # FIXME: this is for compatibility with Python 3.9. Python >=3.10 can use NoneType as `expected_type`
                assert _segmenter is None
            else:
                assert isinstance(_segmenter, expected_type)

    @pytest.mark.parametrize(
        "segmenter, text, expected",
        [
            (
                Segmenter.segmenter("sentence_splitter", "en"),
                "This is a test. This is another test.",
                ["This is a test.", "This is another test."],
            ),
            (
                Segmenter.segmenter("wtp", "en"),
                "This is a test This is another test.",
                ["This is a test ", "This is another test."],
            ),
            (
                Segmenter.segmenter("stanza", "en"),
                "This is a test This is another test.",
                ["This is a test", "This is another test."],
            ),
        ],
    )
    def test_split(self, segmenter, text, expected):
        assert list(segmenter.split(text)) == expected

    def test_get_backend(self):
        assert Segmenter.get_backend() in {"cuda", "mps", "cpu"}
