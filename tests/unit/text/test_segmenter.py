from contextlib import nullcontext as does_not_raise

import pytest

from tempo_embeddings.text.segmenter import Segmenter, StanzaSegmenter, WtpSegmenter


class TestSegmenter:
    @pytest.mark.parametrize(
        "segmenter, language, expected_type, expected_exception",
        [
            ("wtp", "en", WtpSegmenter, does_not_raise()),
            ("wtp", "nl", WtpSegmenter, does_not_raise()),
            # ("wtp", "invalid", WtpSegmenter, pytest.raises(ValueError)),
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

    def test_get_backend(self):
        assert Segmenter.get_backend() in {"cuda", "mps", "cpu"}


class TestWtpSegmenter:
    @pytest.mark.parametrize(
        "text, expected",
        [
            (
                "This is a test This is another test.",
                ["This is a test ", "This is another test."],
            )
        ],
    )
    def test_split(self, wtp_segmenter, text, expected):
        assert wtp_segmenter.split(text) == expected


class TestStanzaSegmenter:
    @pytest.mark.parametrize(
        "text, expected",
        [
            (
                "This is a test This is another test.",
                ["This is a test", "This is another test."],
            )
        ],
    )
    def test_split(self, stanza_segmenter, text, expected):
        assert list(stanza_segmenter.split(text)) == expected
