import platform
from contextlib import nullcontext as does_not_raise

import pytest

from tempo_embeddings.text.segmenter import Segmenter, StanzaSegmenter, WtpSegmenter


@pytest.mark.skipif(
    int(platform.python_version_tuple()[1]) < 10,
    reason="types.NoneType is not available in Python <3.10",
)
class TestSegmenter:
    from types import NoneType

    @pytest.mark.parametrize(
        "segmenter, language, expected_type, expected_exception",
        [
            ("wtp", "en", WtpSegmenter, does_not_raise()),
            ("wtp", "nl", WtpSegmenter, does_not_raise()),
            # ("wtp", "invalid", WtpSegmenter, pytest.raises(ValueError)),
            ("stanza", "en", StanzaSegmenter, does_not_raise()),
            (None, "en", NoneType, does_not_raise()),
            ("invalid", "en", NoneType, pytest.raises(ValueError)),
        ],
    )
    def test_segmenter(self, segmenter, language, expected_type, expected_exception):
        with expected_exception:
            assert isinstance(Segmenter.segmenter(segmenter, language), expected_type)


class TestWtpSegmenter:
    @pytest.fixture
    def segmenter(self):
        return WtpSegmenter(language="en")

    @pytest.mark.parametrize(
        "text, expected",
        [
            (
                "This is a test This is another test.",
                ["This is a test ", "This is another test."],
            )
        ],
    )
    def test_split(self, segmenter, text, expected):
        assert segmenter.split(text) == expected


class TestStanzaSegmenter:
    @pytest.fixture
    def segmenter(self):
        return StanzaSegmenter(language="en")

    @pytest.mark.parametrize(
        "text, expected",
        [
            (
                "This is a test This is another test.",
                ["This is a test", "This is another test."],
            )
        ],
    )
    def test_split(self, segmenter, text, expected):
        assert list(segmenter.split(text)) == expected
