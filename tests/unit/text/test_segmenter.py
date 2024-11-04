import csv
from contextlib import nullcontext as does_not_raise
from io import StringIO

import pytest
from pydantic import ValidationError

from tempo_embeddings.text.highlighting import Highlighting
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
                "This is a test. This is another test.",
                ["This is a test.", "This is another test."],
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
            ),
            (
                "This is a test. This is a test.",
                [
                    Passage("This is a test.", metadata={"sentence_index": 0}),
                    Passage("This is a test.", metadata={"sentence_index": 1}),
                ],
            ),
        ],
    )
    def test_passages(self, text, expected):
        passages = SentenceSplitterSegmenter("en").passages(text)
        assert list(passages) == expected

    @pytest.mark.parametrize(
        "text,metadata,strict,expected,exception",
        [
            ("text", {}, True, [Passage("text", metadata={"sentence_index": 0})], None),
            (
                "text",
                {"date": "01-05-1889"},
                True,
                [Passage("text", metadata={"date": "01-05-1889", "sentence_index": 0})],
                None,
            ),
            (
                "text",
                {"date": "01-05-1889"},
                False,
                [Passage("text", metadata={"date": "01-05-1889", "sentence_index": 0})],
                None,
            ),
            ("text", {"date": "99-05-1889"}, True, [], pytest.raises(ValidationError)),
            ("text", {"date": "99-05-1889"}, False, [], None),
        ],
    )
    def test_passages_metadata_strict(
        self, text, metadata, strict, expected, exception
    ):
        with exception or does_not_raise():
            passages = SentenceSplitterSegmenter("en").passages(
                text, metadata=metadata, strict=strict
            )
            assert list(passages) == expected

    @pytest.mark.parametrize(
        "_csv, length, provenance, filter_terms, expected",
        [
            ("content\n", 512, "test provenance", None, []),
            (
                "content\nsome text",
                512,
                "test provenance",
                None,
                [
                    Passage(
                        "some text",
                        {"provenance": "test provenance", "sentence_index": 0},
                    )
                ],
            ),
            (
                "content\nsome filter text",
                512,
                "test provenance",
                ["filter"],
                [
                    Passage(
                        "some filter text",
                        {"provenance": "test provenance", "sentence_index": 0},
                        Highlighting(start=5, end=11),
                    )
                ],
            ),
            ("content\nsome text", 512, "test provenance", ["filter"], []),
            (
                "content\nThis is the first sentence. This is the second sentence.\n",
                512,
                "test provenance",
                None,
                [
                    Passage(
                        "This is the first sentence. This is the second sentence.",
                        {"provenance": "test provenance", "sentence_index": 0},
                    )
                ],
            ),
            (
                "content\nThis is the first sentence. This is the second sentence.\n",
                10,
                "test provenance",
                None,
                [
                    Passage(
                        "This is the first sentence.",
                        {"provenance": "test provenance", "sentence_index": 0},
                    ),
                    Passage(
                        "This is the second sentence.",
                        {"provenance": "test provenance", "sentence_index": 1},
                    ),
                ],
            ),
        ],
    )
    def test_passages_from_dict_reader(
        self, _csv, length, provenance, filter_terms, expected
    ):
        passages = SentenceSplitterSegmenter("en").passages_from_dict_reader(
            csv.DictReader(StringIO(_csv)),
            length=length,
            provenance=provenance,
            text_columns=["content"],
            filter_terms=filter_terms,
        )

        assert list(passages) == expected


class TestWindowSegmenter:
    @pytest.mark.parametrize(
        "text,window_size,window_overlap,expected",
        [
            ("test text", None, None, [Passage("test text", {"sentence_index": 0})]),
            (
                "test text",
                5,
                None,
                [
                    Passage("test", {"sentence_index": 0}),
                    Passage("text", {"sentence_index": 1}),
                ],
            ),
            (
                "test text",
                5,
                0,
                [
                    Passage("test", {"sentence_index": 0}),
                    Passage("text", {"sentence_index": 1}),
                ],
            ),
            (
                "test text",
                5,
                2,
                [
                    Passage("test", {"sentence_index": 0}),
                    Passage("t tex", {"sentence_index": 1}),
                    Passage("text", {"sentence_index": 2}),
                ],
            ),
        ],
    )
    def test_passages(self, text, window_size, window_overlap, expected):
        passages = WindowSegmenter(
            window_size=window_size, window_overlap=window_overlap
        ).passages(text)

        assert list(passages) == expected
