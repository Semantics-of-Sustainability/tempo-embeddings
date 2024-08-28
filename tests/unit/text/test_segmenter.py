import csv
import logging
from contextlib import nullcontext as does_not_raise
from io import StringIO

import pytest

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
        "sentence,max_sentence_length,expected",
        [
            ("This is a test.", 200, ["This is a test."]),
            ("This is; a test.", 200, ["This is; a test."]),
            ("This is; a test.", 5, ["This is;", "a test."]),
            ("This is a test.", 5, ["This is a test."]),
        ],
    )
    def test_split_sentence(self, sentence, max_sentence_length, expected):
        sentences = SentenceSplitterSegmenter(
            language="en",
            min_sentence_length=0,
            max_sentence_length=max_sentence_length,
        )._split_sentence(sentence)

        assert list(sentences) == expected

    @pytest.mark.parametrize(
        "sentences,min_sentence_length,expected",
        [
            (["This is a sentence."], 0, ["This is a sentence."]),
            (
                ["This is a sentence.", "This is another sentence."],
                0,
                ["This is a sentence.", "This is another sentence."],
            ),
            (
                ["This is a sentence.", "This is another sentence."],
                5,
                ["This is a sentence.", "This is another sentence."],
            ),
            (
                ["This is a sentence.", "This is another sentence."],
                20,
                ["This is a sentence. This is another sentence."],
            ),
            pytest.param(
                ["This is a sentence.", "Short."],
                19,
                ["This is a sentence. Short."],
                marks=pytest.mark.xfail(reason="Trailing short sentence."),
            ),
        ],
    )
    def test_merge_sentences(self, sentences, min_sentence_length, expected):
        merged = SentenceSplitterSegmenter(
            language="en", min_sentence_length=min_sentence_length
        )._merge_sentences(sentences)

        assert list(merged) == expected

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
                SentenceSplitterSegmenter("en", min_sentence_length=5),
                "This is a test. This is another test.",
                ["This is a test.", "This is another test."],
            ),
            (
                WtpSegmenter("en", min_sentence_length=5),
                "This is a test This is another test.",
                ["This is a test", "This is another test."],
            ),
            (
                StanzaSegmenter("en", min_sentence_length=5),
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
        "text,deduplicate,expected",
        [
            (
                "This is a test. This is another test.",
                True,
                [
                    Passage("This is a test.", metadata={"sentence_index": 0}),
                    Passage("This is another test.", metadata={"sentence_index": 1}),
                ],
            ),
            (
                "This is a test. This is a test.",
                True,
                [Passage("This is a test.", metadata={"sentence_index": 0})],
            ),
            (
                "This is a test. THIS is a test.",
                True,
                [Passage("This is a test.", metadata={"sentence_index": 0})],
            ),
            (
                "This is a test. This is a test.",
                False,
                [
                    Passage("This is a test.", metadata={"sentence_index": 0}),
                    Passage("This is a test.", metadata={"sentence_index": 1}),
                ],
            ),
        ],
    )
    def test_passages(self, text, deduplicate, expected):
        passages = SentenceSplitterSegmenter("en", min_sentence_length=5).passages(
            text, deduplicate=deduplicate
        )
        assert list(passages) == expected

    @pytest.mark.parametrize(
        "_csv, provenance, filter_terms, expected",
        [
            ("content\n", "test provenance", None, []),
            (
                "content\nsome text",
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
            ("content\nsome text", "test provenance", ["filter"], []),
        ],
    )
    def test_passages_from_dict_reader(self, _csv, provenance, filter_terms, expected):
        passages = SentenceSplitterSegmenter("en").passages_from_dict_reader(
            csv.DictReader(StringIO(_csv)),
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
    def test_passages(self, text, window_size, window_overlap, expected, caplog):
        with caplog.at_level(logging.WARNING):
            passages = WindowSegmenter(
                min_sentence_length=0,
                window_size=window_size,
                window_overlap=window_overlap,
            ).passages(text)

        assert caplog.record_tuples == [
            (
                "root",
                logging.WARNING,
                "WindowSegmenter does not use 'min_sentence_length' argument.",
            )
        ]

        assert list(passages) == expected
