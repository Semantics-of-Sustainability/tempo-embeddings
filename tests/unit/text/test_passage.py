import platform

import pytest

from tempo_embeddings.text.highlighting import Highlighting
from tempo_embeddings.text.passage import Passage
from tempo_embeddings.text.segmenter import SentenceSplitterSegmenter
from weaviate.exceptions import WeaviateStartUpError


@pytest.fixture
def passage():
    return Passage("this is a test passage")


class TestPassage:
    @pytest.mark.parametrize(
        "term, expected",
        [("test", True), ("test passage", True), ("TEST", True), ("not", False)],
    )
    def test_contains(self, passage, term, expected):
        assert passage.contains(term) == expected

    @pytest.mark.parametrize(
        "terms, expected",
        [(None, True), (["test"], True), (["test", "not"], True), (["not"], False)],
    )
    def test_contains_any(self, passage, terms, expected):
        assert passage.contains_any(terms) == expected

    @pytest.mark.parametrize(
        "passage, metadata_fields, max_context_length, expected",
        [
            (Passage("test", highlighting=Highlighting(0, 4)), [], 200, "<b>test</b>"),
            (
                Passage("this is a test text", highlighting=Highlighting(10, 14)),
                [],
                200,
                "this is a <b>test</b> text",
            ),
            (
                Passage(
                    "this is a test text in context", highlighting=Highlighting(10, 14)
                ),
                [],
                5,
                "is a <b>test</b> text",
            ),
            (
                Passage(
                    "test", metadata={"key": "value"}, highlighting=Highlighting(0, 4)
                ),
                ["key"],
                200,
                "<b>test</b><br>{'key': 'value'}",
            ),
        ],
    )
    def test_highlighted_text(
        self, passage, metadata_fields, max_context_length, expected
    ):
        assert (
            passage.highlighted_text(
                metadata_fields=metadata_fields, max_context_length=max_context_length
            )
            == expected
        )

    @pytest.mark.parametrize(
        "passage, metadata_fields, expected",
        [
            (Passage("test text"), [], {"text": "test text"}),
            (
                Passage("test text", metadata={"key": "value"}),
                ["key"],
                {"text": "test text", "key": "value"},
            ),
        ],
    )
    def test_hover_data(self, passage, metadata_fields, expected):
        assert passage.hover_data(metadata_fields=metadata_fields) == expected

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
        "text,window_size,window_overlap,expected",
        [
            ("test text", None, None, [Passage("test text")]),
            ("test text", 5, None, [Passage("test"), Passage("text")]),
            ("test text", 5, 0, [Passage("test"), Passage("text")]),
            ("test text", 5, 2, [Passage("test"), Passage("t tex"), Passage("text")]),
        ],
    )
    def test_from_text_window(self, text, window_size, window_overlap, expected):
        assert (
            list(
                Passage.from_text(
                    text, window_size=window_size, window_overlap=window_overlap
                )
            )
            == expected
        )

    @pytest.mark.parametrize(
        "segmenter,text,expected",
        [
            (
                SentenceSplitterSegmenter("en"),
                "This is a test. This is another test.",
                [
                    Passage("This is a test.", metadata={"sentence_index": 0}),
                    Passage("This is another test.", metadata={"sentence_index": 1}),
                ],
            )
        ],
    )
    def test_from_text_segmenter(self, segmenter, text, expected):
        assert list(Passage.from_text(text, nlp_pipeline=segmenter)) == expected

    @pytest.mark.parametrize(
        "passage,expected",
        [
            (Passage("test"), ["test"]),
            (Passage("test, token"), ["test", "token"]),
        ],
    )
    def test_words(self, passage, expected):
        assert list(passage.words()) == expected

    @pytest.mark.skipif(
        int(platform.python_version_tuple()[1]) < 10,
        reason="Python 3.10+ required for this test.",
    )
    @pytest.mark.xfail(
        platform.system() == "Windows",
        raises=WeaviateStartUpError,
        reason="Weaviate Embedded not supported on Windows",
    )
    def test_from_weaviate_record(self, weaviate_db_manager_with_data):
        expected_passages = [
            Passage(
                "test",
                metadata={"provenance": "test_file"},
                highlighting=Highlighting(1, 3),
            )
        ]
        objects = (
            weaviate_db_manager_with_data._client.collections.get("TestCorpus")
            .query.fetch_objects(include_vector=True)
            .objects
        )

        for _object, expected in zip(objects, expected_passages, strict=True):
            assert Passage.from_weaviate_record(_object) == expected
