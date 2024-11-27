import datetime
import platform
from contextlib import nullcontext as does_not_raise

import pytest

from tempo_embeddings.settings import STRICT
from tempo_embeddings.text.highlighting import Highlighting
from tempo_embeddings.text.passage import Passage
from weaviate.exceptions import WeaviateStartUpError


@pytest.fixture
def passage():
    return Passage("this is a test passage")


class TestPassage:
    @pytest.mark.parametrize(
        "passage1, passage2, expected, exception",
        [
            (Passage("test1"), Passage("test2"), Passage("test1 test2"), None),
            (
                Passage("test1", metadata={"key": "value"}),
                Passage("test2"),
                Passage("test1 test2", metadata={"key": "value"}),
                None,
            ),
            (
                Passage("test1", metadata={"key1": "value1"}),
                Passage("test2", metadata={"key2": "value2"}),
                Passage("test1 test2", metadata={"key1": "value1", "key2": "value2"}),
                None,
            ),
            (
                Passage("test1", metadata={"key1": "value1"}),
                Passage("test2", metadata={"key1": "value2"}),
                Passage("test1 test2", metadata={"key1": "value1"}),
                pytest.raises(RuntimeError),
            ),
            (
                Passage("test1", metadata={"sentence_index": 1}),
                Passage("test2", metadata={"sentence_index": 2}),
                Passage("test1 test2", metadata={"sentence_index": 1}),
                None,
            ),
        ],
    )
    def test_add(self, passage1, passage2, expected, exception):
        with exception or does_not_raise():
            assert passage1 + passage2 == expected

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
        "passage,expected",
        [
            (
                Passage("test"),
                {
                    "text": "test",
                    "ID_DB": "21daa890fa6ec86a719c9869a5b794c886f0ce348b6b465c13fee12bbc6bda43",
                    "highlight_start": None,
                    "highlight_end": None,
                },
            ),
            (
                Passage(
                    "test",
                    highlighting=Highlighting(1, 3),
                    embedding_compressed=[1.0, 2.0],
                ),
                {
                    "text": "test",
                    "ID_DB": "de5a824827cda1bf8f789c962895a724eed0f2c81168df20f52ee2402f46b15d",
                    "highlight_start": 1,
                    "highlight_end": 3,
                    "x": 1.0,
                    "y": 2.0,
                },
            ),
        ],
    )
    def test_to_dict(self, passage, expected):
        assert passage.to_dict() == expected

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
        "passage,expected",
        [
            (Passage("test"), ["test"]),
            (Passage("test, token"), ["test", "token"]),
        ],
    )
    def test_words(self, passage, expected):
        assert list(passage.words()) == expected

    def test_set_metadata(self):
        passage = Passage("test", metadata={"key": "value"})
        passage.set_metadata("test_key", "test_value")

        assert passage.metadata == {"key": "value", "test_key": "test_value"}

    @pytest.mark.xfail(
        platform.system() == "Windows",
        raises=WeaviateStartUpError,
        reason="Weaviate Embedded not supported on Windows",
    )
    def test_from_weaviate_record(self, weaviate_db_manager_with_data, test_passages):
        collection = "TestCorpus"
        objects = (
            weaviate_db_manager_with_data._client.collections.get(collection)
            .query.fetch_objects(include_vector=True)
            .objects
        )

        for _object, expected in zip(
            sorted(objects, key=lambda o: o.properties["passage"]),
            test_passages,
            **STRICT,
        ):
            expected.set_metadata("collection", collection)
            expected.set_metadata(
                "date", expected.metadata["date"].replace(tzinfo=datetime.timezone.utc)
            )
            assert (
                Passage.from_weaviate_record(_object, collection=collection) == expected
            )

    def test_from_df_row(self):
        row = {
            "text": "test text",
            "year": 2022,
            "date": "2022-01-01T00:00:00Z",
            "sentence_index": 1,
            "x": 1.0,
            "y": 2.0,
        }
        expected = Passage(
            "test text",
            metadata={
                "year": 2022,
                "date": datetime.datetime(
                    2022, 1, 1, 0, 0, 0, tzinfo=datetime.timezone.utc
                ),
                "sentence_index": 1,
            },
            embedding_compressed=[1.0, 2.0],
        )

        assert Passage.from_df_row(row, text_field="text") == expected

    @pytest.mark.parametrize(
        "passages,min_length,expected",
        [
            ([], 512, []),
            ([Passage("test")], 512, [Passage("test")]),
            ([Passage("test")], 3, [Passage("test")]),
            (
                [Passage("test1"), Passage("test2")],
                5,
                [Passage("test1"), Passage("test2")],
            ),
            ([Passage("test1"), Passage("test2")], 10, [Passage("test1 test2")]),
        ],
    )
    def test_merge(self, passages, min_length, expected):
        assert Passage.merge(passages, length=min_length) == expected

    @pytest.mark.parametrize(
        "passage, passages, length, expected, expected_remaining",
        [
            (Passage("test"), [], 5, Passage("test"), []),
            (Passage("test"), [], 3, Passage("test"), []),
            (
                Passage("test1"),
                [Passage("test2"), Passage("test3")],
                512,
                Passage("test1 test2 test3"),
                [],
            ),
            (
                Passage("test1"),
                [Passage("test2"), Passage("test3")],
                12,
                Passage("test1 test2"),
                [Passage("test3")],
            ),
        ],
    )
    def test_merge_until(self, passage, passages, length, expected, expected_remaining):
        assert passage.merge_until(passages, length=length) == expected
        assert passages == expected_remaining

    def test_model_field_names(self):
        assert list(Passage.Metadata.model_field_names()) == [
            ("year", "int"),
            ("date", "datetime"),
            ("sentence_index", "int"),
            ("origin_id", "str"),
        ]
