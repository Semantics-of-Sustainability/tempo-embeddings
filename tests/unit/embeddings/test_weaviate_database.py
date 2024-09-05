import gzip
import json
import logging
import platform

import numpy as np
import pytest

import weaviate
from tempo_embeddings.embeddings.weaviate_database import QueryBuilder, WeaviateConfigDb
from tempo_embeddings.settings import STRICT
from tempo_embeddings.text.corpus import Corpus
from tempo_embeddings.text.passage import Passage
from weaviate.classes.query import Filter
from weaviate.exceptions import WeaviateStartUpError
from weaviate.util import generate_uuid5

from ...conftest import TEST_CORPUS_SIZE


@pytest.mark.xfail(
    platform.system() == "Windows",
    raises=WeaviateStartUpError,
    reason="Weaviate Embedded not supported on Windows",
)
class TestWeaviateDatabase:
    def assert_collection(
        self,
        collection: weaviate.collections.Collection,
        expected_name: str,
        expected_count: int,
        expected_vector_shapes: dict[str, int],
    ):
        """Assert the collection has the expected properties."""

        assert collection.name == expected_name, "Unexpected collection name."

        count = collection.aggregate.over_all(total_count=True).total_count
        assert count == expected_count, f"Unexpected size for '{collection.name}'."

        obj = next(collection.iterator(include_vector=True))

        assert obj.vector.keys() == expected_vector_shapes.keys()
        for vector_name, vector in obj.vector.items():
            assert len(vector) == expected_vector_shapes[vector_name]

    def assert_dict_equals(
        self, dict1, expected, *, length_only_fields={"vector", "uuid"}
    ):
        assert dict1.keys() == expected.keys()

        for key, value in expected.items():
            assert (
                len(dict1[key]) == len(value)
                if key in length_only_fields
                else dict1[key] == value
            ), f"Mismatch for key '{key}': {dict1[key]} != {value}"

    def test_ingest(self, weaviate_db_manager, corpus):
        expected_collections = [
            {"name": "TempoEmbeddings", "count": 1, "vector_shape": {}},
            {"name": "TestCorpus", "count": 5, "vector_shape": {"default": 768}},
        ]

        weaviate_db_manager.ingest(corpus)
        for collection, expected in zip(
            weaviate_db_manager.client.collections.list_all(),
            expected_collections,
            **STRICT,
        ):
            self.assert_collection(
                weaviate_db_manager.client.collections.get(collection),
                *expected.values(),
            )

        weaviate_db_manager.model.embed_corpus.assert_called_once()

    def test_get_collection_count(self, weaviate_db_manager_with_data):
        assert (
            weaviate_db_manager_with_data.get_collection_count("TestCorpus")
            == TEST_CORPUS_SIZE
        )

    def test_get_corpus(self, weaviate_db_manager_with_data):
        corpus = weaviate_db_manager_with_data.get_corpus("TestCorpus")
        assert len(corpus.passages) == TEST_CORPUS_SIZE
        assert corpus.label == "TestCorpus"
        assert all(
            passage.metadata == {"provenance": "test_file"}
            for passage in corpus.passages
        )

    @pytest.mark.parametrize(
        "term, metadata, expected",
        [
            ("test", None, 5),
            ("test", {}, 5),
            ("1", None, 1),
            ("unknown", None, 0),
            ("test", {"provenance": "test_file"}, 5),
            ("test", {"provenance": "unknown"}, 0),
        ],
    )
    def test_doc_frequency(
        self, weaviate_db_manager_with_data, term, metadata, expected
    ):
        doc_freq = weaviate_db_manager_with_data.doc_frequency(
            term, "TestCorpus", metadata=metadata
        )
        assert doc_freq == expected

    @pytest.mark.parametrize("k", [1, 2, 5])
    def test_neighbour_passages(self, weaviate_db_manager_with_data, corpus, k):
        sub_corpus_size = 2
        sub_corpus = Corpus(corpus.passages[:sub_corpus_size])

        neighbours: list[Passage] = weaviate_db_manager_with_data.neighbour_passages(
            sub_corpus, k
        )
        assert len(neighbours) == min(k, TEST_CORPUS_SIZE - sub_corpus_size)

    def test_delete_collection(self, weaviate_db_manager_with_data):
        weaviate_db_manager_with_data.delete_collection("TestCorpus")

        weaviate_client = weaviate_db_manager_with_data.client
        assert not weaviate_client.collections.exists("TestCorpus")

        assert "TestCorpus" not in weaviate_db_manager_with_data._config

    def test_export_from_collection(self, weaviate_db_manager_with_data, tmp_path):
        expected_config_line = {
            "corpus": "TestCorpus",
            "embedder": "mock_model",
            "total_count": TEST_CORPUS_SIZE,
            "uuid": "e7f46979-b88a-5c7b-9eea-f02c0b800b3e",
        }
        expected_passage_lines = [
            {
                "highlighting": "1_3",
                "passage": f"test text {i}",
                "provenance": "test_file",
                "uuid": "5eec7ad3-4802-5c4b-82a5-3456bacec6b0",
                "vector": np.zeros(768),
            }
            for i in range(TEST_CORPUS_SIZE)
        ]

        export_file = tmp_path / "export.json.gz"
        weaviate_db_manager_with_data.export_from_collection("TestCorpus", export_file)

        with gzip.open(export_file) as f:
            lines = [json.loads(line) for line in f]

        self.assert_dict_equals(
            lines.pop(0), expected_config_line, length_only_fields={"vector"}
        )

        for line, expected_line in zip(
            sorted(lines, key=lambda line: line["passage"]),
            expected_passage_lines,
            **STRICT,
        ):
            self.assert_dict_equals(
                line, expected_line, length_only_fields={"vector", "uuid"}
            )

    def test_import_from_file(self, weaviate_db_manager_with_data, tmp_path):
        export_file = tmp_path / "export.json.gz"
        weaviate_db_manager_with_data.export_from_collection("TestCorpus", export_file)

        weaviate_db_manager_with_data.reset()

        weaviate_db_manager_with_data.import_from_file(export_file)
        client = weaviate_db_manager_with_data.client

        objects = list(
            client.collections.get("TestCorpus").iterator(include_vector=True)
        )
        assert [len(str(o.uuid)) for o in objects] == [
            len("0f83cbd4-e727-509c-a169-2d5ffba95f36")
        ] * TEST_CORPUS_SIZE
        assert sorted([o.properties for o in objects], key=lambda o: o["passage"]) == [
            {
                "provenance": "test_file",
                "passage": f"test text {str(i)}",
                "highlighting": "1_3",
            }
            for i in range(TEST_CORPUS_SIZE)
        ]
        assert [len(o.vector["default"]) for o in objects] == [768] * TEST_CORPUS_SIZE

        assert weaviate_db_manager_with_data._config["TestCorpus"] == {
            "corpus": "TestCorpus",
            "embedder": "mock_model",
            "uuid": "e7f46979-b88a-5c7b-9eea-f02c0b800b3e",
        }

    def test_import_from_file_existing(self, weaviate_db_manager_with_data, tmp_path):
        export_file = tmp_path / "export.json.gz"
        weaviate_db_manager_with_data.export_from_collection("TestCorpus", export_file)

        with pytest.raises(ValueError):
            weaviate_db_manager_with_data.import_from_file(export_file)

    def test_reset(self, weaviate_db_manager_with_data):
        client = weaviate_db_manager_with_data.client
        assert list(client.collections.list_all()) == ["TempoEmbeddings", "TestCorpus"]
        assert [
            str(o.uuid) for o in client.collections.get("TempoEmbeddings").iterator()
        ] == [generate_uuid5("TestCorpus")]

        weaviate_db_manager_with_data.reset()

        assert list(client.collections.list_all()) == ["TempoEmbeddings"]

        assert list(client.collections.get("TempoEmbeddings").iterator()) == []

        weaviate_db_manager_with_data.validate_config()

    @pytest.mark.parametrize(
        "corpus_name, expected, expected_warning",
        [
            ("TestCorpus", {"test_file"}, []),
            (
                "NonExistentCorpus",
                set(),
                ["root", logging.WARNING, "No such collection."],
            ),
        ],
    )
    def test_provenances(
        self,
        weaviate_db_manager_with_data,
        caplog,
        corpus_name,
        expected,
        expected_warning,
    ):
        with caplog.at_level(logging.WARNING):
            assert (
                set(weaviate_db_manager_with_data.provenances(corpus_name)) == expected
            )
            caplog.record_tuples == expected_warning

    def test_validate_config_missing_collection(self, weaviate_db_manager, corpus):
        weaviate_db_manager.validate_config()  # Empty collection

        weaviate_db_manager.ingest(corpus)
        weaviate_db_manager.validate_config()  # Collection with data

        weaviate_db_manager.client.collections.delete("TestCorpus")
        with pytest.raises(ValueError):
            weaviate_db_manager.validate_config()

    def test_validate_config_missing_config_entry(self, weaviate_db_manager, caplog):
        expected_error = "Collection 'TestCorpus' exists in the database but is not registered in the configuration database."

        weaviate_db_manager.validate_config()  # Empty collection
        weaviate_db_manager.client.collections.create("TestCorpus")

        with caplog.at_level(logging.WARNING):
            weaviate_db_manager.validate_config()
        assert caplog.record_tuples == [
            (
                "tempo_embeddings.embeddings.weaviate_database",
                logging.WARNING,
                expected_error,
            )
        ]


@pytest.mark.xfail(
    platform.system() == "Windows",
    raises=WeaviateStartUpError,
    reason="Weaviate Embedded not supported on Windows",
)
class TestWeaviateConfigDb:
    def test_init(self, weaviate_client):
        config = WeaviateConfigDb(weaviate_client)
        assert weaviate_client.collections.exists("TempoEmbeddings")
        assert config._exists()

        collection = weaviate_client.collections.get("TempoEmbeddings")
        assert collection.config.get(simple=True).vectorizer_config is None

        with pytest.raises(ValueError):
            config._create()
        assert config._exists()

    def test_getitem(self, weaviate_client):
        config = WeaviateConfigDb(weaviate_client)
        corpus_name = "TestCorpus"

        with pytest.raises(KeyError):
            config[corpus_name]

        config.add_corpus(corpus_name, embedder="test_model")
        assert config[corpus_name] == {
            "corpus": "TestCorpus",
            "uuid": "e7f46979-b88a-5c7b-9eea-f02c0b800b3e",
            "embedder": "test_model",
        }

    def test_setitem(self, weaviate_client):
        config = WeaviateConfigDb(weaviate_client)
        config["TestCorpus"] = {
            "uuid": "e7f46979-b88a-5c7b-9eea-f02c0b800b3e",
            "embedder": "test_model",
        }
        assert config["TestCorpus"] == {
            "corpus": "TestCorpus",
            "uuid": "e7f46979-b88a-5c7b-9eea-f02c0b800b3e",
            "embedder": "test_model",
        }

    def test_contains(self, weaviate_client):
        config = WeaviateConfigDb(weaviate_client)
        corpus_name = "TestCorpus"
        assert corpus_name not in config

        config.add_corpus(corpus_name, embedder="test_model")

        assert corpus_name in config

    def test_delete(self, weaviate_client):
        config = WeaviateConfigDb(weaviate_client)
        assert config._exists()

        config._delete()
        assert not config._exists()

    def test_add_get_corpora(self, weaviate_client):
        config = WeaviateConfigDb(weaviate_client)

        corpus_name = "TestCorpus"
        assert list(config.get_corpora()) == []

        config.add_corpus(
            corpus_name, embedder="test_model", properties={"language": "en"}
        )

        assert [
            o.properties
            for o in weaviate_client.collections.get("TempoEmbeddings").iterator()
        ] == [{"corpus": "TestCorpus", "embedder": "test_model", "language": "en"}]

        assert list(config.get_corpora()) == [corpus_name]

    def test_delete_corpus(self, weaviate_client):
        config = WeaviateConfigDb(weaviate_client)

        corpus_name = "TestCorpus"

        config.add_corpus(corpus_name, embedder="test_model")
        assert corpus_name in config

        config.delete_corpus(corpus_name)
        assert corpus_name not in config


class TestQueryBuilder:
    def assert_filter_equals(self, filter, expected):
        """Assert the filter is equal to the expected filter or filters.

        For combined filter objects, equality is checked for the sub-filters.
        """
        if hasattr(expected, "filters"):
            # Combined filter
            assert filter.filters == expected.filters
            assert filter.operator == expected.operator
        else:
            # Single filter
            assert filter == expected

    @pytest.mark.parametrize(
        "filter_words, year_from, year_to, metadata, expected",
        [
            (None,) * 5,
            (
                ["test term"],
                None,
                None,
                None,
                Filter.by_property("passage").contains_any(["test term"]),
            ),
            (
                ["test term"],
                1999,
                2000,
                None,
                Filter.all_of(
                    [
                        Filter.by_property("passage").contains_any(["test term"]),
                        Filter.by_property("year").greater_or_equal(1999),
                        Filter.by_property("year").less_or_equal(2000),
                    ]
                ),
            ),
            (
                ["test term"],
                1999,
                2000,
                {"test metadata": "test value"},
                Filter.all_of(
                    [
                        Filter.by_property("passage").contains_any(["test term"]),
                        Filter.by_property("year").greater_or_equal(1999),
                        Filter.by_property("year").less_or_equal(2000),
                        Filter.by_property("test metadata").equal("test value"),
                    ]
                ),
            ),
            (
                ["test term"],
                1999,
                2000,
                {"test metadata 1": "test value 1", "test metadata 2": "test value 2"},
                Filter.all_of(
                    [
                        Filter.by_property("passage").contains_any(["test term"]),
                        Filter.by_property("year").greater_or_equal(1999),
                        Filter.by_property("year").less_or_equal(2000),
                        Filter.by_property("test metadata 1").equal("test value 1"),
                        Filter.by_property("test metadata 2").equal("test value 2"),
                    ]
                ),
            ),
        ],
    )
    def test_build_filter(self, filter_words, year_from, year_to, metadata, expected):
        filter = QueryBuilder.build_filter(filter_words, year_from, year_to, metadata)
        self.assert_filter_equals(filter, expected)
