import logging
import platform

import pytest

import weaviate
from tempo_embeddings.embeddings.weaviate_database import (
    WeaviateConfigDb,
    WeaviateDatabaseManager,
)
from weaviate.exceptions import WeaviateStartUpError
from weaviate.util import generate_uuid5


@pytest.fixture
def weaviate_client(tmp_path):
    client = weaviate.connect_to_embedded(persistence_data_path=tmp_path)
    yield client
    client.close()


@pytest.fixture
def weaviate_db_manager(mock_transformer_wrapper, weaviate_client):
    return WeaviateDatabaseManager(
        model=mock_transformer_wrapper, client=weaviate_client
    )


@pytest.fixture
def weaviate_db_manager_with_data(weaviate_db_manager, corpus):
    weaviate_db_manager.ingest(corpus)
    return weaviate_db_manager


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

    @pytest.mark.skipif(
        int(platform.python_version_tuple()[1]) < 10,
        reason="Python 3.10+ required for this test.",
    )
    def test_ingest(self, weaviate_db_manager, corpus):
        expected_collections = [
            {"name": "TempoEmbeddings", "count": 1, "vector_shape": {}},
            {"name": "TestCorpus", "count": 1, "vector_shape": {"default": 768}},
        ]

        weaviate_db_manager.ingest(corpus)
        for collection, expected in zip(
            weaviate_db_manager.client.collections.list_all(),
            expected_collections,
            strict=True,
        ):
            self.assert_collection(
                weaviate_db_manager.client.collections.get(collection),
                *expected.values(),
            )

        weaviate_db_manager.model.embed_corpus.assert_called_once()

    def test_get_collection_count(self, weaviate_db_manager_with_data):
        assert weaviate_db_manager_with_data.get_collection_count("TestCorpus") == 1

    def test_delete_collection(self, weaviate_db_manager_with_data):
        weaviate_db_manager_with_data.delete_collection("TestCorpus")

        weaviate_client = weaviate_db_manager_with_data.client
        assert not weaviate_client.collections.exists("TestCorpus")

        assert "TestCorpus" not in weaviate_db_manager_with_data._config

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

    def test_contains(self, weaviate_client):
        config = WeaviateConfigDb(weaviate_client)
        corpus_name = "TestCorpus"
        assert corpus_name not in config

        config.add_corpus(corpus_name, "test_model")

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

        config.add_corpus(corpus_name, "test_model")

        assert list(config.get_corpora()) == [corpus_name]

    def test_delete_corpus(self, weaviate_client):
        config = WeaviateConfigDb(weaviate_client)

        corpus_name = "TestCorpus"

        config.add_corpus(corpus_name, "test_model")
        assert corpus_name in config

        config.delete_corpus(corpus_name)
        assert corpus_name not in config
