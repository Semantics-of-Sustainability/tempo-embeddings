import logging
import platform
from contextlib import nullcontext as does_not_raise

import pytest

import weaviate
from tempo_embeddings.embeddings.weaviate_database import (
    WeaviateConfigDb,
    WeaviateDatabaseManager,
)
from weaviate.exceptions import WeaviateStartUpError


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
    def test_ingest(self, weaviate_db_manager, corpus):
        weaviate_db_manager.ingest(corpus)

        weaviate_client = weaviate_db_manager.client

        expected_collections = ["TestCorpus", "tempo_embeddings"]
        for collection in expected_collections:
            assert weaviate_client.collections.exists(
                collection
            ), f"Collection '{collection}' has not been created."

            assert (
                weaviate_client.collections.get(collection)
                .aggregate.over_all(total_count=True)
                .total_count
                == 1
            )

        weaviate_db_manager.model.embed_corpus.assert_called_once()

    def test_get_collection_count(self, weaviate_db_manager_with_data):
        assert weaviate_db_manager_with_data.get_collection_count("TestCorpus") == 1

    def test_delete_collection(self, weaviate_db_manager_with_data):
        weaviate_db_manager_with_data.delete_collection("TestCorpus")

        weaviate_client = weaviate_db_manager_with_data.client
        assert not weaviate_client.collections.exists("TestCorpus")

        assert "TestCorpus" not in weaviate_db_manager_with_data._config

    @pytest.mark.parametrize(
        "corpus_name, expected, exception",
        [
            ("TestCorpus", ["test_file"], does_not_raise()),
            ("NonExistentCorpus", [], pytest.raises(ValueError)),
        ],
    )
    def test_filenames(
        self, weaviate_db_manager_with_data, corpus_name, expected, exception
    ):
        with exception:
            assert (
                list(weaviate_db_manager_with_data.filenames(corpus_name)) == expected
            )

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
        assert caplog.record_tuples == [("root", logging.WARNING, expected_error)]


@pytest.mark.xfail(
    platform.system() == "Windows",
    raises=WeaviateStartUpError,
    reason="Weaviate Embedded not supported on Windows",
)
class TestWeaviateConfigDb:
    @pytest.mark.parametrize("create", [True, False])
    def test_init(self, weaviate_client, create):
        config = WeaviateConfigDb(weaviate_client, create=create)
        assert config._exists() == create

        with pytest.raises(ValueError) if create else does_not_raise():
            config._create()
        assert config._exists()

    def test_contains(self, weaviate_client):
        config = WeaviateConfigDb(weaviate_client)
        corpus_name = "TestCorpus"
        assert corpus_name not in config

        config.add_corpus(corpus_name, "test_model")

        assert corpus_name in config

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
