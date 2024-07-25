import pytest
from tempo_embeddings.embeddings.weaviate_database import WeaviateDatabaseManager


@pytest.fixture
def mock_weaviate_database_manager(mocker, tmp_path):
    mock_client = mocker.Mock()

    yield WeaviateDatabaseManager(
        db_path=tmp_path / "weaviate",
        embedder_config={"type": "default", "model": mocker.Mock()},
        client=mock_client,
    )


class TestWeaviateDatabase:
    def test_connect(self, mock_weaviate_database_manager):
        assert mock_weaviate_database_manager.connect()
        mock_weaviate_database_manager.client.is_ready.assert_called()

    def test_get_collection_count(self, mock_weaviate_database_manager):
        collection_name = "test"
        mock_weaviate_database_manager.get_collection_count(collection_name)

        mock_weaviate_database_manager.client.collections.get.assert_called_once_with(
            collection_name
        )
