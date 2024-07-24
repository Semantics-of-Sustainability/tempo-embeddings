import pytest
import weaviate
from tempo_embeddings.embeddings.weaviate_database import WeaviateDatabaseManager


@pytest.fixture
def mock_weaviate_database_manager(mocker, tmp_path):
    ### Mocking weaviate.connect_to_local context manager
    # This should be unnecessary if we use a persistent weviate.Client object
    mocker.patch("weaviate.connect_to_local").return_value.is_ready = True

    yield WeaviateDatabaseManager(
        db_path=tmp_path / "weaviate",
        embedder_config={"type": "default", "model": mocker.MagicMock()},
    )

    weaviate.connect_to_local.assert_called_once()


class TestWeaviateDatabase:
    def test_connect(self, mock_weaviate_database_manager):
        assert mock_weaviate_database_manager.connect()

    @pytest.mark.skip(reason="mock not implemented")
    def test_get_collection_count(self, mock_weaviate_database_manager):
        assert mock_weaviate_database_manager.get_collection_count("test") == 0
