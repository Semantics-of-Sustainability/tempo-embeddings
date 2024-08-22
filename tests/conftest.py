from pathlib import Path

import numpy as np
import pytest

import weaviate
from tempo_embeddings.embeddings.weaviate_database import WeaviateDatabaseManager
from tempo_embeddings.text.corpus import Corpus
from tempo_embeddings.text.highlighting import Highlighting
from tempo_embeddings.text.passage import Passage

CWD = Path(__file__).parent.absolute()


@pytest.fixture
def mock_transformer_wrapper(mocker):
    def array_generator(shape=(1, 768)):
        yield np.random.rand(*shape)

    mock_model = mocker.Mock()
    mock_model.embed_corpus.return_value = array_generator()
    mock_model.batch_size = 1
    mock_model.name = "mock_model"

    return mock_model


@pytest.fixture
def corpus():
    return Corpus(
        [
            Passage(
                "test",
                metadata={"provenance": "test_file"},
                highlighting=Highlighting(1, 3),
                embedding=np.random.rand(768),
            )
            for _ in range(5)
        ],
        label="TestCorpus",
    )


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


CORPUS_DIR: Path = CWD / "data"
