import shutil
from pathlib import Path

import numpy as np
import pytest
import torch

import weaviate
from tempo_embeddings.embeddings.weaviate_database import WeaviateDatabaseManager
from tempo_embeddings.text.corpus import Corpus
from tempo_embeddings.text.highlighting import Highlighting
from tempo_embeddings.text.passage import Passage

CWD = Path(__file__).parent.absolute()

TEST_CORPUS_SIZE = 5


@pytest.fixture
def mock_transformer_wrapper(mocker):
    mock_model = mocker.Mock()
    mock_model.embed_corpus.return_value = [torch.rand(TEST_CORPUS_SIZE, 768)]
    mock_model.batch_size = TEST_CORPUS_SIZE
    mock_model.name = "mock_model"

    return mock_model


@pytest.fixture
def test_passages():
    return [
        Passage(
            f"test text {str(i)}",
            # FIXME: year should be int type
            metadata={"provenance": "test_file", "year": 1950 + i},
            highlighting=Highlighting(1, 3),
            # TODO: make this deterministic for testing
            embedding=np.random.rand(768).tolist(),
        )
        for i in range(TEST_CORPUS_SIZE)
    ]


@pytest.fixture
def corpus(test_passages):
    return Corpus(test_passages, label="TestCorpus")


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

    total, used, free = shutil.disk_usage(__file__)
    if free / total < 0.1:
        pytest.xfail(reason="Less than 10% of disk space available.")
    return weaviate_db_manager


CORPUS_DIR: Path = CWD / "data"
