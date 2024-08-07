from pathlib import Path

import numpy as np
import pytest

from tempo_embeddings.text.corpus import Corpus
from tempo_embeddings.text.passage import Passage
from tempo_embeddings.text.segmenter import StanzaSegmenter, WtpSegmenter

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
        [Passage("test", metadata={"provenance": "test_file"})], label="TestCorpus"
    )


@pytest.fixture
def wtp_segmenter():
    return WtpSegmenter(language="en")


@pytest.fixture
def stanza_segmenter():
    return StanzaSegmenter(language="en")


CORPUS_DIR: Path = CWD / "data"
