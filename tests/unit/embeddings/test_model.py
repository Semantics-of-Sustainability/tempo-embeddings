import platform

import pytest

from tempo_embeddings.embeddings.model import (
    RobertaModelWrapper,
    SentenceTransformerModelWrapper,
    TransformerModelWrapper,
    XModModelWrapper,
)
from tempo_embeddings.settings import DEFAULT_LANGUAGE_MODEL

from ...conftest import IN_GITHUB_ACTIONS


@pytest.mark.skipif(
    platform.platform().startswith("macOS") and IN_GITHUB_ACTIONS,
    reason="MacOS Github Runners don't have enough memory.",
)
class TestSentenceTransformerModelWrapper:
    def test_embed_corpus(self, corpus):
        model = SentenceTransformerModelWrapper.from_pretrained(DEFAULT_LANGUAGE_MODEL)
        embedding_batches = list(model.embed_corpus(corpus))
        assert len(embedding_batches) == 1
        for batch in embedding_batches:
            assert len(batch) == 5
            for embedding in batch:
                assert embedding.shape == (768,)

    @pytest.mark.parametrize(
        "model_name, expected_model_class",
        [
            ("GroNLP/bert-base-dutch-cased", TransformerModelWrapper),
            ("FacebookAI/roberta-base", RobertaModelWrapper),
            (
                "NetherlandsForensicInstitute/robbert-2022-dutch-sentence-transformers",
                SentenceTransformerModelWrapper,
            ),
            ("facebook/xmod-base", XModModelWrapper),
        ],
    )
    def test_from_model_name(self, model_name, expected_model_class):
        """This will download and load the model defined in `model_name`."""
        assert isinstance(
            TransformerModelWrapper.from_model_name(model_name), expected_model_class
        )
