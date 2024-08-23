import platform

import pytest

from tempo_embeddings.embeddings.model import SentenceTransformerModelWrapper
from tempo_embeddings.settings import DEFAULT_LANGUAGE_MODEL


class TestSentenceTransformerModelWrapper:
    @pytest.mark.skipif(
        platform.platform().startswith("macOS"),
        reason="MacOS Github Runners don't have enough memory.",
    )
    def test_embed_corpus(self, corpus):
        model = SentenceTransformerModelWrapper.from_pretrained(DEFAULT_LANGUAGE_MODEL)
        embedding_batches = list(model.embed_corpus(corpus))
        assert len(embedding_batches) == 1
        for batch in embedding_batches:
            assert len(batch) == 5
            for embedding in batch:
                assert embedding.shape == (768,)
