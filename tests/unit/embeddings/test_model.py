import pytest
from tempo_embeddings.embeddings.model import RobertaModelWrapper
from tempo_embeddings.text.corpus import Corpus
from tempo_embeddings.text.corpus import TokenInfo
from tempo_embeddings.text.passage import Passage


class TestRobertaModelWrapper:
    @pytest.mark.parametrize(
        "model_name, corpus",
        [
            ("roberta-base", Corpus()),
            ("roberta-base", Corpus.from_lines(["This is a test."])),
            ("roberta-base", Corpus({Passage("This is a test."): {TokenInfo(0, 4)}})),
            (
                "roberta-base",
                Corpus(
                    {Passage("This is a test."): {TokenInfo(0, 4), TokenInfo(5, 7)}}
                ),
            ),
        ],
    )
    def test_add_embeddings(self, model_name, corpus):
        RobertaModelWrapper.from_pretrained(model_name).add_embeddings(corpus)
        for token_info in corpus.token_infos:
            assert token_info.embedding is not None
