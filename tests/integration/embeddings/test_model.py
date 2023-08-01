import pytest
from tempo_embeddings.embeddings.model import RobertaModelWrapper
from tempo_embeddings.text.corpus import Corpus
from tempo_embeddings.text.highlighting import Highlighting
from tempo_embeddings.text.passage import Passage


class TestRobertaModelWrapper:
    @pytest.mark.parametrize(
        "model_name, corpus",
        [
            ("roberta-base", Corpus()),
            ("roberta-base", Corpus.from_lines(["This is a test."])),
            (
                "roberta-base",
                Corpus(
                    [Passage("This is a test.")],
                    [Highlighting(0, 4, Passage("This is a test."))],
                ),
            ),
            (
                "roberta-base",
                Corpus(
                    [Passage("This is a test.")],
                    [
                        Highlighting(0, 4, Passage("This is a test.")),
                        Highlighting(5, 7, Passage("This is a test.")),
                    ],
                ),
            ),
        ],
    )
    def test_compute_embeddings(self, model_name, corpus):
        RobertaModelWrapper.from_pretrained(model_name).compute_embeddings(corpus)
        for passage in corpus.passages:
            assert passage.embeddings is not None
