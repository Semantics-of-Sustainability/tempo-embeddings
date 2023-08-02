import pytest
from tempo_embeddings.embeddings.model import RobertaModelWrapper
from tempo_embeddings.text.highlighting import Highlighting
from tempo_embeddings.text.passage import Passage


class TestHighlighting:
    # TODO: mock this
    model = RobertaModelWrapper.from_pretrained("roberta-base")

    @pytest.mark.parametrize(
        "start, end, passage, metadata_fields, max_context_length, expected",
        [
            (0, 4, Passage("test", model=model), [], 200, "<b>test</b>"),
            (
                10,
                14,
                Passage("this is a test text", model=model),
                [],
                200,
                "this is a <b>test</b> text",
            ),
            (
                10,
                14,
                Passage("this is a test text in context", model=model),
                [],
                5,
                "is a <b>test</b> text",
            ),
            (
                0,
                4,
                Passage("test", metadata={"key": "value"}, model=model),
                ["key"],
                200,
                "<b>test</b><br>{'key': 'value'}",
            ),
        ],
    )
    # pylint: disable=too-many-arguments
    def test_text(
        self, start, end, passage, metadata_fields, max_context_length, expected
    ):
        assert (
            Highlighting(start, end, passage).text(
                metadata_fields, max_context_length=max_context_length
            )
            == expected
        )
