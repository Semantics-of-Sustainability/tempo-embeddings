import pytest
from tempo_embeddings.embeddings.model import RobertaModelWrapper
from tempo_embeddings.text.highlighting import Highlighting
from tempo_embeddings.text.passage import Passage


class TestPassage:
    # TODO: mock this
    model = RobertaModelWrapper.from_pretrained("roberta-base")

    @pytest.mark.parametrize(
        "passage, metadata_fields, max_context_length, expected",
        [
            (Passage("test", highlighting=Highlighting(0, 4)), [], 200, "<b>test</b>"),
            (
                Passage("this is a test text", highlighting=Highlighting(10, 14)),
                [],
                200,
                "this is a <b>test</b> text",
            ),
            (
                Passage(
                    "this is a test text in context", highlighting=Highlighting(10, 14)
                ),
                [],
                5,
                "is a <b>test</b> text",
            ),
            (
                Passage(
                    "test", metadata={"key": "value"}, highlighting=Highlighting(0, 4)
                ),
                ["key"],
                200,
                "<b>test</b><br>{'key': 'value'}",
            ),
        ],
    )
    def test_highlighted_text(
        self, passage, metadata_fields, max_context_length, expected
    ):
        self.model.compute_passage_embeddings(passage)
        self.model.tokenize_passage(passage)

        assert (
            passage.highlighted_text(
                metadata_fields=metadata_fields, max_context_length=max_context_length
            )
            == expected
        )

    @pytest.mark.parametrize(
        "passage, metadata_fields, expected",
        [
            (Passage("test text"), [], {"text": "test text"}),
            (
                Passage("test text", metadata={"key": "value"}),
                ["key"],
                {"text": "test text", "key": "value"},
            ),
            (
                Passage("test text", highlighting=Highlighting(0, 4)),
                [],
                {"text": "<b>test</b> text"},
            ),
        ],
    )
    def test_hover_data(self, passage, metadata_fields, expected):
        if passage.highlighting is not None:
            pytest.skip("Testing with highlighted text not implemented.")
        assert passage.hover_data(metadata_fields=metadata_fields) == expected

    @pytest.mark.parametrize(
        "text,metadata,expected",
        [
            ("", None, Passage("", {})),
            ("", {}, Passage("", {})),
            ("test", None, Passage("test", {})),
            (" test ", None, Passage("test", {})),
            ("test", {"key": "value"}, Passage("test", {"key": "value"})),
        ],
    )
    def test_init(self, text, metadata, expected):
        assert Passage(text, metadata) == expected

    @pytest.mark.parametrize(
        "text,window_size,window_overlap,expected",
        [
            ("test text", None, None, [Passage("test text")]),
            ("test text", 5, None, [Passage("test"), Passage("text")]),
            ("test text", 5, 0, [Passage("test"), Passage("text")]),
            ("test text", 5, 2, [Passage("test"), Passage("t tex"), Passage("ext")]),
        ],
    )
    def test_from_text(self, text, window_size, window_overlap, expected):
        assert (
            list(
                Passage.from_text(
                    text, window_size=window_size, window_overlap=window_overlap
                )
            )
            == expected
        )

    # TODO: test with use_tokenizer=True, mock model
    @pytest.mark.parametrize(
        "passage,use_tokenizer,expected",
        [
            (Passage("test"), False, ["test"]),
            (Passage("test, token"), False, ["test", "token"]),
        ],
    )
    def test_words(self, passage, use_tokenizer, expected):
        assert list(passage.words(use_tokenizer)) == expected
