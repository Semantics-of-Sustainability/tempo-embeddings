import abc
from functools import lru_cache
from typing import Generator
from typing import Optional
from typing import Union
import torch
from transformers import AutoTokenizer
from transformers import RobertaModel
from transformers.tokenization_utils_base import BatchEncoding
from transformers.tokenization_utils_base import CharSpan
from transformers.tokenization_utils_base import TokenSpan
from ..text.passage import Passage


class TransformerModelWrapper(abc.ABC):
    # TODO: handle model casing: can casing be extracted from model or should it be a parameter?
    def __init__(self, model_name_or_path: str) -> None:
        self._init_model(model_name_or_path)

    @abc.abstractmethod
    def _init_model(self, model_name_or_path: str):
        return NotImplemented

    @lru_cache(maxsize=1024)
    def _tokenize_text(self, text: str, **kwargs) -> list[int]:
        """Variant of the tokenize() method with caching.

        This does not work with batch input because a list is not hashable."""

        return self._tokenizer(text, **kwargs)["input_ids"]

    def tokenize(
        self,
        passages: list[Passage],
        *,
        return_tensors="pt",
        truncation=True,
        **kwargs,
    ) -> list[list[int]]:
        tokenizer_args = {
            "return_tensors": return_tensors,
            "truncation": truncation,
        } | kwargs

        texts = [passage.text for passage in passages]
        return [self._tokenize_text(texts, **tokenizer_args)]

    @torch.no_grad()
    def embeddings(self, texts: Union[str, list[str]], layer: Optional[int] = None):
        token_ids = self.tokenize(texts, return_tensors="pt")

        outputs = self._model(**token_ids)
        return (
            outputs.last_hidden_state if layer is None else outputs.hidden_states[layer]
        )

    def _token_embeddings(self, embeddings, token: str):
        pass

    def token_embeddings(self, text: str, token: str):
        # TODO
        match_index = text.find(token)
        while match_index >= 0:
            pass

    def _find_tokens(
        self, token_ids: BatchEncoding, token: str
    ) -> list[list[TokenSpan]]:
        """
        Find sequences of (sub-word) tokens that match a (word) token if merged

        Args:
            - token: a token (word) to find, exact match

        Yields: TokenSpan objects representing the tokens matching the search token
        """
        batch_size = len(token_ids["input_ids"])
        return [
            list(self._find_tokens_for_sentence(token_ids, batch_index, token))
            for batch_index, input_ids in range(batch_size)
        ]

    def _find_tokens_for_sentence(
        self, token_ids: BatchEncoding, batch_index: int, token: str
    ) -> Generator[TokenSpan, None, None]:
        for word_index, token_index in enumerate(token_ids.words(batch_index)):
            if word_index is not None:
                charspan: CharSpan = token_ids.word_to_chars(batch_index, word_index)
                # TODO: get text
                if self.text[charspan.start : charspan.end] == token:
                    yield token_ids.word_to_tokens(token_index)


class RobertaModelWrapper(TransformerModelWrapper):
    def _init_model(self, model_name_or_path: str):
        self._model = RobertaModel.from_pretrained(model_name_or_path)
        self._tokenizer = AutoTokenizer.from_pretrained(model_name_or_path)
