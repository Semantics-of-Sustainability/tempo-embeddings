import abc
import numpy as np
import torch
from numpy.typing import ArrayLike
from transformers import AutoTokenizer
from transformers import RobertaModel
from transformers import pipeline
from transformers.tokenization_utils_base import CharSpan
from ..text.passage import Passage


class TransformerModelWrapper(abc.ABC):
    def __init__(self, model, tokenizer):
        self._model = model
        self._tokenizer = tokenizer
        self._pipeline = pipeline(
            "feature-extraction", model=self._model, tokenizer=self._tokenizer
        )

    @property
    def dimensionality(self) -> int:
        """Returns the dimensionality of the embeddings"""
        return self._model.embeddings.word_embeddings.embedding_dim

    @torch.no_grad()
    def token_embedding(self, passage: Passage, char_span: CharSpan) -> ArrayLike:
        """Returns the token embedding for the given char span in the given passage."""

        # encoding: Mapping = self.embeddings([passage.text])
        # outputs = self._model(**(encoding["input_ids"]))
        outputs = self._pipeline(passage.text)

        encodings = self._tokenizer(passage.text, return_tensors="pt")
        token_index_start = encodings.char_to_token(char_span.start)
        token_index_end = encodings.char_to_token(char_span.end - 1)

        # TODO: implement for other hidden layers but last one
        if token_index_start == token_index_end:
            # single token
            embedding = outputs[0][token_index_start]
        else:
            # multiple tokens
            embeddings = outputs[0][token_index_start : token_index_end + 1]
            embedding = np.mean(embeddings, axis=0)
        assert len(embedding) == self.dimensionality
        return embedding

    @classmethod
    @abc.abstractmethod
    def from_pretrained(cls, model_name_or_path: str):
        return NotImplemented


class RobertaModelWrapper(TransformerModelWrapper):
    @classmethod
    def from_pretrained(cls, model_name_or_path: str):
        return cls(
            RobertaModel.from_pretrained(model_name_or_path),
            AutoTokenizer.from_pretrained(model_name_or_path),
        )
