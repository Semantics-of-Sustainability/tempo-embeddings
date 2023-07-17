import abc
import logging
import numpy as np
import torch
from numpy.typing import ArrayLike
from tqdm import tqdm
from transformers import AutoTokenizer
from transformers import RobertaModel
from transformers import pipeline
from ..text.corpus import Corpus
from ..text.corpus import TokenInfo
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

    @property
    def name(self) -> str:
        return self._model.config._name_or_path  # pylint: disable=protected-access

    @torch.no_grad()
    def _batch_embeddings(self, passages: list[Passage]) -> ArrayLike:
        # TODO: use this method in token_embedding()
        """Returns the embeddings for the given passages."""
        encodings = self._tokenizer(
            [passage.text for passage in passages], return_tensors="pt"
        )
        outputs = self._model(**(encodings["input_ids"]))
        return outputs

    @torch.no_grad()
    def token_embedding(self, passage: Passage, token_info: TokenInfo) -> ArrayLike:
        """Returns the token embedding for the given char span in the given passage."""

        if token_info.embedding is not None:
            raise ValueError("TokenInfo already has an embedding")

        # encoding: Mapping = self.embeddings([passage.text])
        # outputs = self._model(**(encoding["input_ids"]))

        outputs = self._pipeline(passage.text)
        encodings = self._tokenizer(passage.text, return_tensors="pt")

        first_token = encodings.char_to_token(token_info.start)
        last_token = encodings.char_to_token(token_info.end - 1)

        # TODO: implement for other hidden layers but last one
        if first_token == last_token:
            # single token
            embedding = outputs[0][first_token]
        else:
            # multiple tokens
            embeddings = outputs[0][first_token : last_token + 1]
            embedding = np.mean(embeddings, axis=0)
        assert (
            len(embedding) == self.dimensionality
        ), f"Wrong embedding dimensionality: {len(embedding)} != {self.dimensionality}"
        return embedding

    def compute_embeddings(self, corpus: Corpus, overwrite: bool = False):
        """Adds the embeddings for the given corpus in place."""
        if corpus.has_embeddings() and not overwrite:
            raise ValueError("Corpus already has embeddings")

        if not corpus.token_infos:
            logging.warning("Corpus does not have any token infos")

        for passage, token_infos in tqdm(
            corpus.passages.items(), unit="passage", desc="Computing embeddings"
        ):
            for token_info in token_infos:
                token_info.embedding = self.token_embedding(passage, token_info)

        corpus.embeddings_model_name = self.name

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
