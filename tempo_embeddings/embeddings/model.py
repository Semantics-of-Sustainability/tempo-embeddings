import abc
import logging
from typing import TYPE_CHECKING
import numpy as np
import torch
from transformers import AutoTokenizer
from transformers import RobertaModel
from transformers import pipeline


if TYPE_CHECKING:
    from ..text.corpus import Corpus
    from ..text.highlighting import Highlighting
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
    def compute_passage_embeddings(self, passage: "Passage") -> None:
        """Adds the embeddings for the given passage."""
        if passage.has_embeddings():
            raise ValueError("Passage already has embeddings")

        embeddings = self._pipeline([passage.text])
        passage.embeddings = embeddings[0][0]

    @torch.no_grad()
    def compute_embeddings(self, corpus: "Corpus", overwrite: bool = False):
        """Adds the embeddings for all passages in the given corpus."""

        if corpus.has_embeddings() and not overwrite:
            raise ValueError("Corpus already has embeddings")

        if not corpus.highlightings:
            logging.warning("Corpus does not have any token infos")

        # TODO: implement for other hidden layers but last one

        passages = corpus.passages_unembeddened()
        embeddings = self._pipeline([passage.text for passage in passages])
        for passage, embedding in zip(passages, embeddings, strict=True):
            passage.embeddings = embedding[0]

        self.tokenize(corpus)

    def tokenize_passage(self, passage: "Passage") -> None:
        passage.tokenization = self._tokenize([passage.text])[0]

    def tokenize(self, corpus: "Corpus") -> None:
        passages = corpus.passages_untokenized()
        if passages:
            tokenizations = self._tokenize([passage.text for passage in passages])
            for i, passage in enumerate(passages):
                if passage.tokenization is not None:
                    raise ValueError(f"Passage {passage} already has a tokenization")
                passage.tokenization = tokenizations[i]

    def compute_token_embedding(
        self, passage: "Passage", highlighting: "Highlighting"
    ) -> None:
        """Returns the token embedding for the given char span in the given passage."""

        if highlighting.token_embedding is not None:
            raise ValueError(
                f"Highlighting already has a token embedding: {highlighting}"
            )
        if passage.embeddings is None:
            self._model.compute_passage_embeddings(passage)

        first_token, last_token = passage.token_span(
            highlighting.start, highlighting.end
        )

        if first_token == last_token:
            token_embedding = passage.embeddings[first_token]
        else:
            # highlighting spans multiple tokens
            token_embeddings = passage.embeddings[first_token : last_token + 1]
            token_embedding = np.mean(token_embeddings, axis=0)

        highlighting.token_embedding = token_embedding

    @torch.no_grad()
    def _tokenize(self, texts):
        return self._tokenizer(texts)

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
