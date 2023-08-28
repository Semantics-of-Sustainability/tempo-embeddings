import abc
import logging
from typing import TYPE_CHECKING
import numpy as np
import torch
from transformers import AutoModelForMaskedLM
from transformers import AutoTokenizer
from transformers import RobertaModel
from transformers import XmodModel
from transformers import pipeline


if TYPE_CHECKING:
    from ..text.corpus import Corpus
    from ..text.passage import Passage


class TransformerModelWrapper(abc.ABC):
    """A Wrapper around a transformer model."""

    _MODEL_CLASS = AutoModelForMaskedLM
    """Overwrite in subclasses for other model types."""
    _TOKENIZER_CLASS = AutoTokenizer
    """Overwrite in subclasses for other tokenizer types."""

    def __init__(self, model, tokenizer):
        """Constructor.

        Args:
            model: The transformer model to use (name or transformer.model object)
            tokenizer: The tokenizer to use (name or transformer.tokenizer object)
        """
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
        """Computes the embeddings for a passage in-place."""

        if passage.embedding:
            raise ValueError("Passage already has embeddings")

        embeddings = self._pipeline([passage.text])
        passage.embedding = embeddings[0][0]

    @torch.no_grad()
    def compute_embeddings(self, corpus: "Corpus"):
        """Computes the embeddings for all passages in a corpus."""

        if corpus.has_embeddings(validate=False):
            logging.warning("Corpus already has embeddings")

        if not corpus.highlightings:
            logging.warning("Corpus does not have any highlighted tokens.")

        # TODO: implement for other hidden layers but last one
        # https://github.com/Semantics-of-Sustainability/tempo-embeddings/issues/15

        passages = corpus.passages_unembeddened()
        embeddings = self._pipeline([passage.text for passage in passages])
        for passage, embedding in zip(passages, embeddings, strict=True):
            passage.embedding = embedding[0]

    def tokenize_passage(self, passage: "Passage") -> None:
        """Tokenizes a passage in-place.

        Args:
            passage: The passage to tokenize
        """

        passage.tokenization = self._tokenize([passage.text])[0]

    def tokenize(self, corpus: "Corpus") -> None:
        """Tokenizes all passages in a corpus.

        Args:
            corpus: The corpus to tokenize
        """

        passages = corpus.passages_untokenized()
        if passages:
            tokenizations = self._tokenize([passage.text for passage in passages])
            for i, passage in enumerate(passages):
                if passage.tokenization is not None:
                    raise ValueError(f"Passage {passage} already has a tokenization")
                passage.tokenization = tokenizations[i]

                if passage.highlighting:
                    self.compute_token_embedding(passage)

    def compute_token_embeddings(self, corpus: "Corpus") -> None:
        """Computes the embeddings for highlightings in all passages in a corpus.

        Args:
            corpus: The corpus to compute token embeddings for
        """

        for passage in corpus.passages:
            if passage.highlighting and passage.highlighting.token_embedding is None:
                self.compute_token_embedding(passage)

    def compute_token_embedding(self, passage: "Passage") -> None:
        """Computes the token embedding for the highlighting in a passage.

        Args:
            passage: The passage to compute the token embedding for
        """

        if passage.highlighting is None:
            raise ValueError(f"Passage {passage} does not have a highlighting")
        if passage.highlighting.token_embedding is not None:
            raise ValueError(
                f"Highlighting already has a token embedding: {passage.highlighting}"
            )
        if passage.embedding is None:
            raise ValueError("Passage does not have embeddings")

        first_token = passage.tokenization.char_to_token(passage.highlighting.start)
        last_token = passage.tokenization.char_to_token(passage.highlighting.end - 1)

        if first_token == last_token:
            token_embedding = passage.embedding[first_token]
        else:
            # highlighting spans multiple tokens
            token_embeddings = passage.embedding[first_token : last_token + 1]
            token_embedding = np.mean(token_embeddings, axis=0)

        passage.highlighting.token_embedding = token_embedding

    @torch.no_grad()
    def _tokenize(self, texts):
        return self._tokenizer(texts)

    @classmethod
    def from_pretrained(cls, model_name_or_path: str):
        return cls(
            cls._MODEL_CLASS.from_pretrained(model_name_or_path),
            cls._TOKENIZER_CLASS.from_pretrained(model_name_or_path),
        )


class RobertaModelWrapper(TransformerModelWrapper):
    _MODEL_CLASS = RobertaModel


class XModModelWrapper(TransformerModelWrapper):
    _MODEL_CLASS = XmodModel

    @classmethod
    def from_pretrained(
        cls,
        model_name_or_path: str,
        default_language: str = "en_XX",
        tokenizer_name_or_path: str = "xlm-roberta-base",
    ):
        model = XmodModel.from_pretrained(model_name_or_path)
        model.set_default_language(default_language)

        tokenizer = AutoTokenizer.from_pretrained(tokenizer_name_or_path)

        return cls(model, tokenizer)
