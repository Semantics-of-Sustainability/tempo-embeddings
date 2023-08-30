import abc
import logging
from typing import TYPE_CHECKING
import numpy as np
import torch
from accelerate import Accelerator
from transformers import AutoModelForMaskedLM
from transformers import AutoTokenizer
from transformers import RobertaModel
from transformers import XmodModel


if TYPE_CHECKING:
    from ..text.corpus import Corpus
    from ..text.passage import Passage


class TransformerModelWrapper(abc.ABC):
    """A Wrapper around a transformer model."""

    def __init__(self, model, tokenizer, accelerate: bool = True, layer: int = -1):
        """Constructor.

        Args:
            model: The transformer model to use (name or transformer.model object)
            tokenizer: The tokenizer to use (name or transformer.tokenizer object)
        """
        if not model.config.output_hidden_states:
            raise ValueError(
                "Model must output hidden states. "
                "Please set output_hidden_states=True when initializing the model."
            )

        if abs(layer) > model.config.num_hidden_layers:
            raise ValueError(
                f"Layer {layer} does not exist. "
                f"Model only has {model.config.num_hidden_layers} layers."
            )

        self._model = model
        self._tokenizer = tokenizer
        self._layer = layer

        if accelerate:
            accelerator = Accelerator()
            accelerator.prepare(self._model)

    @property
    def dimensionality(self) -> int:
        """Returns the dimensionality of the embeddings"""
        return self._model.embeddings.word_embeddings.embedding_dim

    @property
    def device(self) -> torch.device:
        return self._model.device

    @property
    def layer(self) -> int:
        return self._layer

    @layer.setter
    def layer(self, layer: int) -> None:
        self._layer = layer

    @property
    def name(self) -> str:
        return self._model.config._name_or_path  # pylint: disable=protected-access

    @torch.no_grad()
    def compute_passage_embeddings(self, passage: "Passage") -> None:
        """Computes the embeddings for a passage in-place."""

        if passage.embedding:
            raise ValueError("Passage already has embeddings")

        encoding = self._tokenizer(passage.text, return_tensors="pt").to(self.device)
        embeddings = self._model(**encoding).hidden_states[self.layer]
        passage.embedding = embeddings[0]

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
        encodings = self._tokenizer(
            [passage.text for passage in passages],
            return_tensors="pt",
            padding=True,
            truncation=True,
        ).to(self.device)

        embeddings = self._model(**encodings).hidden_states[self.layer]

        for passage, embedding in zip(passages, embeddings, strict=True):
            passage.embedding = embedding[0]

    def tokenize(self, corpus: "Corpus") -> None:
        """Tokenizes all passages in a corpus.

        Args:
            corpus: The corpus to tokenize
        """

        passages = corpus.passages_untokenized()
        if passages:
            tokenizations = self._tokenizer(
                [passage.text for passage in passages],
                return_tensors="pt",
                padding=True,
                truncation=True,
            )
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

    @classmethod
    def from_pretrained(
        cls,
        model_name_or_path: str,
        tokenizer_name_or_path: str = None,
        model_class=AutoModelForMaskedLM,
        layer: int = -1,
        **kwargs,
    ):
        return cls(
            model_class.from_pretrained(
                model_name_or_path, output_hidden_states=True, layer=layer
            ),
            AutoTokenizer.from_pretrained(tokenizer_name_or_path or model_name_or_path),
            **kwargs,
        )


class RobertaModelWrapper(TransformerModelWrapper):
    # TODO: Is this subclass necessary, or can I use AutoModelForMaskedLM?
    @classmethod
    def from_pretrained(
        cls,
        model_name_or_path: str,
        tokenizer_name_or_path: str = None,
        model_class=RobertaModel,
        layer: int = -1,
        **kwargs,
    ):
        # TODO: this should return a RobertaModelWrapper object
        return TransformerModelWrapper.from_pretrained(
            model_name_or_path, tokenizer_name_or_path, model_class, layer, **kwargs
        )


class XModModelWrapper(TransformerModelWrapper):
    @classmethod
    def from_pretrained(
        cls,
        model_name_or_path: str,
        tokenizer_name_or_path: str = "xlm-roberta-base",
        model_class=XmodModel,
        layer: int = -1,
        *,
        default_language: str = "en_XX",
        **kwargs,
    ):
        # TODO: this should return a XModModelWrapper object
        model = TransformerModelWrapper.from_pretrained(
            model_name_or_path, tokenizer_name_or_path, model_class, layer, **kwargs
        )

        _model = model._model  # pylint: disable=protected-access
        if default_language in _model.config.languages:
            _model.set_default_language(default_language)
        else:
            raise ValueError(
                f"Default language '{default_language}' not in model languages. "
                f"Valid languages are: {str(_model.config.languages)}"
            )

        return model
