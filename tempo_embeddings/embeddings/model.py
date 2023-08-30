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
    def _compute_embeddings(self, passages: list["Passage"]) -> torch.Tensor:
        """Computes the embeddings for a list of passages."""

        encodings = self._tokenizer(
            [passage.text for passage in passages],
            return_tensors="pt",
            padding=True,
            truncation=True,
        ).to(self.device)

        for passage, encoding in zip(passages, encodings["input_ids"], strict=True):
            # Storing tokenizations for each passage
            passage.tokenization = encoding

        embeddings = self._model(**encodings)

        return embeddings.hidden_states[self.layer]

    def compute_embeddings(self, corpus: "Corpus") -> None:
        """Computes the embeddings for highlightings in all passages in a corpus.

        Args:
            corpus: The corpus to compute token embeddings for
        """

        if passages := corpus.passages_with_highlighting():
            # TODO: optionally split into batches
            embeddings = self._compute_embeddings(passages)

            for embedding, passage in zip(embeddings, passages):
                first_token = passage.tokenization.char_to_token(
                    passage.highlighting.start
                )
                last_token = passage.tokenization.char_to_token(
                    passage.highlighting.end - 1
                )

                if first_token == last_token:
                    token_embedding = embedding[first_token]
                else:
                    # highlighting spans across multiple tokens
                    token_embeddings = embedding[first_token : last_token + 1]
                    token_embedding = np.mean(token_embeddings, axis=0)

                passage.highlighting.token_embedding = token_embedding
        else:
            logging.error("No passages with highlighting found.")

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
            model_class.from_pretrained(model_name_or_path, output_hidden_states=True),
            AutoTokenizer.from_pretrained(tokenizer_name_or_path or model_name_or_path),
            layer=layer,
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
