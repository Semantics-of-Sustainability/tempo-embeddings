import abc
import logging
from enum import Enum
from enum import auto
from typing import TYPE_CHECKING
from typing import Iterable
import numpy as np
import torch
from accelerate import Accelerator
from numpy.typing import ArrayLike
from tqdm import tqdm
from transformers import AutoModel
from transformers import AutoModelForMaskedLM
from transformers import AutoTokenizer
from transformers import RobertaModel
from transformers import XmodModel
from umap.umap_ import UMAP
from ..text.passage import Passage


if TYPE_CHECKING:
    from ..text.corpus import Corpus


class EmbeddingsMethod(Enum):
    """Enum for the different methods to compute embeddings."""

    CLS = auto()
    """Use the embedding of the [CLS] token"""
    TOKEN = auto()
    """Use the embedding of the highlighted token"""
    MEAN = auto()
    """Use the mean of the embeddings of the highlighted tokens"""


class TransformerModelWrapper(abc.ABC):
    """A Wrapper around a transformer model."""

    def __init__(
        self,
        model,
        tokenizer,
        *,
        accelerate: bool = True,
        layer: int = -1,
        batch_size: int = 128,
        embeddings_method: EmbeddingsMethod = EmbeddingsMethod.CLS,
    ):
        """Constructor.

        Args:
            model: The transformer model to use (name or transformer.model object)
            tokenizer: The tokenizer to use (name or transformer.tokenizer object)
            accelerate: if True, use the accelerate library to speed up computations
            layer: The hidden layer in the model to use for embeddings
            batch_size: The batch size to use for computing embeddings
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
        self._batch_size = batch_size
        self._embeddings_method = embeddings_method

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
    def batch_size(self) -> int:
        return self._batch_size

    @batch_size.setter
    def batch_size(self, batch_size: int) -> None:
        self._batch_size = batch_size

    @property
    def embeddings_method(self) -> EmbeddingsMethod:
        return self._embeddings_method

    @embeddings_method.setter
    def embeddings_method(self, embeddings_method: EmbeddingsMethod) -> None:
        self._embeddings_method = embeddings_method

    @property
    def layer(self) -> int:
        return self._layer

    @layer.setter
    def layer(self, layer: int) -> None:
        self._layer = layer

    @property
    def name(self) -> str:
        return self._model.config._name_or_path  # pylint: disable=protected-access

    def _batches(self, passages: list[Passage]) -> Iterable[list[Passage]]:
        for batch_start in tqdm(
            range(0, len(passages), self.batch_size),
            desc="Embeddings",
            unit="batch",
            total=len(passages) // self.batch_size + 1,
        ):
            yield passages[batch_start : batch_start + self.batch_size]

    def _encodings(self, passages: list[Passage], store_tokenizations: bool):
        encodings = self._tokenizer(
            [passage.text for passage in passages],
            return_tensors="pt",
            padding=True,
            truncation=True,
        )
        if store_tokenizations:
            for i, passage in enumerate(passages):
                # Store tokenizations for each passage
                passage.tokenization = encodings[i]
        return encodings

    @torch.no_grad()
    def _passage_embeddings(
        self, passages: list[Passage], store_tokenizations: bool
    ) -> Iterable[torch.Tensor]:
        for batch in self._batches(passages):
            encodings = self._encodings(batch, store_tokenizations)
            embeddings = self._model(**encodings.to(self.device))
            layer_output = embeddings.hidden_states[self.layer]

            if self.embeddings_method == EmbeddingsMethod.CLS:
                passage_embeddings = layer_output[:, 0, :]
            elif self.embeddings_method == EmbeddingsMethod.TOKEN:
                passage_embeddings = torch.stack(
                    [
                        self._token_embedding(passage, passage_embedding)
                        for passage, passage_embedding in zip(batch, layer_output)
                    ]
                )
            elif self.embeddings_method == EmbeddingsMethod.MEAN:
                # TODO: test if this is the correct axis
                passage_embeddings = layer_output.mean(axis=1)
            else:
                raise RuntimeError(self.embeddings_method)

            yield passage_embeddings

    def _token_embedding(self, passage, embedding) -> torch.Tensor:
        tokenization = passage.tokenization

        first_token = tokenization.char_to_token(passage.highlighting.start)
        last_token = tokenization.char_to_token(passage.highlighting.end - 1)

        if first_token == last_token:
            token_embedding = embedding[first_token]
        else:
            # highlighting spans across multiple tokens
            token_embeddings = embedding[first_token : last_token + 1]
            token_embedding = torch.mean(token_embeddings, axis=0)
        return token_embedding

    def compute_embeddings(
        self,
        corpus: "Corpus",
        store_tokenizations: bool = True,
        umap_verbose: bool = True,
        **umap_args,
    ) -> ArrayLike:
        # TODO: add relevant UMAP arguments with reasonable defaults

        """Computes the embeddings for highlightings in all passages in a corpus.

        Args:
            corpus: The corpus to compute token embeddings for
            store_tokenizations: if True, passage tokenizations are kept in memory
            umap_verbose: if True (default), print UMAP progress
            **umap_args: other keyword arguments to the UMAP algorithm,
                see https://umap-learn.readthedocs.io/en/latest/parameters.html
        """

        embeddings: ArrayLike = np.concatenate(
            [
                tensor.cpu()
                for tensor in self._passage_embeddings(
                    corpus.passages, store_tokenizations
                )
            ],
            axis=0,
        )

        assert embeddings.shape[0] == len(
            corpus
        ), f"Embeddings shape is {embeddings.shape}, corpus length is {len(corpus)}."

        umap = UMAP(verbose=umap_verbose, **umap_args)
        return umap.fit_transform(embeddings)

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


class SentenceTransformerModelWrapper(TransformerModelWrapper):
    @TransformerModelWrapper.embeddings_method.setter
    def embeddings_method(self, embeddings_method: EmbeddingsMethod) -> None:
        if embeddings_method != EmbeddingsMethod.MEAN:
            raise ValueError(
                "SentenceTransformers do not support other embeddings methods than MEAN."
            )
        self._embeddings_method = embeddings_method

    # Mean Pooling - Take attention mask into account for correct averaging
    # TODO: use this in TransformerModelWrapper._passage_embeddings() too? How
    @staticmethod
    def mean_pooling(model_output, attention_mask):
        # First element of model_output contains all token embeddings
        token_embeddings = model_output[0]
        input_mask_expanded = (
            attention_mask.unsqueeze(-1).expand(token_embeddings.size()).float()
        )
        return torch.sum(token_embeddings * input_mask_expanded, 1) / torch.clamp(
            input_mask_expanded.sum(1), min=1e-9
        )

    @torch.no_grad()
    def _passage_embeddings(
        self, passages: list[Passage], store_tokenizations: bool
    ) -> Iterable[torch.Tensor]:
        for batch in self._batches(passages):
            encodings = self._encodings(batch, store_tokenizations)
            embeddings = self._model(**encodings.to(self.device))
            sentence_embeddings = SentenceTransformerModelWrapper.mean_pooling(
                embeddings, encodings["attention_mask"]
            )
            yield sentence_embeddings

    @classmethod
    def from_pretrained(
        cls,
        model_name_or_path: str,
        tokenizer_name_or_path: str = None,
        model_class=AutoModel,
        layer: int = -1,
        **kwargs,
    ):
        if layer != -1:
            logging.warning(
                "SentenceTransformerModelWrapper does not support different layers, ignoring."
            )
        return cls(
            model_class.from_pretrained(model_name_or_path, output_hidden_states=True),
            AutoTokenizer.from_pretrained(tokenizer_name_or_path or model_name_or_path),
            layer=layer,
            embeddings_method=EmbeddingsMethod.MEAN,
            **kwargs,
        )
