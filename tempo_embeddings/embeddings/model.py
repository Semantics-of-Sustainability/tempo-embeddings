import abc
import logging
from enum import Enum, auto
from typing import Iterable, Optional

import torch
from accelerate import Accelerator
from huggingface_hub import HfApi, ModelInfo
from tokenizers import Encoding
from transformers import AutoModel, AutoModelForMaskedLM, AutoTokenizer

from ..text.corpus import Corpus
from ..text.passage import Passage


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

    @staticmethod
    def _get_token_spans(encoding: Encoding) -> list[tuple[int, int]]:
        char_spans = [
            encoding.word_to_chars(i) for i in encoding.word_ids if i is not None
        ]
        return sorted(set(char_spans))

    @staticmethod
    def _get_char2tokens(passage: Passage, encoding: Encoding) -> list[int]:
        return [encoding.char_to_token(i) for i in range(len(passage.text))]

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
                passage.full_word_spans = self._get_token_spans(encodings[i])
                passage.char2tokens = self._get_char2tokens(passage, encodings[i])
        return encodings

    @torch.no_grad()
    def embed_corpus(
        self,
        corpus: Corpus,
        store_tokenizations: bool,
        batch_size: Optional[int] = None,
    ) -> Iterable[torch.Tensor]:
        batch_size = batch_size or self.batch_size
        for batch in corpus.batches(batch_size):
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

    @staticmethod
    def _token_embedding(passage: Passage, embedding) -> torch.Tensor:
        tokenization = passage.char2tokens
        first_token = tokenization[passage.highlighting.start]
        last_token = tokenization[passage.highlighting.end - 1]

        if first_token == last_token:
            token_embedding = embedding[first_token]
        else:
            # highlighting spans across multiple tokens
            token_embeddings = embedding[first_token : last_token + 1]
            token_embedding = torch.mean(token_embeddings, axis=0)
        return token_embedding

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

    @classmethod
    def from_model_name(cls, model_name_or_path: str, **kwargs):
        """Look up model class on Hugging Face model hub and return a model wrapper.

        Args:
            model_name_or_path: The name or path of the model to load
            **kwargs: Additional keyword arguments to pass to the model wrapper
        Returns:
            A TransformerModelWrapper or subclass instance
        """

        model_info: ModelInfo = HfApi().model_info(model_name_or_path)

        if model_info.library_name == "sentence-transformers":
            logging.info("Using SentenceTransformerModelWrapper")
            cls = SentenceTransformerModelWrapper
        else:
            logging.info("Using default TransformerModelWrapper")

        return cls.from_pretrained(model_name_or_path, **kwargs)


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
    def embed_corpus(
        self,
        corpus: Corpus,
        store_tokenizations: bool = False,
        batch_size: Optional[int] = None,
    ) -> Iterable[torch.Tensor]:
        batch_size = batch_size or self.batch_size
        for batch in corpus.batches(batch_size):
            encodings = self._encodings(batch, store_tokenizations)
            embeddings = self._model(**encodings.to(self.device))
            sentence_embeddings = SentenceTransformerModelWrapper.mean_pooling(
                embeddings, encodings["attention_mask"]
            )
            yield sentence_embeddings

    @torch.no_grad()
    def embed_passage(
        self, passage: Passage, store_tokenizations: bool = False
    ) -> Iterable[torch.Tensor]:
        encodings = self._encodings([passage], store_tokenizations)
        embeddings = self._model(**encodings.to(self.device))
        sentence_embeddings = SentenceTransformerModelWrapper.mean_pooling(
            embeddings, encodings["attention_mask"]
        )
        return sentence_embeddings[0]

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
