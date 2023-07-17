import abc
import logging
import torch
from tqdm import tqdm
from transformers import AutoTokenizer
from transformers import RobertaModel
from transformers import pipeline
from ..text.corpus import Corpus


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
    def compute_embeddings(self, corpus: Corpus, overwrite: bool = False):
        """Adds the embeddings for all passages in the given corpus."""

        if corpus.has_embeddings() and not overwrite:
            raise ValueError("Corpus already has embeddings")

        if not corpus.token_infos:
            logging.warning("Corpus does not have any token infos")

        # TODO: implement for other hidden layers but last one

        # TODO: run in batches
        for passage in tqdm(
            corpus.passages, unit="passage", desc="Computing embeddings"
        ):
            passage.embeddings = self._pipeline(passage.text)[0]
            passage.tokenization = self._tokenizer(passage.text)[0]

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
