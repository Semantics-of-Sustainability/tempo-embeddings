import abc
import logging
from functools import lru_cache
from typing import Iterable, Optional

import stanza
import torch
import wtpsplit
from sentence_splitter import SentenceSplitter

from .. import settings


class Segmenter(abc.ABC):
    """An abstract class for segmenting text into units."""

    @staticmethod
    def get_backend() -> str:
        """Determine which torch backend to use.

        If a device is specified in the settings, that device is used.
        Otherwise, try for cuda, then mps. Fall back to cpu.

        Returns:
            str: the backend to use.
        """
        if settings.DEVICE is not None:
            backend = settings.DEVICE
        elif torch.cuda.is_available():
            backend = "cuda"
        elif torch.backends.mps.is_available():
            backend = "mps"
        else:
            backend = "cpu"
        return backend

    @abc.abstractmethod
    def split(self, text: str, *, language: str) -> Iterable[str]:
        """Split the text into sentences.

        Args:
            text: the text to split.
        Returns:
            Iterable[str]: the sentences in the text.
        """
        return NotImplemented

    @classmethod
    @lru_cache(maxsize=4)
    def segmenter(cls, segmenter: Optional[str], language: str, **kwargs):
        """Return a segmenter of given type for the given language.

        Args:
            segmenter: the type of segmenter to use.
            language: the language code for the segmenter.
        Keyword Args:
            kwargs: additional arguments for the segmenter.
        Returns:
            Segmenter: a segmenter of the given type for the given language.
        """
        if segmenter == "sentence_splitter":
            _class = SentenceSplitterSegmenter
        elif segmenter == "stanza":
            _class = StanzaSegmenter
        elif segmenter == "wtp":
            _class = WtpSegmenter
        elif segmenter is None:
            logging.info("No segmenter specified, using None.")
            return None
        else:
            raise ValueError(f"Unknown segmenter: {segmenter}")
        return _class(language=language, **kwargs)


class SentenceSplitterSegmenter(Segmenter):
    def __init__(self, language: str) -> None:
        super().__init__()
        self._model = SentenceSplitter(language=language)
        self._language = language

    def split(self, text: str) -> Iterable[str]:
        return self._model.split(text)


class WtpSegmenter(Segmenter):
    """A segmenter that uses the WTP tokenizer."""

    def __init__(
        self,
        language: str,
        model: str = settings.WTPSPLIT_MODEL,
        style_or_domain: str = "-",
    ) -> None:
        super().__init__()

        self._model = wtpsplit.SaT(
            model, language=language, style_or_domain=style_or_domain
        )

        device: str = Segmenter.get_backend()
        logging.info("Using WtpSegemter on device: %s", device)
        self._model.half().to(device)

    def split(self, text: str) -> list[str]:
        return self._model.split(text)


class StanzaSegmenter(Segmenter):
    """A segmenter that uses the Stanza tokenizer."""

    def __init__(self, language: str, *, processors: str = "tokenize") -> None:
        super().__init__()
        self._model = stanza.Pipeline(lang=language, processors=processors)

    def split(self, text: str) -> Iterable[str]:
        for sentence in self._model(text).sentences:
            yield sentence.text
