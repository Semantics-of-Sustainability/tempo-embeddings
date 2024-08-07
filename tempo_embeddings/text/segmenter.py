import abc
from functools import lru_cache
from typing import Iterable

import stanza
import wtsplit

from .. import settings


class Segmenter(abc.ABC):
    """An abstract class for segmenting text into units."""

    @abc.abstractmethod
    def split(self, text: str, *, language: str) -> Iterable[str]:
        return NotImplemented

    @classmethod
    @lru_cache(maxsize=4)
    def segmenter(cls, segmenter: str, language: str, **kwargs):
        if segmenter == "stanza":
            _class = StanzaSplitter
        elif segmenter == "wtp":
            _class = WtpSegmenter
        else:
            raise ValueError(f"Unknown segmenter: {segmenter}")
        return _class(language=language, **kwargs)


class WtpSegmenter(Segmenter):
    """A segmenter that uses the WTP tokenizer."""

    def __init__(
        self,
        language: str,
        model: str = settings.WTPSPLIT_DEFAULT_MODEL,
        style_or_domain: str = "-",
    ) -> None:
        super().__init__()

        self._model = wtsplit.SaT(
            model, language=language, style_or_domain=style_or_domain
        )

    def split(self, text: str) -> list[str]:
        return self._model.split(text)


class StanzaSplitter(Segmenter):
    """A segmenter that uses the Stanza tokenizer."""

    def __init__(self, language: str, *, processors: str = "tokenize") -> None:
        super().__init__()
        self._model = stanza.Pipeline(lang=language, processors=processors)

    def split(self, text: str) -> Iterable[str]:
        for sentence in self._model(text).sentences:
            yield sentence.text
