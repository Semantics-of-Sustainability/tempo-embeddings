import abc
import csv
import logging
import re
from functools import lru_cache
from posixpath import basename
from typing import Any, Iterable, Optional

import stanza
import torch
import wtpsplit
from sentence_splitter import SentenceSplitter

from .. import settings
from .passage import Passage


class Segmenter(abc.ABC):
    """An abstract class for segmenting text into units."""

    _ALPHABET_REGEX = re.compile(r"[^a-zA-Z]")

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

    def __init__(self, language: str) -> None:
        """The common constructor for inheriting segmenters.

        Args:
            language: the language code for the segmenter. Specifics depend on the segmenter.
        """
        self._language = language

    @abc.abstractmethod
    def split(self, text: str) -> Iterable[str]:
        """Split the text into sentences.

        Args:
            text: the text to split.
        Returns:
            Iterable[str]: the sentences in the text.
        """
        return NotImplemented

    def passages(
        self, text: str, *, metadata: Optional[dict[str, Any]] = None
    ) -> Iterable[Passage]:
        """Yield passages from the text.

        Args:
            text: the text to split into passages.
            metadata: the metadata to attach to the passages.
        Yields:
            Passage: the passages from the text.
        """

        for idx, sentence in enumerate(self.split(text)):
            metadata = (metadata or {}) | {"sentence_index": idx}

            yield Passage(sentence, metadata)

    def passages_from_dict_reader(
        self,
        reader: csv.DictReader,
        min_length: int = settings.PASSAGE_LENGTH,
        *,
        provenance: str,
        text_columns: list[str],
        filter_terms: Optional[Iterable[str]] = None,
    ) -> list[Passage]:
        """Read passages from a CSV file.

        Args:
            reader: a CSV reader object
            min_length: the minimum sentence length. Defaults to settings.PASSAGE_LENGTH
            provenance: the name of the CSV file.
            text_columns: the columns in the CSV file that contain text.
            filter_terms: a list of terms; if given, only passages containing these terms are returned.
        Return:
            list[Passage]: the passages from the CSV file.
        """
        for column in text_columns:
            if column not in reader.fieldnames:
                raise ValueError(
                    f"Text column(s) {text_columns} not found in CSV file '{provenance}'."
                )

        passages = []
        metadata = {"provenance": basename(provenance)}
        for row in reader:
            row_metadata = metadata | {
                column: row[column]
                for column in reader.fieldnames
                # do not include blank column names and text columns in metadata
                if column and column not in text_columns
            }

            for text_column in text_columns:
                if filter_terms and not any(
                    term.casefold() in row[text_column].casefold()
                    for term in filter_terms
                ):
                    continue

                column_passages: Iterable[Passage] = Passage.merge(
                    self.passages(row[text_column], metadata=row_metadata),
                    min_length=min_length,
                )

                if filter_terms:
                    # Highlight terms in passages
                    for passage in column_passages:
                        for term in filter_terms:
                            if passage.contains(term):
                                passages.extend(
                                    passage.highlight(term, exact_match=False)
                                )
                else:
                    passages.extend(column_passages)
        return passages

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
        elif segmenter == "window":
            _class = WindowSegmenter
        elif segmenter is None:
            logging.info("No segmenter specified, using None.")
            return None
        else:
            raise ValueError(f"Unknown segmenter: {segmenter}")
        return _class(language=language, **kwargs)


class SentenceSplitterSegmenter(Segmenter):
    def __init__(self, language: str, **kwargs) -> None:
        super().__init__(language=language, **kwargs)
        self._model = SentenceSplitter(language=self._language)

    def split(self, text: str) -> Iterable[str]:
        return self._model.split(text)


class WtpSegmenter(Segmenter):
    """A segmenter that uses the WTP tokenizer."""

    def __init__(
        self,
        language: str,
        model: str = settings.WTPSPLIT_MODEL,
        style_or_domain: str = "-",
        **kwargs,
    ) -> None:
        super().__init__(language=language, **kwargs)

        self._model = wtpsplit.SaT(
            model, language=self._language, style_or_domain=style_or_domain
        )

        device: str = Segmenter.get_backend()
        logging.info("Using WtpSegemter on device: %s", device)
        self._model.half().to(device)

    def split(self, text: str) -> Iterable[str]:
        return self._model.split(text, strip_whitespace=True)


class StanzaSegmenter(Segmenter):
    """A segmenter that uses the Stanza tokenizer."""

    def __init__(
        self, language: str, *, processors: str = "tokenize", **kwargs
    ) -> None:
        super().__init__(language=language, **kwargs)
        self._model = stanza.Pipeline(lang=self._language, processors=processors)

    def split(self, text: str) -> Iterable[str]:
        return (sentence.text for sentence in self._model(text).sentences)


class WindowSegmenter(Segmenter):
    """A segmenter that uses a sliding window to segment text."""

    def __init__(
        self,
        language=None,
        *,
        window_size: int,
        window_overlap: Optional[int] = None,
        **kwargs,
    ) -> None:
        super().__init__(language=language, **kwargs)

        self._window_size = window_size

        # Default to 10% overlap
        if self._window_size is not None:
            self._window_overlap = window_overlap or self._window_size // 10
        elif window_overlap is not None:
            raise TypeError("Window overlap specified without window size.")
        else:
            logging.info("No window size specified, not splitting the input texts.")
            self._window_overlap = None

    def split(self, text: str) -> Iterable[str]:
        if self._window_size is None:
            yield text
        else:
            for start in range(0, len(text), self._window_size - self._window_overlap):
                try:
                    # Find first whitespace before current window start
                    _start = text.rindex(" ", 0, start) + 1
                except ValueError:
                    _start = start

                try:
                    # Find next whitespace after current window end
                    _end = text.index(" ", _start + self._window_size)
                except ValueError:
                    _end = _start + self._window_size

                yield text[_start:_end]
