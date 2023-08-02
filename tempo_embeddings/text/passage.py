import logging
from typing import Any
from typing import Iterable
from typing import Optional
from numpy.typing import ArrayLike
from tokenizers import Encoding
from ..embeddings.model import TransformerModelWrapper
from .highlighting import Highlighting


class Passage:
    def __init__(
        self,
        text: str,
        metadata: dict = None,
        model: Optional[TransformerModelWrapper] = None,
        highlightings: Optional[list[Highlighting]] = None,
    ) -> None:
        self._text = text.strip()
        self._metadata = metadata or {}
        self._model = model
        self._highlightings = highlightings or []

        self._embeddings: Optional[ArrayLike] = None
        self._tokenization: Optional[Encoding] = None

    @property
    def text(self) -> str:
        return self._text

    @property
    def metadata(self) -> dict:
        return self._metadata

    @property
    def highlightings(self) -> list[Highlighting]:
        return self._highlightings

    @highlightings.setter
    def highlightings(self, value: list[Highlighting]) -> None:
        self._highlightings = value

    @property
    def model(self) -> Optional[TransformerModelWrapper]:
        return self._model

    @model.setter
    def model(self, value: TransformerModelWrapper) -> None:
        self._model = value

    @property
    def embeddings(self) -> Optional[ArrayLike]:
        return self._embeddings

    @embeddings.setter
    def embeddings(self, value: Any):
        self._embeddings = value

    @property
    def tokenization(self) -> Optional[Encoding]:
        return self._tokenization

    @tokenization.setter
    def tokenization(self, value: Encoding):
        self._tokenization = value

    def highlighted_texts(self, metadata_fields: list[str]) -> list[str]:
        return [
            highlighting.text(self, metadata_fields)
            for highlighting in self._highlightings
        ]

    def hover_datas(self) -> list[dict[str, Any]]:
        return [highlighting.hover_data(self) for highlighting in self._highlightings]

    def token_embeddings(self) -> Iterable[ArrayLike]:
        for highlighting in self.highlightings:
            if highlighting.token_embedding is None:
                self._model.compute_token_embedding(self, highlighting)
            yield highlighting.token_embedding

    def has_metadata(self, key: str) -> bool:
        """Returns True if the metadata key exists."""
        return key in self.metadata

    def get_metadata(self, key: str, strict: bool = True) -> Any:
        """Returns the value for a given metadata key.

        Args:
            key: The metadata key to return the value for.
            strict: If True, raises KeyError if the key does not exist.

        Raises:
            KeyError: If the metadata key does not exist and strict is True.

        Returns:
            The value for the given metadata key or
            None if the key does not exist strict is False.
        """
        return self._metadata[key] if strict else self._metadata.get(key)

    def set_metadata(self, key: str, value: Any) -> None:
        """Sets a metadata key to a value.

        Args:
            key: The metadata key to set.
            value: The value to set the metadata key to.
        """
        self._metadata[key] = value

    def tokenize(self) -> None:
        if self._model is None:
            raise ValueError("No model set.")

        self._model.tokenize_passage(self)

    def token_span(self, start, end) -> tuple[int, int]:
        """Returns the start and end index of the token in the passage."""
        return (
            self._tokenization.char_to_token(start),
            self._tokenization.char_to_token(end - 1),
        )

    def word_span(self, start, end) -> tuple[int, int]:
        if not self.tokenization and self._model:
            self._model.tokenize_passage(self)
        if self.tokenization is None:
            raise ValueError("Passage has no tokenization.")

        word_index = self.tokenization.char_to_word(start)
        assert (
            self.tokenization.char_to_word(end - 1) == word_index
        ), "Token spans multiple words"

        return self.tokenization.word_to_chars(word_index)

    def words(self, use_tokenizer: bool = True) -> Iterable[str]:
        """Returns the words in the passage.

        Args:
            use_tokenizer: If True, uses the tokenizer to split the passage into words.
                Otherwise, splits the passage by whitespace.

        Returns:
            An iterable of words in the passage.
        """

        if not use_tokenizer and self.tokenization is None:
            yield from self._text.split()
        else:
            self.tokenize()

            for i in self.tokenization.words:
                if i is not None:
                    start, end = self.tokenization.word_to_chars(i)
                    yield self._text[start:end]

    def __contains__(self, token: str) -> bool:
        return token in self._text

    def __len__(self) -> int:
        return len(self._text)

    def __hash__(self) -> int:
        return hash(self._text)

    def __eq__(self, other: object) -> bool:
        return self._text == other._text and self._metadata == other._metadata

    def __repr__(self) -> str:
        return f"Passage({self._text!r}, {self._metadata!r})"

    def find(
        self, token: str, start: Optional[int] = None, end: Optional[int] = None
    ) -> tuple[int, int]:
        start = self._text.casefold().find(token.casefold(), start, end)
        end = start + len(token)
        return start, end

    def add_highlightings(self, token: str) -> bool:
        """Find matches for token string and add highlightings to the passage."""
        # Quick check for match:
        highlightings = []

        if token.casefold() in self._text.casefold():
            if self.tokenization is None and self._model is not None:
                # Tokenize if needed
                self._model.tokenize_passage(self)

            if self.tokenization is None:
                logging.warning(
                    "Passage has no tokenization. "
                    "Proceeding with simple string matching."
                )
                start, end = self.find(token)

                while start >= 0:
                    highlightings.append(Highlighting(start, end, self))
                    start, end = self.find(token, end + 1)
            else:
                # Search for full tokens
                for word_index in self.tokenization.words:
                    if word_index is not None:
                        start, end = self.tokenization.word_to_chars(word_index)
                        word = self._text[start:end]
                        if word.casefold() == token.casefold():
                            highlightings.append(Highlighting(start, end, self))

        self.highlightings += highlightings
        return len(highlightings) > 0

    def with_highlightings(self, *spans: tuple[int, int]) -> "Passage":
        self.highlightings.extend(
            [Highlighting(start, end, self) for start, end in spans]
        )
        return self

    @classmethod
    def from_text(
        cls,
        text: str,
        model: Optional[TransformerModelWrapper] = None,
        *,
        window_size: int = None,
        window_overlap: int = None,
        metadata: dict = None,
    ) -> Iterable["Passage"]:
        """Create a Passage from a text string."""

        # TODO validate parameters

        if window_size is None:
            # Single "window" of the entire text
            yield cls(text, metadata)
        else:
            if window_overlap is None:
                window_overlap = int(window_size / 10)

            for start in range(0, len(text), window_size - window_overlap):
                yield cls(text[start : start + window_size], metadata, model)
