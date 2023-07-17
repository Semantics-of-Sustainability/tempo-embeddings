from typing import Any
from typing import Iterable
from typing import Optional
import numpy as np
from numpy.typing import ArrayLike
from tokenizers import Encoding
from .types import TokenInfo


class Passage:
    def __init__(self, text: str, metadata: dict = None) -> None:
        self._text = text.strip()
        self._metadata = metadata or {}
        self._embeddings: Optional[Any] = None
        self._tokenization: Optional[Encoding] = None

    @property
    def text(self) -> str:
        return self._text

    @property
    def metadata(self) -> dict:
        return self._metadata

    @property
    def embeddings(self) -> Optional[Any]:
        return self._embeddings

    @embeddings.setter
    def embeddings(self, value: Any):
        self._embeddings = value

    def has_embeddings(self) -> bool:
        return self.embeddings is not None

    @property
    def tokenization(self) -> Optional[Encoding]:
        return self._tokenization

    @tokenization.setter
    def tokenization(self, value: Encoding):
        self._tokenization = value

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

    def highlighted_text(
        self,
        token_info: TokenInfo,
        metadata_fields: Iterable[str] = None,
        max_context_length: int = 200,
    ) -> str:
        """Returns the text with the given word highlighted
        and metadata appended."""

        start, end = self.word_span(token_info)

        text = (
            self._text[:start] + f" <b>{self._text[start:end]}</b> " + self._text[end:]
        )
        if len(text) > max_context_length:
            text = text[start - max_context_length // 2 : end + max_context_length // 2]

        if metadata_fields:
            metadata = {key: self.get_metadata(key) for key in metadata_fields}
            text += f"<br>{metadata}"
        return text

    def token_embedding(self, token_info: TokenInfo) -> ArrayLike:
        """Returns the token embedding for the given char span in the given passage."""

        if self._embeddings is None:
            raise ValueError(
                "Passage has no embeddings. Call compute_embeddings first."
            )

        first_token, last_token = self.token_span(token_info)

        if first_token == last_token:
            token_embedding = self.embeddings[first_token]
        else:
            # multiple tokens
            token_embeddings = self.embeddings[first_token : last_token + 1]
            token_embedding = np.mean(token_embeddings, axis=0)

        return token_embedding

    def token_span(self, token_info: TokenInfo) -> tuple[int, int]:
        """Returns the start and end index of the token in the passage."""
        return (
            self._tokenization.char_to_token(token_info.start),
            self._tokenization.char_to_token(token_info.end - 1),
        )

    def word_span(self, token_info: TokenInfo) -> tuple[int, int]:
        word_index = self.tokenization.char_to_word(token_info.start)
        assert (
            self.tokenization.char_to_word(token_info.end - 1) == word_index
        ), "Token spans multiple words"

        return self.tokenization.word_to_chars(word_index)

    def __contains__(self, token: str) -> bool:
        return token in self._text

    def __len__(self) -> int:
        return len(self._text)

    def __hash__(self) -> int:
        return hash(self._text)

    def __eq__(self, other: object) -> bool:
        return self._text == other._text and self._metadata == other._metadata

    def __repr__(self) -> str:
        return f"Passage({self._text!r},{self.has_embeddings()!r}, {self._metadata!r})"

    def find(
        self, token: str, start: Optional[int] = None, end: Optional[int] = None
    ) -> int:
        return self._text.find(token, start, end)

    def findall(self, token: str) -> Iterable[int]:
        match_index = self.find(token)
        while match_index >= 0:
            yield match_index
            match_index = self.find(token, match_index + 1)

    @classmethod
    def from_text(
        cls,
        text: str,
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
                yield cls(text[start : start + window_size], metadata)
