from typing import Any
from typing import Iterable
from typing import Optional
from .types import TokenInfo


class Passage:
    def __init__(self, text: str, metadata: dict = None) -> None:
        self._text = text.strip()
        self._metadata = metadata or {}

    @property
    def text(self) -> str:
        return self._text

    @property
    def metadata(self) -> dict:
        return self._metadata

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
        self, token_info: TokenInfo, metadata_fields: Iterable[str] = None
    ) -> str:
        start = self.word_begin(token_info)
        end = self.word_end(token_info)
        text = (
            self._text[:start] + f" <b>{self._text[start:end]}</b> " + self._text[end:]
        )

        if metadata_fields:
            metadata = {key: self.get_metadata(key) for key in metadata_fields}
            text += f"<br>{metadata}"
        return text

    def word_begin(self, token_info: TokenInfo) -> int:
        """Returns the index of the beginning of the word containing the token."""
        for i in range(token_info.start, 0, -1):
            if not self._text[i - 1].isalnum():
                return i
        return 0

    def word_end(self, token_info: TokenInfo) -> int:
        """Returns the index of the beginning of the word containing the token."""
        for i in range(token_info.end, len(self._text), 1):
            if not self._text[i].isalnum():
                return i
        return len(self._text)

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
