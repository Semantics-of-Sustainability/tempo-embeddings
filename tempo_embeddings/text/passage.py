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

    def highlighted_text(self, token_info: TokenInfo) -> str:
        return (
            self._text[: token_info.start]
            + f"<b>{self._text[token_info.start:token_info.end]}</b>"
            + self._text[token_info.end :]
        )

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
