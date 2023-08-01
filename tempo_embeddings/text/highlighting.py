from dataclasses import dataclass
from typing import TYPE_CHECKING
from typing import Any
from typing import Iterable
from typing import Optional
from numpy.typing import ArrayLike


if TYPE_CHECKING:
    from .passage import Passage


@dataclass
class Highlighting:
    start: int
    end: int
    passage: "Passage"
    label: Any = None
    token_embedding: Optional[ArrayLike] = None

    def get_token_embedding(self) -> ArrayLike:
        if self.token_embedding is None:
            self.token_embedding = self.passage.token_embedding(self.start, self.end)
        return self.token_embedding

    def text(
        self, metadata_fields: Iterable[str] = None, max_context_length: int = 200
    ) -> str:
        """Returns the text with the given word highlighted
        and metadata appended."""

        word_start, word_end = self.passage.word_span(self.start, self.end)

        pre_context = self.passage.text[:word_start][-max_context_length:]
        post_context = self.passage.text[word_end:][:max_context_length]

        text = (
            pre_context
            + f"<b>{self.passage.text[word_start:word_end]}</b>"
            + post_context
        ).strip()

        if metadata_fields:
            metadata = {key: self.passage.get_metadata(key) for key in metadata_fields}
            text += f"<br>{metadata}"
        return text

    def hover_data(
        self, *, metadata_keys: Optional[list[str]] = None
    ) -> dict[str, Any]:
        if metadata_keys is None:
            metadata = self.passage.metadata
        else:
            metadata = {key: self.passage.get_metadata(key) for key in metadata_keys}
        if self.label is not None:
            metadata["label"] = self.label

        return {"text": self.text()} | metadata
