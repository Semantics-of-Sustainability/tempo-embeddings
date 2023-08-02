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
    label: Any = None
    token_embedding: Optional[ArrayLike] = None

    def text(
        self,
        passage: "Passage",
        metadata_fields: Iterable[str] = None,
        *,
        max_context_length: int = 200,
    ) -> str:
        """Returns the text with the given word highlighted
        and metadata appended."""

        word_start, word_end = passage.word_span(self.start, self.end)

        pre_context = passage.text[:word_start][-max_context_length:]
        post_context = passage.text[word_end:][:max_context_length]

        text = (
            pre_context + f"<b>{passage.text[word_start:word_end]}</b>" + post_context
        ).strip()

        if metadata_fields:
            metadata = {key: passage.get_metadata(key) for key in metadata_fields}
            text += f"<br>{metadata}"
        return text

    def hover_data(
        self, passage: "Passage", *, metadata_keys: Optional[list[str]] = None
    ) -> dict[str, Any]:
        if metadata_keys is None:
            metadata = passage.metadata
        else:
            metadata = {key: passage.get_metadata(key) for key in metadata_keys}
        if self.label is not None:
            metadata["label"] = self.label

        return {"text": self.text(passage)} | metadata
