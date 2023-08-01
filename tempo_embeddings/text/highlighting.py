from dataclasses import dataclass
from typing import TYPE_CHECKING
from typing import Iterable


if TYPE_CHECKING:
    from .passage import Passage


@dataclass
class Highlighting:
    start: int
    end: int
    passage: "Passage"

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
