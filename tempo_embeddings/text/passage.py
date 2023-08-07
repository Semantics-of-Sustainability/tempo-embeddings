import logging
from typing import Any
from typing import Iterable
from typing import Optional
from numpy.typing import ArrayLike
from scipy.spatial.distance import euclidean
from tokenizers import Encoding
from ..embeddings.model import TransformerModelWrapper
from .highlighting import Highlighting


class Passage:
    """A text passage with optional metadata and highlighting."""

    def __init__(
        self,
        text: str,
        metadata: dict = None,
        model: Optional[TransformerModelWrapper] = None,
        highlighting: Optional[Highlighting] = None,
    ) -> None:
        self._text = text.strip()
        self._metadata = metadata or {}
        self._model = model
        self._highlighting = highlighting

        self._embedding: Optional[ArrayLike] = None
        self._tokenization: Optional[Encoding] = None

    @property
    def text(self) -> str:
        return self._text

    @property
    def metadata(self) -> dict:
        return self._metadata

    @property
    def highlighting(self) -> Optional[Highlighting]:
        return self._highlighting

    @property
    def model(self) -> Optional[TransformerModelWrapper]:
        return self._model

    @model.setter
    def model(self, value: TransformerModelWrapper) -> None:
        self._model = value

    @property
    def embedding(self) -> Optional[ArrayLike]:
        return self._embedding

    @embedding.setter
    def embedding(self, value: Any):
        self._embedding = value

    @property
    def tokenization(self) -> Optional[Encoding]:
        return self._tokenization

    @tokenization.setter
    def tokenization(self, value: Encoding):
        self._tokenization = value

    def highlighted_text(
        self,
        metadata_fields: list[str],
        *,
        max_context_length: int = 200,
    ) -> str:
        word_start, word_end = self.word_span(
            self.highlighting.start, self.highlighting.end
        )

        pre_context = self.text[:word_start][-max_context_length:]
        post_context = self.text[word_end:][:max_context_length]

        text = (
            pre_context + f"<b>{self.text[word_start:word_end]}</b>" + post_context
        ).strip()

        if metadata_fields:
            metadata = {key: self.metadata.get(key) for key in metadata_fields}
            text += f"<br>{metadata}"
        return text

    def hover_data(self, metadata_keys: Optional[list[str]] = None) -> dict[str, Any]:
        if metadata_keys is None:
            metadata = self.metadata
        else:
            metadata = {key: self.metadata.get(key) for key in metadata_keys}

        return {"text": self.highlighting.text(self)} | metadata

    def token_embedding(self) -> ArrayLike:
        if self.highlighting is None:
            raise ValueError("No highlighting set.")
        if self.highlighting.token_embedding is None:
            self._model.compute_token_embedding(self)
        return self.highlighting.token_embedding

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

    def distance(self, vector: ArrayLike) -> float:
        """Returns the distance for UMAP embedding of highlighting to a vector.

        Args:
            vector: The vector to compute the distance to.
        Returns (float):
            The distance between the UMAP embedding of the highlighting and the vector.
        """
        return euclidean(self.highlighting.umap_embedding, vector)

    def words(self, use_tokenizer: bool = True) -> Iterable[str]:
        """Returns the words in the passage.

        Args:
            use_tokenizer: If True, uses the tokenizer to split the passage into words.
                Otherwise, splits the passage by whitespace.

        Returns:
            An iterable of words in the passage.
        """

        if not use_tokenizer and self.tokenization is None:
            logging.warning("Passage has no tokenization. Using whitespace splitting.")
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
        return f"Passage({self._text!r}, {self._metadata!r}, {self._highlighting!r})"

    def _partial_match(self, token: str, case_sensitive) -> Iterable[Highlighting]:
        if case_sensitive:
            text = self._text
        else:
            text = self._text.casefold()
            token = token.casefold()

        start: int = text.find(token)
        while start > -1:
            yield Highlighting(start, start + len(token))
            start = text.find(token, start + 1)

    def _exact_match(self, token: str, case_sensitive) -> list[Highlighting]:
        highlightings: list[Highlighting] = []
        if case_sensitive:
            text = self._text
        else:
            token = token.casefold()
            text = self._text.casefold()

        if token in text:  # quick check
            ### Tokenize if needed
            if self.tokenization is None:
                if self._model:
                    self._model.tokenize_passage(self)
                else:
                    raise RuntimeError(
                        "Passage has no tokenization and no model. "
                        "Cannot match exact tokens."
                    )

            ### Search for full tokens
            for word_index in self.tokenization.words:
                if word_index is not None:
                    start, end = self.tokenization.word_to_chars(word_index)
                    word = text[start:end]
                    if word == token:
                        highlightings.append(Highlighting(start, end))
        return highlightings

    def highlight(
        self, token: str, case_sensitive: bool = False, exact_match: bool = True
    ) -> list["Passage"]:
        """Match a token in this Passage and return a Passage for each match.

        Args:
            token: The token to match.
            case_sensitive: If True, matches tokens with same capitalization only.
            partial_match: If True, matches tokens that are part of a word.

        Returns:
            A Passage for each match. Empty if there is no match

        """

        highlightings: list[Highlighting] = (
            self._exact_match(token, case_sensitive)
            if exact_match
            else list(self._partial_match(token, case_sensitive))
        )

        match highlightings:
            case []:
                return []
            case [highlighting]:
                self._highlighting = highlighting
                return [self]
            case [*highlightings]:
                return [
                    self.with_highlighting(highlighting)
                    for highlighting in highlightings
                ]

    def with_highlighting(self, highlighting: Highlighting) -> "Passage":
        """Returns a new passage with the given highlighting.

        Args:
            highlighting: The highlighting to add.

        Returns:
            A new passage with the given highlighting.
        """
        if self.highlighting is not None:
            raise ValueError("Passage already has a highlighting.")
        return Passage(
            self._text,
            self._metadata,
            self.tokenization,
            highlighting=highlighting,
        )

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
        # TODO: use window_size on tokens instead of characters

        if window_size is None:
            # Single "window" of the entire text
            yield cls(text, metadata, model)
        else:
            if window_overlap is None:
                window_overlap = int(window_size / 10)

            for start in range(0, len(text), window_size - window_overlap):
                yield cls(text[start : start + window_size], metadata, model)
