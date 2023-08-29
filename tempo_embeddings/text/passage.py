import string
from typing import Any
from typing import Iterable
from typing import Optional
from numpy.typing import ArrayLike
from scipy.spatial.distance import euclidean
from tokenizers import Encoding
from .highlighting import Highlighting


class Passage:
    """A text passage with optional metadata and highlighting."""

    # pylint: disable=too-many-arguments
    def __init__(
        self,
        text: str,
        metadata: dict = None,
        highlighting: Optional[Highlighting] = None,
        embedding: Optional[ArrayLike] = None,
        tokenization: Optional[Encoding] = None,
    ) -> None:
        self._text = text.strip()
        self._metadata = metadata or {}
        self._highlighting = highlighting

        self._embedding: Optional[ArrayLike] = embedding
        self._tokenization: Optional[Encoding] = tokenization

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
        *,
        metadata_fields: Optional[list[str]] = None,
        max_context_length: int = 200,
    ) -> str:
        """Returns the text with the highlighted word in bold.

        Args:
            metadata_fields (Optional[list[str]]): A list of metadata keys to include in the text.
            max_context_length (int): The maximum number of characters to include before and after the highlighted word.

        Returns (str):
            The text with the highlighted word in bold.
        """
        if not self.highlighting:
            raise ValueError(f"Passage does not have a highlighting: {str(self)}")
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

    def hover_data(
        self, metadata_fields: Optional[list[str]] = None, max_length: int = 100
    ) -> dict[str, Any]:
        if metadata_fields is None:
            metadata = self.metadata
        else:
            metadata = {
                key: str(self.metadata.get(key))[:max_length] for key in metadata_fields
            }

        text = self.highlighted_text() if self.highlighting else self.text
        return {"text": text} | metadata

    def token_embedding(self) -> ArrayLike:
        if self.highlighting is None:
            raise RuntimeError("No highlighting set.")
        if self.highlighting.token_embedding is None:
            raise RuntimeError("No embeddings computed for passage.")
        return self.highlighting.token_embedding

    def set_metadata(self, key: str, value: Any) -> None:
        """Sets a metadata key to a value.

        Args:
            key: The metadata key to set.
            value: The value to set the metadata key to.
        """
        self._metadata[key] = value

    def word_span(self, start, end) -> tuple[int, int]:
        if not self.tokenization:
            raise RuntimeError("Passage has no tokenization.")
        if self.tokenization is None:
            raise RuntimeError("Passage has no tokenization.")

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

        if not use_tokenizer:
            tokens = [
                token.strip(string.punctuation).strip() for token in self._text.split()
            ]
            for token in tokens:
                if len(token) > 1:
                    yield token
        else:
            if self.tokenization is None:
                raise RuntimeError("Passage has no tokenization.")

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
                raise RuntimeError("Passage has no tokenization.")

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
        """Returns a new Passage object with the given highlighting.

        Args:
            highlighting: The highlighting to add.

        Returns:
            A new Passage object with the given highlighting.
        """
        if self.highlighting is not None:
            raise RuntimeError("Passage already has a highlighting.")
        return Passage(
            self._text,
            self._metadata,
            highlighting=highlighting,
            embedding=self.embedding,
            tokenization=self.tokenization,
        )

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
        # TODO: use window_size on tokens instead of characters

        if window_size is None:
            # Single window comprising the entire text
            yield cls(text, metadata)
        else:
            if window_overlap is None:
                window_overlap = int(window_size / 10)

            for start in range(0, len(text), window_size - window_overlap):
                yield cls(text[start : start + window_size], metadata)
