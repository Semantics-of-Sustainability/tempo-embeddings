import hashlib
import logging
import string
from typing import Any
from typing import Iterable
from typing import Optional
from .highlighting import Highlighting


class Passage:
    """A text passage with optional metadata and highlighting."""

    def __init__(
        self,
        text: str,
        metadata: dict = None,
        highlighting: Optional[Highlighting] = None,
        embedding: Optional[list[float]] = None,
        full_word_spans: Optional[list[tuple[int, int]]] = None,
        char2tokens: Optional[list[int]] = None,
        unique_id: str = None
    ) -> None:
        # pylint: disable=too-many-arguments
        self._text = text.strip()
        self._unique_id = unique_id
        self._metadata = metadata or {}
        self._highlighting = highlighting
        self._embedding = embedding
        self._full_word_spans = full_word_spans
        self._char2tokens = char2tokens

    @property
    def text(self) -> str:
        return self._text
    
    @property
    def unique_id(self) -> str:
        return self._unique_id

    @property
    def tokenized_text(self) -> str:
        return self._tokenized_text

    @property
    def metadata(self) -> dict:
        return self._metadata

    @property
    def highlighting(self) -> Optional[Highlighting]:
        return self._highlighting

    @property
    def embedding(self) -> Optional[list[float]]:
        return self._embedding

    @property
    def full_word_spans(self) -> Optional[list[tuple[int, int]]]:
        return self._full_word_spans
    
    @property
    def char2tokens(self) -> Optional[list[int]]:
        return self._char2tokens

    @unique_id.setter
    def unique_id(self, value: str):
        if not isinstance(value, str):
            raise ValueError("The Unique ID of a Passage should be a SHA256 string")
        self._unique_id = value

    @tokenized_text.setter
    def tokenized_text(self, value: list[str]):
        if not isinstance(value, list):
            raise ValueError("You should pass a list of strings as tokenized text")
        self._tokenized_text = value

    @embedding.setter
    def embedding(self, value: list[float]):
        if not isinstance(value, list):
            raise ValueError("You should pass a list of floats as the embedding vector")
        self._embedding = value

    @full_word_spans.setter
    def full_word_spans(self, value: list[tuple[int, int]]):
        self._full_word_spans = value

    @char2tokens.setter
    def char2tokens(self, value: list[int]):
        self._char2tokens = value

    def get_unique_id(self) -> str:
        if not self._unique_id:
            meta_sorted = sorted(self.metadata.items(), key=lambda x: x[0]) if self.metadata else []
            key = self.text + str(meta_sorted) + str(self.highlighting)
            hex_dig = hashlib.sha256(key.encode()).hexdigest()
            self._unique_id = hex_dig
        return self._unique_id
            

    def contains_any(self, tokens: Iterable[str]) -> bool:
        """Returns True if any of the tokens are contained in the passage.

        Checks for case-insensitive matches.

        Args:
            tokens: The tokens to check.

        Returns (bool):
            True if any of the tokens are contained in the passage.
        """
        return tokens is None or any(
            token.casefold() in self._text.casefold() for token in tokens
        )

    def highlighted_text(
        self,
        *,
        metadata_fields: Optional[list[str]] = None,
        max_context_length: int = 1000,
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
        
        word_start, word_end = self.highlighting.start, self.highlighting.end


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

        return {"text": self.text} | metadata

    def set_metadata(self, key: str, value: Any) -> None:
        """Sets a metadata key to a value.

        Args:
            key: The metadata key to set.
            value: The value to set the metadata key to.
        """
        self._metadata[key] = value

    def word_span(self, start, end) -> tuple[int, int]:
        word_index = self.tokenization.char_to_word(start)
        if self.tokenization.char_to_word(end - 1) != word_index:
            logging.info(
                "Token spans from %d to %d over multiple words in passage '%s'",
                start,
                end,
                self.text,
            )
        return self.tokenization.word_to_chars(word_index)

    def words(self) -> list[str]:
        """Returns the full words in the passage if word spans are known. Otherwise, splits the passage by whitespace.

        Returns:
            An list of words in the passage.
        """

        if self.full_word_spans is None:
            tokens = [
                token.strip(string.punctuation).strip() for token in self._text.split()
            ]
            tokens = [token for token in tokens if len(token) > 1]
        else:
            tokens = [self.text[span[0]:span[1]] for span in self.full_word_spans]

        return tokens

    def __contains__(self, token: str) -> bool:
        return token in self._text

    def __len__(self) -> int:
        return len(self._text)

    def __hash__(self) -> int:
        return (
            hash(self._text)
            + hash(frozenset(self._metadata.keys()))
            + hash(frozenset(self._metadata.values()))
            + hash(self._highlighting)
        )

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
            exact_match: If True, only matches tokens that are a complete word (no substrings).
                Requires tokenization.

        Returns:
            A Passage for each match. Empty if there is no match

        """

        highlightings: list[Highlighting] = (
            self._exact_match(token, case_sensitive)
            if exact_match
            else list(self._partial_match(token, case_sensitive))
        )

        # note: this creates a new Passage object even if there is only one highlighting
        return [self.with_highlighting(highlighting) for highlighting in highlightings]

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
            text=self._text,
            metadata=self._metadata,
            highlighting=highlighting,
            full_word_spans=self.full_word_spans,
            char2tokens=self.char2tokens
        )

    @classmethod
    def from_text(
        cls,
        text: str,
        *,
        window_size: int = None,
        window_overlap: int = None,
        metadata: dict = None,
        nlp_pipeline = None,
    ) -> Iterable["Passage"]:
        """Create a Passage from a text string."""

        # TODO validate parameters
        # TODO: use window_size on tokens instead of characters

        if nlp_pipeline:
            doc = nlp_pipeline(text)
            for ix, sentence in enumerate(doc.sentences):
                print(ix, sentence.text)
                metadata["sentence_index"] = ix
                yield cls(sentence.text, metadata)
        elif window_size is None:
            # Single window comprising the entire text
            yield cls(text, metadata)
        else:
            if window_overlap is None:
                window_overlap = int(window_size / 10)

            for start in range(0, len(text), window_size - window_overlap):
                try:
                    # Expand window to first preceeding whitespace
                    _start = text.rindex(" ", 0, start) + 1
                except ValueError:
                    _start = start
                try:
                    # Expand window to next succeeding whitespace
                    _end = text.index(" ", _start + window_size)
                except ValueError:
                    _end = _start + window_size

                yield cls(text[_start:_end], metadata)
