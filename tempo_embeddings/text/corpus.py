import csv
from pathlib import Path
from typing import Iterable
from typing import TextIO
from transformers.tokenization_utils_base import CharSpan
from ..settings import DEFAULT_ENCODING
from .passage import Passage


class Corpus(dict):
    def __init__(self, passages: dict[Passage, set[CharSpan]] = None):
        super().__init__()
        self._passages: dict[Passage, set[CharSpan]] = passages or {}

    def __add__(self, other: "Corpus") -> "Corpus":
        # FIXME: handle passages with multiple highlightings
        return Corpus(passages=self._passages | other._passages)

    def add_highlight(self, passage: Passage, highlight: CharSpan):
        if passage not in self:
            raise ValueError("Passage not in corpus.")

        self._passages.setdefault(passage, set()).add(highlight)

    def find(self, token: str) -> Iterable[tuple[Passage, int]]:
        for passage in self._passages:
            for match_index in passage.findall(token):
                yield (passage, match_index)

    def subcorpus(self, token: str) -> "Corpus":
        """Uses find() to generate a new Corpus object with matching passages and
        highlightings."""

        # TODO make this more efficient (using map/reduce)

        passages = {}
        for passage, match_index in self.find(token):
            passages.setdefault(passage, set()).add(
                Passage(passage.text[match_index : match_index + len(token)])
            )

        return Corpus(passages)

    @classmethod
    def from_lines(cls, f: TextIO, metadata: dict = None):
        """Read input data from an open file handler, one sequence per line."""
        return Corpus.from_passages((Passage(line, metadata) for line in f))

    @classmethod
    def from_file(cls, filepath: Path, encoding=DEFAULT_ENCODING):
        """Read input data from a file, one sequence per line."""
        with open(filepath, "rt", encoding=encoding) as f:
            return Corpus.from_lines(f)

    @classmethod
    def from_csv(
        cls, filepath: Path, text_columns: list[str], encoding=DEFAULT_ENCODING
    ):
        """Read input data from a CSV file."""
        with open(filepath, "rt", encoding=encoding) as f:
            reader = csv.DictReader(f)
            if not all(column in reader.fieldnames for column in text_columns):
                raise ValueError("Not all text columns found in CSV file.")

            passages = []
            for row in reader:
                metadata = {
                    column: row[column]
                    for column in reader.fieldnames
                    if column not in text_columns
                }
                for text_column in text_columns:
                    passages.append(Passage(row[text_column], metadata))
        return Corpus.from_passages(passages)

    @classmethod
    def from_passages(cls, passages: Iterable[Passage]):
        """Create a Corpus from a list of passages."""
        return cls({passage: set() for passage in passages})
