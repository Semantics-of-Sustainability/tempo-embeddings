import json
from functools import lru_cache
from pathlib import Path
from typing import Any, Iterable, Optional

import stanza
from pydantic import Field, model_validator
from pydantic.dataclasses import dataclass
from tqdm import tqdm

from .. import settings
from ..text.corpus import Corpus
from ..text.segmenter import Segmenter


@lru_cache(maxsize=2)
def nlp_pipeline(lang: Optional[str]) -> stanza.Pipeline:
    """Return a Stanza pipeline for the given language.

    Args:
        lang: the language code for the pipeline. Can be None.
    Returns:
        stanza.Pipeline: a Stanza pipeline for the given language. None if lang is None.
    """
    return stanza.Pipeline(lang=lang, processors="tokenize") if lang else None


@dataclass
class CorpusConfig:
    directory: Path
    glob_pattern: str = "*_????.csv"
    loader_type: str = "csv"
    text_columns: list[str] = Field(default=["content"])
    encoding: str = "utf-8"
    delimiter: str = ";"
    compression: Optional[str] = None
    language: Optional[str] = None
    """The 'language' parameter is used for the NLP pipeline."""
    segmenter: Optional[str] = settings.SEGMENTER

    @model_validator(mode="after")
    def validate_segmenter(self):
        if self.segmenter and not self.language:
            raise ValueError(
                f"Segmenter '{self.segmenter}' was specified, but no language"
            )

        return self

    def asdict(self, *, properties: Optional[list[str]] = None):
        _converters = {"directory": str}
        return {
            property: _converters.get(property, lambda x: x)(getattr(self, property))
            for property in properties or self.__annotations__.keys()
        }

    def not_empty(self):
        return any(self.files())

    def files(self):
        return self.directory.glob(self.glob_pattern)

    def build_corpora(
        self,
        filter_terms,
        *,
        skip_files: Optional[set[str]] = None,
        max_files: Optional[int] = None,
        **kwargs,
    ) -> Iterable[Corpus]:
        """Build one corpus per file from the configuration.

        Args:
            filter_terms: a list of terms to filter out.
            skip_files: a set of file names to skip. Defaults to None.
            max_files: the maximum number of files to process.
            **kwargs: additional parameters to pass to the Corpus constructor: window_size

        Yields:
            Corpus: a corpus object, one per corpus file
        """
        skip_files: set[str] = skip_files or set()
        files = [file for file in self.files() if file.name not in skip_files]

        for file in tqdm(files[:max_files], desc=self.directory.name, unit="file"):
            if self.loader_type == "csv":
                yield Corpus.from_csv_file(
                    filepath=file,
                    text_columns=self.text_columns,
                    filter_terms=filter_terms,
                    encoding=self.encoding,
                    compression=self.compression,
                    delimiter=self.delimiter,
                    nlp_pipeline=Segmenter.segmenter(self.segmenter, self.language),
                    **kwargs,
                )
            else:
                raise NotImplementedError(f"Unrecognized format '{self.file_type}'")

    def build_corpus(
        self,
        filter_terms,
        *,
        skip_files: Optional[set[str]] = None,
        max_files: Optional[int] = None,
        **kwargs,
    ) -> Corpus:
        """Build a corpus from the configuration.

        Args:
            filter_terms: a list of terms to filter out.
            skip_files: a set of file names to skip. Defaults to None.
            max_files: the maximum number of files to process.
            **kwargs: additional parameters to pass to the Corpus constructor: window_size.

        Returns:
            Corpus: a corpus object.
        """
        skip_files: set[str] = skip_files or set()
        files: Iterable[Path] = [
            file for file in self.files() if file.name not in skip_files
        ][:max_files]

        if self.loader_type == "csv":
            corpus = Corpus.from_csv_files(
                files=files,
                filter_terms=filter_terms,
                text_columns=self.text_columns,
                encoding=self.encoding,
                compression=self.compression,
                delimiter=self.delimiter,
                nlp_pipeline=Segmenter.segmenter(self.segmenter, self.language),
                **kwargs,
            )
        else:
            raise NotImplementedError(f"Unrecognized format '{self.file_type}'")
        return corpus


class CorpusReader:
    """Helper class for processing corpus configuration files."""

    def __init__(
        self,
        config_file: Path = settings.CORPORA_CONFIG_FILE,
        *,
        corpora: list[str] = None,
        base_dir: Path = settings.CORPUS_DIR,
    ):
        """
        Read corpora.

        Args:
            config_file (Path): the path to the corpora configuration file.
            corpora: a list of corpora to read. If None, read all corpora.
            base_dir (Path, optional): the base directory in which corpora are stored. Defaults to settings.CORPUS_DIR.
        """
        self._corpus_dir = base_dir

        with open(config_file, "rt") as f:
            corpora_configs: dict[str, dict[str, Any]] = json.load(f)

        self._corpora = {}
        for name, config in corpora_configs.items():
            if corpora is None or name in corpora:
                directory = base_dir / config.get("sub-directory", "") / name

                self._corpora[name] = CorpusConfig(
                    **(config | {"directory": directory})
                )

        if corpora is not None:
            for corpus in corpora:
                if corpus not in self._corpora:
                    raise ValueError(f"Corpus '{corpus}' not defined in {config_file}")

    def __contains__(self, name) -> bool:
        return name in self._corpora

    def __getitem__(self, name: str) -> CorpusConfig:
        return self._corpora[name]

    def corpora(self, must_exist: bool = False) -> Iterable[str]:
        """Return the names of all corpora.

        Args:
            exists: If True, only return corpora for which the directory exists.
        Returns:
            Iterable[str]: the available corpus names
        """
        for name, config in self._corpora.items():
            if (not must_exist) or config.exists():
                yield name
