import json
from contextlib import nullcontext as does_not_raise
from pathlib import Path
from tempfile import gettempdir

import pytest

from tempo_embeddings.io.corpus_reader import CorpusConfig, CorpusReader
from tempo_embeddings.settings import CORPORA_CONFIG_FILE

from ...conftest import CORPUS_DIR


class TestCorpusReader:
    _CONFIGURED_CORPORA = json.load(CORPORA_CONFIG_FILE.open("rt")).keys()

    @pytest.mark.parametrize(
        "corpora, base_dir, must_exist, expected, expected_exception",
        [
            (None, gettempdir(), False, _CONFIGURED_CORPORA, does_not_raise()),
            (None, gettempdir(), True, [], does_not_raise()),
            (None, CORPUS_DIR, False, _CONFIGURED_CORPORA, does_not_raise()),
            # FIXME: this test depends on local data being present or not:
            (None, CORPUS_DIR, True, ["ANP"], does_not_raise()),
            (["ANP"], CORPUS_DIR, True, ["ANP"], does_not_raise()),
            (["test corpus"], CORPUS_DIR, True, None, pytest.raises(ValueError)),
        ],
    )
    def test_corpora(self, corpora, base_dir, must_exist, expected, expected_exception):
        with expected_exception:
            reader = CorpusReader(corpora=corpora, base_dir=Path(base_dir))
            assert sorted(reader.corpora(must_exist=must_exist)) == sorted(expected)


class TestCorpusConfig:
    @pytest.fixture
    def tmp_corpus_config(self, tmp_path):
        return CorpusConfig(directory=tmp_path / "test_corpus", segmenter=None)

    @pytest.fixture
    def anp_corpus_config(self):
        return CorpusConfig(
            directory=CORPUS_DIR / "ANP",
            glob_pattern="ANP_????.csv.gz",
            loader_type="csv",
            text_columns=["content"],
            encoding="iso8859_1",
            compression="gzip",
            delimiter=";",
            language=None,
            segmenter=None,
        )

    @pytest.mark.parametrize(
        "properties, expected",
        [
            (
                None,
                {
                    "directory": str(CORPUS_DIR / "ANP"),
                    "glob_pattern": "ANP_????.csv.gz",
                    "loader_type": "csv",
                    "text_columns": ["content"],
                    "encoding": "iso8859_1",
                    "compression": "gzip",
                    "delimiter": ";",
                    "language": None,
                    "segmenter": None,
                },
            ),
            (["language"], {"language": None}),
        ],
    )
    def test_asdict(self, anp_corpus_config, properties, expected):
        assert anp_corpus_config.asdict(properties=properties) == expected

    def test_files_tmp(self, tmp_corpus_config):
        assert sorted(tmp_corpus_config.files()) == []

        tmp_corpus_config.directory.mkdir()
        (tmp_corpus_config.directory / "file1.txt").touch()
        (tmp_corpus_config.directory / "file2.csv").touch()
        (tmp_corpus_config.directory / "file2_1984.csv").touch()

        assert sorted(tmp_corpus_config.files()) == [
            tmp_corpus_config.directory / "file2_1984.csv"
        ]

    def test_files(self, anp_corpus_config):
        assert sorted(anp_corpus_config.files()) == [
            CORPUS_DIR / "ANP" / "ANP_1937.csv.gz"
        ]

    @pytest.mark.parametrize(
        "skip_files,expected_size", [(None, 4893), (["ANP_1937.csv.gz"], 0)]
    )
    def test_build_corpus(self, anp_corpus_config, skip_files, expected_size):
        assert (
            len(anp_corpus_config.build_corpus(filter_terms=[], skip_files=skip_files))
            == expected_size
        )

    @pytest.mark.parametrize(
        "skip_files,expected_sizes", [(None, [4893]), (["ANP_1937.csv.gz"], [])]
    )
    def test_build_corpora(self, anp_corpus_config, skip_files, expected_sizes):
        assert [
            len(corpus)
            for corpus in anp_corpus_config.build_corpora(
                filter_terms=[], skip_files=skip_files
            )
        ] == expected_sizes
