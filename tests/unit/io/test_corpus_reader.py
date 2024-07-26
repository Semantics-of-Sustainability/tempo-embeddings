from pathlib import Path
from tempfile import gettempdir

import pytest

from tempo_embeddings.io.corpus_reader import CorpusConfig, CorpusReader
from tempo_embeddings.settings import CORPUS_DIR


class TestCorpusReader:
    _CONFIGURED_CORPORA = ["ANP_mini", "ANP", "Parool", "NRC", "StatenGeneraal_clean"]

    @pytest.mark.skipif(CORPUS_DIR is None, reason="No corpus directory found")
    @pytest.mark.parametrize(
        "base_dir, must_exist, expected",
        [
            (gettempdir(), False, _CONFIGURED_CORPORA),
            (gettempdir(), True, []),
            (CORPUS_DIR, False, _CONFIGURED_CORPORA),
            # FIXME: this test depends on local data being present or not:
            (CORPUS_DIR, True, ["ANP", "StatenGeneraal_clean"]),
        ],
    )
    def test_corpora(self, base_dir, must_exist, expected):
        reader = CorpusReader(base_dir=Path(base_dir))
        assert sorted(reader.corpora(must_exist=must_exist)) == sorted(
            expected
        ), f"Expected corpora not found in {base_dir}"


class TestCorpusConfig:
    @pytest.fixture
    def corpus_config(self, tmp_path):
        return CorpusConfig(directory=tmp_path / "test_corpus")

    def test_exists(self, corpus_config):
        assert not corpus_config.exists()

        corpus_config.directory.mkdir()
        assert corpus_config.exists()

    def test_files(self, corpus_config):
        assert sorted(corpus_config.files()) == []

        corpus_config.directory.mkdir()
        (corpus_config.directory / "file1.txt").touch()
        (corpus_config.directory / "file2.csv").touch()
        (corpus_config.directory / "file2_1984.csv").touch()

        assert sorted(corpus_config.files()) == [
            corpus_config.directory / "file2_1984.csv"
        ]

    # TODO: test CorpusConfig.build_corpus()
