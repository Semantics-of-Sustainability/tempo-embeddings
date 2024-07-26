from contextlib import nullcontext as does_not_raise
from pathlib import Path
from tempfile import gettempdir

import pytest

from tempo_embeddings.io.corpus_reader import CorpusConfig, CorpusReader
from tempo_embeddings.settings import CORPUS_DIR


class TestCorpusReader:
    _CONFIGURED_CORPORA = ["ANP_mini", "ANP", "Parool", "NRC", "StatenGeneraal_clean"]

    @pytest.mark.skipif(CORPUS_DIR is None, reason="No corpus directory found")
    @pytest.mark.parametrize(
        "corpora, base_dir, must_exist, expected, expected_exception",
        [
            (None, gettempdir(), False, _CONFIGURED_CORPORA, does_not_raise()),
            (None, gettempdir(), True, [], does_not_raise()),
            (None, CORPUS_DIR, False, _CONFIGURED_CORPORA, does_not_raise()),
            # FIXME: this test depends on local data being present or not:
            (None, CORPUS_DIR, True, ["ANP", "StatenGeneraal_clean"], does_not_raise()),
            (["ANP"], CORPUS_DIR, True, ["ANP"], does_not_raise()),
            (["test corpus"], CORPUS_DIR, True, None, pytest.raises(ValueError)),
        ],
    )
    def test_corpora(self, corpora, base_dir, must_exist, expected, expected_exception):
        with expected_exception:
            reader = CorpusReader(corpora=corpora, base_dir=Path(base_dir))
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
