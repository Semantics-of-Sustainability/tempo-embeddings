import logging
import os
from pathlib import Path

DEFAULT_ENCODING = os.environ.get("ENCODING", "utf-8")

OUTLIERS_LABEL: str = "Outliers"

CWD = Path(__file__).parent.absolute()
ROOT_DIR: Path = CWD.parent

CORPORA_CONFIG_FILE: Path = CWD / "corpora.json"
assert (
    CORPORA_CONFIG_FILE.is_file()
), f"Corpora config file '{CORPORA_CONFIG_FILE}' not found."


_CORPUS_DIRS: list[Path] = [
    # Local dirs:
    Path.home() / "Documents" / "SemanticsOfSustainability" / "data" / "Joris",
    Path.home() / "SEED_DATA" / "SemanticsSustainability",
    # Research Cloud:
    Path("/data/volume_2/data"),
    Path("/data/storage-semantics-of-sustainability/data/"),
    # Snellius:
    Path("/home/cschnober/data/"),
    # Yoda drive mounted on MacOS:
    Path(
        "/Volumes/i-lab.data.uu.nl/research-semantics-of-sustainability/semantics-of-sustainability/data"
    ),
]
"""Directories in which corpora are stored; the first one found is used."""

try:
    CORPUS_DIR: Path = next(dir for dir in _CORPUS_DIRS if dir.is_dir())
    """The base directory in which corpora are stored."""
except StopIteration:
    logging.error(f"No corpus directory found in {_CORPUS_DIRS}")
    CORPUS_DIR = None

DEFAULT_LANGUAGE_MODEL: str = (
    "NetherlandsForensicInstitute/robbert-2022-dutch-sentence-transformers"
)
