import logging
import os
import platform
from pathlib import Path
from typing import Optional

DEFAULT_ENCODING = os.environ.get("ENCODING", "utf-8")

OUTLIERS_LABEL: str = "Outliers"

CWD = Path(__file__).parent.absolute()
ROOT_DIR: Path = CWD.parent
DATA_DIR: Path = CWD / "data"
if not DATA_DIR.is_dir():
    logging.error(f"Data directory '{DATA_DIR}' not found.")

STOPWORDS_FILE: Path = DATA_DIR / "stopwords-filter-nl.txt"
assert STOPWORDS_FILE.is_file(), f"Stopwords file '{STOPWORDS_FILE}' not found."

STOPWORDS: set[str] = frozenset(STOPWORDS_FILE.read_text().splitlines())

CORPORA_CONFIG_FILE: Path = DATA_DIR / "corpora.json"
if not (CORPORA_CONFIG_FILE.is_file()):
    logging.error(f"Corpora config file '{CORPORA_CONFIG_FILE}' not found.")


_CORPUS_DIRS: list[Path] = [
    # Local dirs:
    Path.home() / "Documents" / "SemanticsOfSustainability" / "data" / "Joris",
    Path.home() / "SEED_DATA" / "SemanticsSustainability",
    # Research Cloud:
    Path(
        "/data/datasets/research-semantics-of-sustainability/semantics-of-sustainability/data/"
    ),
    # Snellius:
    Path().home() / "data",
    # Yoda drive mounted on MacOS:
    Path(
        "/Volumes/i-lab.data.uu.nl/research-semantics-of-sustainability/semantics-of-sustainability/data"
    ),
]
"""Directories in which corpora are stored; the first one found is used."""

try:
    CORPUS_DIR: Path = next(dir for dir in _CORPUS_DIRS if dir.is_dir())
    """The base directory in which corpora are stored."""
    print(f"Using corpus directory: '{CORPUS_DIR}'")
except StopIteration:
    logging.error(f"No corpus directory found in {_CORPUS_DIRS}")
    CORPUS_DIR = Path(".")

DEFAULT_LANGUAGE_MODEL: str = (
    "NetherlandsForensicInstitute/robbert-2022-dutch-sentence-transformers"
)

### Segmentation settings
SEGMENTER: str = os.environ.get("SEGMENTER", "sentence_splitter")
WTPSPLIT_MODEL = os.environ.get("WTPSPLIT_MODEL", "sat-3l-sm")

# ~4 * max model input size (in tokens)
PASSAGE_LENGTH = int(os.environ.get("PASSAGE_LENGTH", 2048))

DEVICE: Optional[str] = os.environ.get("DEVICE")

WEAVIATE_CONFIG_COLLECTION: str = "TempoEmbeddings"
WEAVIATE_SERVERS = [
    ("UU", ("semantics-of-sustainability.hum.uu.nl", 443, True)),
    ("Research Cloud", ("145.38.187.187", 8087, False)),
    ("local", ("localhost", 8087, False)),
]
"""Values provide a tuple (host,port,use SSL)"""

WEAVIATE_API_KEY: str = os.environ.get("WEAVIATE_API_KEY", None)

STRICT = {"strict": True} if int(platform.python_version_tuple()[1]) >= 10 else {}
"""Optional argument for zip() to enforce strict mode in Python 3.10+."""
