import argparse
import csv
import logging
import sys
from functools import lru_cache
from pathlib import Path

import spacy
import spacy.cli
from spacy.language import Language
from tqdm import tqdm

from tempo_embeddings.io.corpus_reader import CorpusReader

MODEL_NAMES: dict[str, str] = {"en": "en_core_web_sm", "nl": "nl_core_news_lg"}


@lru_cache(maxsize=None)
def load_spacy_model(language: str, *, download: bool = True) -> Language:
    """Load SpaCy model for a given language.

    Args:
        language (str): Language code.
        download (bool): Whether to download the model if not available.
    Raises:
        ValueError: If no model is available for the given language.
        OSError: If the model cannot be loaded and 'download' is False.
    """

    try:
        model_name = MODEL_NAMES[language]
        model: Language = spacy.load(model_name)
    except KeyError as e:
        raise ValueError(
            f"No SpaCy model available for language '{language}'. Available languages are: {list(MODEL_NAMES.keys())}"
        ) from e
    except OSError as e:
        if download:
            logging.warning(
                f"Failed to load Spacy model for language '{language}': '{e}. '{e}'. Downloading and re-trying."
            )
            spacy.cli.download(model_name)

            # retry loading the model, but don't retry downloading:
            model = load_spacy_model(language, download=False)
        else:
            raise RuntimeError(e)
    return model


def extract_years_from_csv(csvfile: Path):
    years = set()
    with csvfile.open(mode="r", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        for row in reader:
            date = row["date"]
            if date != "unknown":
                year = date.split("-")[0]
                years.add(year)
    return years


def main(corpora, csvfile: Path, resume: bool):
    file_exists = csvfile.exists()

    if resume and file_exists:
        years_to_skip = extract_years_from_csv(csvfile)
        logging.info(f"Skipping years: {years_to_skip}")
    else:
        years_to_skip = set()

    fieldnames = ["date", "source", "place_name"]
    with csvfile.open(mode="a", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)

        if not file_exists:
            writer.writeheader()

        corpus_reader = CorpusReader(corpora=corpora)

        for corpus_name in corpora:
            corpus_config = corpus_reader[corpus_name]
            nlp = load_spacy_model(corpus_config.language)

            skip_files: set[str] = {
                file.name
                for file in corpus_config.files()
                if any(year in file.name for year in years_to_skip)
            }
            logging.debug(f"Skipping files: {skip_files}")

            for corpus in corpus_config.build_corpora(
                filter_terms=[], skip_files=skip_files
            ):
                try:
                    provenance = corpus.passages[0].metadata.get("provenance")
                except IndexError:
                    logging.warning(f"Empty corpus: {corpus_name}")
                    continue
                rows = [
                    {
                        "date": passage.metadata["date"],
                        "source": corpus_name,
                        "place_name": ent.text,
                    }
                    for passage in tqdm(
                        corpus.passages, desc=provenance, unit="passage"
                    )
                    for ent in nlp(passage.text).ents
                    if ent.label_ == "GPE"
                ]
                writer.writerows(rows)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Perform NER on corpora and extract place names."
    )
    parser.add_argument("--corpora", nargs="+", help="List of corpora to process")
    parser.add_argument(
        "--output",
        "-o",
        type=Path,
        default=Path(sys.stdout.name),
        help="Output CSV file",
    )
    parser.add_argument(
        "--resume",
        action="store_true",
        help="Resume from the last run by reading the existing output file",
    )
    args = parser.parse_args()

    if not args.resume and args.output.exists():
        parser.error(f"Output file already exists: {args.output}")

    main(args.corpora, args.output, args.resume)
