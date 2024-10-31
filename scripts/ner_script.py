import argparse
import csv
import sys
from functools import lru_cache

import spacy
from tqdm import tqdm

from tempo_embeddings.io.corpus_reader import CorpusReader


@lru_cache(maxsize=None)
def load_spacy_model(language: str):
    if language == "en":
        return spacy.load("en_core_web_sm")
    elif language == "nl":
        return spacy.load("nl_core_news_sm")
    else:
        raise ValueError(f"No SpaCy model available for language: {language}")


def main(corpora, output_file):
    with open(output_file, mode="w", newline="", encoding="utf-8") as csvfile:
        fieldnames = ["date", "source", "place_name"]
        writer = csv.DictWriter(csvfile, fieldnames=fieldnames)
        writer.writeheader()

        corpus_reader = CorpusReader(corpora=corpora)
        for corpus_name in corpora:
            corpus_config = corpus_reader[corpus_name]
            nlp = load_spacy_model(corpus_config.language)
            for corpus in corpus_config.build_corpora(filter_terms=[]):
                for passage in tqdm(corpus.passages, desc=f"Processing {corpus_name}"):
                    doc = nlp(passage.text)
                    for ent in doc.ents:
                        if (
                            ent.label_ == "GPE"
                        ):  # GPE (Geopolitical Entity) for place names
                            writer.writerow(
                                {
                                    "date": passage.metadata.get("date", "unknown"),
                                    "source": corpus_name,
                                    "place_name": ent.text,
                                }
                            )


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Perform NER on corpora and extract place names."
    )
    parser.add_argument("--corpora", nargs="+", help="List of corpora to process")
    parser.add_argument(
        "--output",
        "-o",
        type=argparse.FileType("w"),
        default=sys.stdout,
        help="Output CSV file",
    )
    args = parser.parse_args()

    main(args.corpora, args.output)
