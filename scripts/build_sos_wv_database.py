import argparse
import csv
from pathlib import Path

from tqdm import tqdm

from tempo_embeddings import settings
from tempo_embeddings.embeddings.model import SentenceTransformerModelWrapper
from tempo_embeddings.embeddings.weaviate_database import WeaviateDatabaseManager
from tempo_embeddings.io.corpus_reader import CorpusReader
from tempo_embeddings.text.corpus import Corpus

csv.field_size_limit(1000000000)


def arguments_parser():
    parser = argparse.ArgumentParser(
        "Build a Vector Database for the Semantics of Sustainability Project"
    )
    parser.add_argument("--corpora", type=str, nargs="+", help="Corpora to process")
    parser.add_argument(
        "--corpus-dir", type=Path, default=settings.CORPUS_DIR, help="Corpora directory"
    )
    parser.add_argument(
        "--max-files",
        type=int,
        required=False,
        help="Maximum number of files to process per corpus. All files if not specified.",
    )

    parser.add_argument("--db-name", type=str, default="testing_db")
    parser.add_argument("--window-size", type=int, default=200)
    parser.add_argument("--batch-size", type=int, default=400)
    parser.add_argument("--use-full-sentences", action="store_true")
    parser.add_argument(
        "--language-model",
        "--lm",
        type=str,
        default=settings.DEFAULT_LANGUAGE_MODEL,
        help=f"The language model to use, defaults to {settings.DEFAULT_LANGUAGE_MODEL}",
    )
    filter_terms_args = parser.add_mutually_exclusive_group(required=True)
    filter_terms_args.add_argument(
        "--filter-terms", type=str, metavar="TERM", nargs="+", help="Filter terms"
    )
    filter_terms_args.add_argument(
        "--filter-terms-file",
        type=argparse.FileType("rt"),
        metavar="FILE",
        help="File with filter terms, one per line",
    )
    # TODO: add options for logging (level, output file)

    return parser


if __name__ == "__main__":
    parser = arguments_parser()
    args = parser.parse_args()

    corpus_reader = CorpusReader(
        config_file=settings.CORPORA_CONFIG_FILE,
        corpora=args.corpora,
        base_dir=args.corpus_dir,
    )

    filter_terms = args.filter_terms or [
        line.strip() for line in args.filter_terms_file
    ]

    # TODO: requires a different wrapper class for non-SentenceBert models
    model = SentenceTransformerModelWrapper.from_pretrained(
        args.language_model, layer=9, accelerate=True
    )

    db = WeaviateDatabaseManager(
        settings.ROOT_DIR / args.db_name,
        embedder_name=args.language_model,
        embedder_config={"type": "default", "model": model},
        batch_size=args.batch_size,
    )

    for corpus_name in tqdm(
        list(corpus_reader.corpora(must_exist=True)), desc="Reading", unit="corpus"
    ):
        collection_name = f"corpus_{corpus_name}"
        corpus_config = corpus_reader[corpus_name]

        corpus: Corpus = corpus_config.build_corpus(
            args.filter_terms, max_files=args.max_files
        )
        if len(corpus) > 0:
            if collection_name not in db.config["existing_collections"]:
                db.create_collection(collection_name, corpus)
            else:
                db.insert_corpus(collection_name, corpus)
