import argparse
import csv
import logging
from pathlib import Path

from tqdm import tqdm

import weaviate
from tempo_embeddings import settings
from tempo_embeddings.embeddings.model import SentenceTransformerModelWrapper
from tempo_embeddings.embeddings.weaviate_database import WeaviateDatabaseManager
from tempo_embeddings.io.corpus_reader import CorpusReader

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
    parser.add_argument(
        "--reset-db", action="store_true", help="Reset the database, delete all data"
    )

    parser.add_argument("--window-size", type=int, default=200)
    parser.add_argument("--batch-size", type=int, default=64)
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

    weaviate_args = parser.add_argument_group("Weaviate arguments")
    weaviate_args.add_argument(
        "--weaviate-host",
        "--host",
        type=str,
        default="localhost",
        help="Weaviate server host",
    )
    weaviate_args.add_argument(
        "--weaviate-port",
        "--port",
        type=int,
        default=8087,
        help="Weaviate server port, defaults to 8087 to match the setting in docker-compose.yml",
    )
    # TODO: add options for logging (level, output file)

    return parser


if __name__ == "__main__":
    parser = arguments_parser()
    args = parser.parse_args()

    try:
        corpus_reader = CorpusReader(
            config_file=settings.CORPORA_CONFIG_FILE,
            corpora=args.corpora,
            base_dir=args.corpus_dir,
        )
    except ValueError as e:
        parser.error(e)

    filter_terms = args.filter_terms or [
        line.strip() for line in args.filter_terms_file
    ]

    with weaviate.connect_to_local(args.weaviate_host, args.weaviate_port) as client:
        # TODO: model requires a different wrapper class for non-SentenceBert models
        db = WeaviateDatabaseManager(
            client=client,
            model=SentenceTransformerModelWrapper.from_pretrained(args.language_model),
            batch_size=args.batch_size,
        )
        db.validate_config()

        if args.reset_db:
            db.reset()

        for corpus_name in tqdm(
            list(corpus_reader.corpora(must_exist=True)), desc="Reading", unit="corpus"
        ):
            ingested_files = set(db.provenances(corpus_name))
            logging.info(f"Skipping {len(ingested_files)} files for '{corpus_name}'.")

            for corpus in corpus_reader[corpus_name].build_corpora(
                filter_terms, skip_files=ingested_files, max_files=args.max_files
            ):
                db.ingest(corpus, corpus_name)

    if args.filter_terms_file is not None:
        args.filter_terms_file.close()
