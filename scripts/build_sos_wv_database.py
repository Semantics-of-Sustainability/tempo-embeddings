import argparse
import csv
from pathlib import Path

from tqdm import tqdm

from tempo_embeddings import settings
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
        "--corpus-config-file",
        type=argparse.FileType("rt"),
        default=settings.CORPORA_CONFIG_FILE,
        help="Corpora configuration file",
    )
    parser.add_argument(
        "--max-files",
        type=int,
        required=False,
        help="Maximum number of files to process per corpus. All files if not specified.",
    )
    parser.add_argument(
        "--skip-existing",
        action="store_true",
        help="Skip existing files, based on the 'provenance' metadata field.",
    )

    overwrite_args = parser.add_mutually_exclusive_group(required=False)
    overwrite_args.add_argument(
        "--overwrite",
        action="store_true",
        help="Overwrite collections that are already in the database, delete all their data.",
    )
    overwrite_args.add_argument(
        "--reset-db", action="store_true", help="Reset the database, delete all data"
    )

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
    weaviate_args.add_argument(
        "--grpc-port", type=int, default=50051, help="Weaviate gRPC port"
    )
    weaviate_args.add_argument("--use-ssl", action="store_true", help="Use SSL.")
    weaviate_args.add_argument(
        "--api-key", type=str, required=False, help="Weaviate API key"
    )
    # TODO: add options for logging (level, output file)

    return parser


if __name__ == "__main__":
    parser = arguments_parser()
    args = parser.parse_args()

    try:
        corpus_reader = CorpusReader(
            config_file=args.corpus_config_file,
            corpora=args.corpora,
            base_dir=args.corpus_dir,
        )
    except ValueError as e:
        parser.error(e)

    corpora = list(corpus_reader.corpora(must_exist=True))

    for corpus in args.corpora or []:
        if corpus not in corpora:
            parser.error(f"Corpus '{corpus}' not found in the configuration file.")

    # TODO: ignore comment lines (startwith('#'))?
    filter_terms = args.filter_terms or [
        line.strip() for line in args.filter_terms_file if line.strip()
    ]

    # TODO: model requires a different wrapper class for non-SentenceBert models
    db = WeaviateDatabaseManager.from_args(
        model_name=args.language_model,
        http_host=args.weaviate_host,
        http_port=args.weaviate_port,
        http_secure=args.use_ssl,
        api_key=args.api_key,
        batch_size=args.batch_size,
    )

    if args.reset_db:
        db.reset()

    db.validate_config()

    for corpus_name in tqdm(corpora, desc="Reading", unit="corpus"):
        if args.overwrite:
            db.delete_collection(corpus_name)

        ingested_files = (
            set(db.get_metadata_values(corpus_name, "provenance"))
            if args.skip_existing
            else set()
        )

        corpus_config = corpus_reader[corpus_name]
        for corpus in corpus_config.build_corpora(
            filter_terms, skip_files=ingested_files, max_files=args.max_files
        ):
            db.ingest(
                corpus,
                corpus_name,
                embedder=args.language_model,
                properties=corpus_config.asdict(properties=["language", "segmenter"]),
            )

    if args.filter_terms_file is not None:
        args.filter_terms_file.close()
