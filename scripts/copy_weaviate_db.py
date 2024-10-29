import argparse
import logging
from itertools import islice

from tempo_embeddings.embeddings.weaviate_database import WeaviateDatabaseManager

if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        "Copy a Weaviate Vector Database for the Semantics of Sustainability Project"
    )
    parser.add_argument("--corpora", type=str, nargs="+", help="Corpora to copy.")
    parser.add_argument(
        "--overwrite",
        action="store_true",
        help="Overwrite existing corpus in target database.",
    )
    parser.add_argument(
        "--limit",
        type=int,
        required=False,
        help="Maximum number of objects to copy per collection.",
    )

    source_args = parser.add_argument_group("Weaviate export database arguments")
    source_args.add_argument(
        "--source-host",
        type=str,
        default="localhost",
        help="Weaviate server host for source.",
    )
    source_args.add_argument(
        "--source-port", type=int, default=8087, help="Weaviate server port for source."
    )
    source_args.add_argument(
        "--source-api-key",
        type=str,
        required=False,
        help="Weaviate API key for source.",
    )
    source_args.add_argument(
        "--source-ssl",
        action="store_true",
        help="Use SSL for source Weaviate connection.",
    )

    target_args = parser.add_argument_group("Weaviate import database arguments")
    target_args.add_argument(
        "--target-host",
        type=str,
        default="localhost",
        help="Weaviate server host for target.",
    )
    target_args.add_argument(
        "--target-port", type=int, default=8087, help="Weaviate server port for target."
    )
    target_args.add_argument(
        "--target-api-key",
        type=str,
        required=False,
        help="Weaviate API key for target.",
    )
    target_args.add_argument(
        "--target-ssl",
        action="store_true",
        help="Use SSL for target Weaviate connection.",
    )

    args = parser.parse_args()

    if args.source_host == args.target_host:
        # Should we allow for two DBs with different ports on the same host?
        parser.error(
            f"Source host ({args.source_host}) and target host ({args.target_host}) must be different."
        )

    source_db = WeaviateDatabaseManager.from_args(
        model_name=None,
        http_host=args.source_host,
        http_port=args.source_port,
        http_secure=args.source_ssl,
        api_key=args.source_api_key,
    )
    target_db = WeaviateDatabaseManager.from_args(
        model_name=None,
        http_host=args.target_host,
        http_port=args.target_port,
        http_secure=args.target_ssl,
        api_key=args.target_api_key,
    )

    for corpus in args.corpora:
        if args.overwrite and corpus in target_db:
            logging.warning(
                f"Removing existing corpus '{corpus}' in database on {args.target_host}:{args.target_port}"
            )
            target_db.delete_collection(corpus)

        config = source_db.collection_config(corpus)

        if args.limit:
            config["total_count"] = min(config["total_count"], args.limit)
        objects = islice(source_db.collection_objects(corpus), args.limit)

        target_db.import_config(config)
        target_db.import_objects(
            objects, config["corpus"], total_count=config["total_count"]
        )

    target_db.validate_config()
