import argparse
import logging

import weaviate
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

    args = parser.parse_args()

    if args.source_host == args.target_host:
        # Should we allow for two DBs with different ports on the same host?
        parser.error(
            f"Source host ({args.source_host}) and target host ({args.target_host}) must be different."
        )

    with weaviate.connect_to_local(
        args.source_host, args.source_port
    ) as source_client, weaviate.connect_to_local(
        args.target_host, args.target_port
    ) as target_client:
        source_db = WeaviateDatabaseManager(client=source_client, model=None)
        target_db = WeaviateDatabaseManager(client=target_client, model=None)

        for corpus in args.corpora:
            if args.overwrite and corpus in target_db:
                logging.warning(
                    f"Removing existing corpus '{corpus}' in database on {args.target_host}:{args.target_port}"
                )
                target_db.delete_collection(corpus)

            config = source_db.collection_config(corpus)
            target_db.import_config(config)
            target_db.import_objects(
                source_db.collection_objects(corpus),
                config["corpus"],
                total_count=config["total_count"],
            )

        target_db.validate_config()
