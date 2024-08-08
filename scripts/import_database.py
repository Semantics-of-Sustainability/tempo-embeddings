import argparse

import weaviate
from tempo_embeddings.embeddings.weaviate_database import WeaviateDatabaseManager

if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        "Import a Weaviate Vector Database for the Semantics of Sustainability Project"
    )
    parser.add_argument("--corpus", type=str, required=True, help="Corpus to export")
    parser.add_argument(
        "--input",
        "-i",
        type=argparse.FileType("rb"),
        required=True,
        help="Input file containing a previously exported database",
    )
    parser.add_argument(
        "--overwrite", action="store_true", help="Overwrite existing corpus"
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

    args = parser.parse_args()

    with weaviate.connect_to_local(args.weaviate_host, args.weaviate_port) as client:
        db = WeaviateDatabaseManager(client=client, model=None)
        if args.overwrite:
            db.delete_collection(args.corpus)
        db.import_into_collection(args.input, args.corpus)

    args.input.close()
