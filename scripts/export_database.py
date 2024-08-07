import argparse

import weaviate
from tempo_embeddings.embeddings.weaviate_database import WeaviateDatabaseManager
from tempo_embeddings.settings import DEFAULT_LANGUAGE_MODEL

if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        "Export a Weaviate Vector Database for the Semantics of Sustainability Project"
    )
    parser.add_argument("--corpus", type=str, required=True, help="Corpus to export")
    parser.add_argument(
        "--output",
        "-o",
        type=argparse.FileType("xb"),
        required=True,
        help="Output file for the exported database",
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
        db = WeaviateDatabaseManager(client=client, model=DEFAULT_LANGUAGE_MODEL)
        db.export_from_collection(args.corpus, args.output)

    args.output.close()
