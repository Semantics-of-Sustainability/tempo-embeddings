import argparse

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
    weaviate_args.add_argument("--use-ssl", action="store_true", help="Use SSL.")
    weaviate_args.add_argument(
        "--api-key", type=str, required=False, help="Weaviate API key"
    )

    args = parser.parse_args()

    db = WeaviateDatabaseManager.from_args(
        model_name=DEFAULT_LANGUAGE_MODEL,
        http_host=args.weaviate_host,
        http_port=args.weaviate_port,
        use_ssl=args.use_ssl,
        api_key=args.api_key,
    )
    db.export_from_collection(args.corpus, args.output)

    args.output.close()
