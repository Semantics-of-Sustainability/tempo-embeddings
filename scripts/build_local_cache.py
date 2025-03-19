import argparse

from tqdm import tqdm

from tempo_embeddings import settings
from tempo_embeddings.embeddings.weaviate_database import WeaviateDatabaseManager
from tempo_embeddings.text.year_span import YearSpan

if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        "A script to build or update the local document frequency cache."
    )

    parser.add_argument(
        "--cache-file",
        type=str,
        default=settings.DOC_FREQUENCY_CACHE_FILE,
        help="The cache file. Suffix .db will be appended.",
    )
    parser.add_argument(
        "--collections", type=str, nargs="+", help="Collections to cache"
    )
    parser.add_argument("--start", type=int, default=1850, help="Start year")
    parser.add_argument("--end", type=int, default=2025, help="End year (exclusive)")

    parser.add_argument(
        "--terms",
        type=str,
        nargs="*",
        help="If given, cache frequencies for documents containing those terms.",
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

    args = parser.parse_args()

    db = WeaviateDatabaseManager.from_args(
        model_name=None,
        http_host=args.weaviate_host,
        http_port=args.weaviate_port,
        http_secure=args.use_ssl,
        api_key=args.api_key,
    )

    for collection in args.collections:
        for year in tqdm(range(args.start, args.end), desc=collection, unit="year"):
            if args.terms:
                for term in args.terms:
                    db.doc_frequency(
                        term, collection, year_span=YearSpan(year, year + 1)
                    )
            else:
                db.doc_frequency("", collection, year_span=YearSpan(year, year + 1))
