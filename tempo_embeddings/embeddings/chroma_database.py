# pylint:disable=logging-fstring-interpolation
import json
import logging
import platform
from typing import Any, Iterable, Optional

import chromadb
import numpy as np
from chromadb.db.base import UniqueConstraintError
from chromadb.types import Collection
from chromadb.utils import embedding_functions
from transformers import AutoTokenizer

from ..text.corpus import Corpus
from ..text.highlighting import Highlighting
from ..text.passage import Passage
from .vector_database import VectorDatabaseManagerWrapper

logger = logging.getLogger(__name__)


class ChromaDatabaseManager(VectorDatabaseManagerWrapper):
    """A Chroma Database can create N collections and will always use THE SAME embedder function and tokenizer for all collections.
    To create collections with different embedders one needs to create separate databases

    WARNING: This class does not work because the currently used version of ChromaDB is not compatible
    with the current version of protobuf that is required by Weaviate.
    """

    def __init__(
        self,
        db_path: str = "default_db",
        embedder_name: str = None,
        embedder_config: dict[str, Any] = None,
        batch_size: int = 8,
    ):
        super().__init__(batch_size)
        self.db_path = db_path
        self.embedder_name = embedder_name
        self.embedder_config = embedder_config or {"type": "default"}
        self.tokenizer = (
            AutoTokenizer.from_pretrained(embedder_name) if embedder_name else None
        )
        self.model = None
        self.client = None
        if self.embedder_config.get("type") == "hf":
            self.embedding_function = embedding_functions.HuggingFaceEmbeddingFunction(
                api_key=self.embedder_config.get("api_key"), model_name=embedder_name
            )
        elif self.embedder_config.get("type") == "custom_model":
            try:
                self.model = self.embedder_config["model"]
                self.model.batch_size = self.batch_size
            except KeyError:
                logger.error(
                    "If the type is 'custom_model' you should pass the model object under Key 'model'"
                )
            self.embedding_function = None
        elif self.embedder_config.get("type") == "default":
            self.embedding_function = None
        else:
            raise ValueError(f"Malformed embedder_config {self.embedder_config}")

        self.config = {
            "embedder_name": self.embedder_name,
            "embedder_type": self.embedder_config["type"],
            "existing_collections": [],
        }

    def _save_config(self):
        if self.config:
            with open(f"{self.db_path}/config.json", "w", encoding="utf-8") as f:
                json.dump(self.config, f, indent=2)

    def get_available_collections(self):
        if self.config:
            return self.config.get("existing_collections", [])
        return []

    def connect(self):
        if self.client:
            logger.info(
                "A connection to the client already exists in this session: %s",
                self.client,
            )
        else:
            self.client = chromadb.PersistentClient(path=self.db_path)
            # If the given path existed then we will load the records AND override the parameters with the config
            config_path = f"{self.db_path}/config.json"
            try:
                with open(config_path, encoding="utf-8") as f:
                    self.config = json.load(f)
                    logger.info(
                        f"Path '{self.db_path}' already exists. Loading DB configuration:\n{self.config}"
                    )
            except FileNotFoundError:
                logger.info(f"Creating NEW Database in {self.db_path}...")
                self._save_config()

    def get_collection_count(self, collection: Collection):
        return collection.count()

    def create_new_collection(
        self,
        name: str,
        corpus: Corpus = None,
        collection_metadata=None,
    ) -> Optional[Collection]:
        if collection_metadata is None:
            collection_metadata = {"hnsw:space": "cosine"}
        if not self.client:
            raise RuntimeError("Please connect to a valid database first")
        # Create NEW collection and Embeds the given passages. Do nothing otherwise
        try:
            collection = self.client.create_collection(
                name,
                embedding_function=self.embedding_function,
                metadata=collection_metadata,
            )
        except UniqueConstraintError as exc:
            raise ValueError from exc

        self.config["existing_collections"].append(name)
        self.active_collection_name = name
        self._save_config()
        # If the collection is new then insert the corresponding passages already
        if corpus:
            self.ingest(collection, corpus)
        logger.info(f"Created NEW collection '{name}'")
        return collection

    def get_existing_collection(self, name: str) -> Optional[Collection]:
        collection = self.client.get_collection(
            name, embedding_function=self.embedding_function
        )
        self.active_collection_name = name
        logger.info(f"Retrieved existing collection '{name}'")
        return collection

    def delete_collection(self, name):
        try:
            self.client.delete_collection(name)
            self.config["existing_collections"] = [
                c for c in self.config["existing_collections"] if c != name
            ]
            logger.info(f"Succesfully deleted {name}")
        except Exception as e:
            logger.warning(f"delete_collection() caused exception {e}")
            logger.info(f"Collection '{name}' does not exist in this database.")

    def _prepare_insertion_batch(self, batch, embeddings):
        docs, metas, ids = [], [], []
        insert_embeds = []
        for k, p in enumerate(batch):
            pid = p.get_unique_id()
            # Save in the DB as "pre-tokenized" to avoid using more space
            docs.append(" ".join(p.words()))
            p.metadata["highlighting"] = str(p.highlighting)
            metas.append(p.metadata)
            ids.append(pid)
            if embeddings is not None:
                insert_embeds.append(embeddings[k])
        return docs, metas, ids, insert_embeds

    def ingest(self, collection: Collection, corpus: Corpus):
        if len(corpus.passages) == 0:
            raise ValueError("There should be at least one passage to insert.")

        passages_need_embeddings = corpus.passages[0].embedding is None
        num_records = collection.count()

        if passages_need_embeddings and not self.model and not self.embedding_function:
            raise RuntimeError(
                "These passages need embeddings but no valid model or embedding function was provided to the Database Object"
            )

        for batch_pass in corpus.batches(self.batch_size):
            if passages_need_embeddings and self.model:
                batch_embeds = self.model.embed_corpus(
                    Corpus(batch_pass), store_tokenizations=True, batch_size=1
                )
                logger.debug(f"Batch pass...{type(batch_pass)} | {type(batch_embeds)}")
                logger.debug(f"NODE: {platform.node()}")
                embeddings = [tensor.tolist() for tensor in batch_embeds][0]
            else:
                embeddings = None

            docs, metas, ids, insert_embeds = self._prepare_insertion_batch(
                batch_pass, embeddings
            )

            if len(insert_embeds) > 0:
                collection.add(
                    documents=docs, metadatas=metas, ids=ids, embeddings=insert_embeds
                )
                logger.debug(f"Inserting {len(docs)} records with custom embeddings")
            else:
                collection.add(documents=docs, metadatas=metas, ids=ids)
                logger.debug(
                    f"Inserting {len(docs)} records, database engine will compute embeddings"
                )

        new_count = collection.count()
        logger.info(
            f"Added {new_count - num_records} new documents. Total = {new_count}"
        )

    def _build_filter_text_expression(self, filter_words):
        if filter_words is None:
            filter_words = []
        if len(filter_words) == 0:
            where_doc = None
        elif len(filter_words) == 1:
            where_doc = {"$contains": filter_words[0]}
        else:
            where_doc = {"$and": [{"$contains": w} for w in filter_words]}
        return where_doc

    def _create_passage_from_record(self, rec_id, doc, meta):
        # Get Highlighting
        highlighting = None
        hl = meta["highlighting"]
        if "_" in hl:
            start, end = [int(x) for x in hl.split("_")]
            highlighting = Highlighting(start, end)
            # filter_terms.add(doc[start:end])
        # Create Passage
        p = Passage(
            doc, meta, highlighting, unique_id=rec_id
        )  # meta["full_word_spans"], meta["char2tokens"]
        p.tokenized_text = doc.split()
        return p

    def get_corpus(
        self,
        collection: Collection,
        filter_words: list[str] = None,
        where_obj: dict[str, Any] = None,
        limit: int = 0,
        include_embeddings: bool = False,
    ) -> Corpus:
        # pylint: disable=too-many-arguments
        # Result OBJ has these keys: dict_keys(['ids', 'embeddings', 'metadatas', 'documents', 'uris', 'data'])
        # by default only "metadatas" and "documents" are populated
        include = (
            ["metadatas", "documents", "embeddings"]
            if include_embeddings
            else ["metadatas", "documents"]
        )

        # Build the WHERE_DOCUMENT Object
        where_doc = self._build_filter_text_expression(filter_words)

        # Query the collection
        if limit == 0:
            limit = None
        records = collection.get(
            where=where_obj, where_document=where_doc, include=include, limit=limit
        )
        # Return empty corpus if no records where found
        if len(records) > 0:
            if "embeddings" not in include:
                records["embeddings"] = []
            # Build Passage objects
            passages = []
            for ix, rec_id in enumerate(records["ids"]):
                doc = records["documents"][ix]
                meta = records["metadatas"][ix]
                passage = self._create_passage_from_record(rec_id, doc, meta)
                # Assign Embedding if requested
                if "embeddings" in include:
                    passage.embedding = records["embeddings"][ix]
                passages.append(passage)
            return Corpus(
                passages, label="; ".join(filter_words) if filter_words else None
            )
        return Corpus()

    def query_vector_neighbors(
        self, collection: Collection, vector: list[float], k_neighbors=10
    ) -> Iterable[tuple[Passage, float]]:
        include = ["metadatas", "documents", "embeddings", "distances"]

        result = collection.query(
            query_embeddings=[vector], n_results=k_neighbors, include=include
        )

        for rec_id, doc, meta, emb, dist in zip(
            result["ids"][0],
            result["documents"][0],
            result["metadatas"][0],
            result["embeddings"][0],
            result["distances"][0],
        ):
            yield (Passage(doc, meta, embedding=emb, unique_id=rec_id), dist)

    def query_text_neighbors(
        self,
        collection: Collection,
        text: str,
        k_neighbors=10,
        include: list[str] = None,
    ):
        if include is None:
            include = ["metadatas", "documents", "embeddings", "distances"]

        if not self.embedding_function:
            logger.warning("There is no embedding function defined in this database")
            return None
        result = collection.query(
            query_texts=[text], n_results=k_neighbors, include=include
        )

        res_dict = {
            "ids": result.get("ids", [[]])[0],
            "metadatas": result.get("metadatas", [[]])[0],
            "documents": result.get("documents", [[]])[0],
            "embeddings": []
            if "embeddings" not in result
            else np.array(result["embeddings"][0], dtype=np.float32),
        }
        if "distances" in result:
            res_dict["distances"] = result["distances"][0]
        return res_dict

    def load_tokenizer(self, tokenizer_name):
        if tokenizer_name != self.config["embedder_name"]:
            logger.warning(
                f"The original database used '{self.config['embedder_name']}' tokenizer and you are loading '{tokenizer_name}'"
            )
        self.tokenizer = AutoTokenizer.from_pretrained(tokenizer_name)

    def _tokenize(self, sentence: str):
        if self.tokenizer is None:
            raise AttributeError(
                "This Database Client does not have a tokenizer. Run load_tokenizer() first!"
            )
        encoded_input = self.tokenizer(sentence, return_tensors="pt")
        return encoded_input

    def embed_text_batch(self, text_batch: list[str]):
        batch_embeddings = None
        if self.embedding_function:
            batch_embeddings = self.embedding_function(text_batch)
        else:
            logger.warning(
                "There is no valid embedding function in this database. Returning None"
            )
        return batch_embeddings

    def get_metadata_stats(
        self, collection: Collection, include_only: list[str] = None
    ) -> dict[dict[str, int]]:
        result = collection.get(include=["metadatas"])
        stats = {}
        for meta_dict in result["metadatas"]:
            for field_name, value in meta_dict.items():
                if include_only is None or field_name in include_only:
                    if field_name in stats:
                        stats[field_name][value] = stats[field_name].get(value, 0) + 1
                    else:
                        stats[field_name] = {value: 1}
        return stats

    def is_in_collection(self, collection: Collection, text: str):
        results = collection.get(
            where_document={"$contains": text}, include=["documents"]
        )
        for ret_doc in results["documents"]:
            if len(text) == len(ret_doc):
                return True
        return False

    def get_vector_from_db(self, collection: Collection, text: str):
        results = collection.get(
            where_document={"$contains": text}, include=["documents", "embeddings"]
        )
        if len(results) == 0:
            return None
        for ix, ret_doc in enumerate(results["documents"]):
            if len(text) == len(ret_doc):
                return results["embeddings"][ix]
        return None
