# pylint: disable=logging-fstring-interpolation
import json
import logging
from abc import ABC
from abc import abstractmethod
from typing import Any
from typing import Iterable
from typing import Optional
import chromadb
import numpy as np
from chromadb.db.base import UniqueConstraintError
from chromadb.types import Collection
from chromadb.utils import embedding_functions
from numpy.typing import ArrayLike
from transformers import AutoTokenizer
from umap.umap_ import UMAP
from ..text.passage import Passage
from ..text.corpus import Corpus
from ..text.highlighting import Highlighting


class VectorDatabaseManagerWrapper(ABC):
    """A Wrapper for different Vector Databases"""

    def __init__(self, batch_size: int):
        """Constructor.

        Args:
            batch_size: The batch size to process records
        """
        self._batch_size = batch_size

    @property
    def batch_size(self) -> int:
        return self._batch_size

    @batch_size.setter
    def batch_size(self, batch_size: int) -> None:
        self._batch_size = batch_size

    # def _batches(self, passages: list[Passage]) -> Iterable[list[Passage]]:
    #     for batch_start in tqdm(
    #         range(0, len(passages), self.batch_size),
    #         desc="Embeddings Batches",
    #         unit="batch",
    #         total=len(passages) // self.batch_size + 1,
    #     ):
    #         yield passages[batch_start : batch_start + self.batch_size]

    @abstractmethod
    def connect(self):
        return NotImplemented

    @abstractmethod
    def insert_corpus(
        self, collection: Collection, corpus: Corpus
    ):
        return NotImplemented

    @abstractmethod
    def retrieve_vectors_if_exist(
        self, collection: Collection, passages: list[Passage]
    ):
        return NotImplemented


class ChromaDatabaseManager(VectorDatabaseManagerWrapper):
    """A Chroma Database can create N collections and will always use THE SAME embedder function and tokenizer for all collections.
    To create collections with different embedders one needs to create separate databases
    """

    def __init__(
        self,
        db_path: str = "default_db",
        embedder_name: str = "sentence-transformers/all-MiniLM-L6-v2",
        embedder_config: dict[str, Any] = None,
        batch_size: int = 8,
    ):
        super().__init__(batch_size)
        self.db_path = db_path
        self.embedder_name = embedder_name
        self.embedder_config = embedder_config or {"type": "default"}
        self.tokenizer = AutoTokenizer.from_pretrained(embedder_name)
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
                logging.error("If the type is 'custom_model' you should pass the model object under Key 'model'")
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
            logging.info(
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
                    logging.info(
                        f"Path '{self.db_path}' already exists. Loading DB configuration:\n{self.config}"
                    )
            except FileNotFoundError:
                logging.info(f"Creating NEW Database in {self.db_path}...")
                self._save_config()

    def create_new_collection(
        self,
        name: str,
        corpus: Corpus = None,
        collection_metadata = None,
    ) -> Optional[Collection]:
        if collection_metadata is None:
            collection_metadata = {"hnsw:space": "cosine"}
        if not self.client:
            raise RuntimeError("Please connect to a valid database first")
        # Create NEW collection and Embeds the given passages. Do nothing otherwise
        try:
            collection = self.client.create_collection(
                name, embedding_function=self.embedding_function, metadata=collection_metadata
            )
        except UniqueConstraintError as exc:
            raise ValueError from exc
        
        self.config["existing_collections"].append(name)
        self.active_collection_name = name
        self._save_config()
        # If the collection is new then insert the corresponding passages already
        if corpus:
            self.insert_corpus(collection, corpus)
        print(f"Created NEW collection '{name}'")
        return collection
        

    def get_existing_collection(self, name: str) -> Optional[Collection]:
        collection = self.client.get_collection(
            name, embedding_function=self.embedding_function
        )
        self.active_collection_name = name
        print(f"Retrieved existing collection '{name}'")
        return collection

    def delete_collection(self, name):
        try:
            self.client.delete_collection(name)
            self.config["existing_collections"] = [c for c in self.config["existing_collections"] if c != name]
            logging.info(f"Succesfully deleted {name}")
        except Exception as e:
            logging.warning(f"delete_collection() caused exception {e}")
            logging.info(f"Collection '{name}' does not exist in this database.")

    def _prepare_insertion_batch(self, batch, embeddings, seen_ids):
        docs, metas, ids = [], [], []
        insert_embeds = []
        for k, p in enumerate(batch):
            pid = p.get_unique_id()
            if pid not in seen_ids:
                docs.append(p.text)
                p.metadata["tokenized_text"] = " ".join(p.words())
                p.metadata["highlighting"] = str(p.highlighting)
                metas.append(p.metadata)
                ids.append(pid)
                seen_ids.add(pid)
                if embeddings:
                    p.embedding = embeddings[k]
                if p.embedding is not None:
                    insert_embeds.append(p.embedding)
        return docs, metas, ids, insert_embeds

    def insert_corpus(
        self,
        collection: Collection,
        corpus: Corpus
    ):

        if len(corpus.passages) == 0:
            raise ValueError("There should be at least one passage to insert.") 

        passages_need_embeddings = corpus.passages[0].embedding is None
        num_records = collection.count()
        seen_ids = set()

        for batch_pass in corpus.batches(self.batch_size):
            if passages_need_embeddings and self.model:
                embedded_tensors = self.model.embed_corpus(Corpus(batch_pass), store_tokenizations=True, batch_size=1) 
                embeddings = [tensor.tolist() for tensor in embedded_tensors][0]
            elif passages_need_embeddings and not self.model and not self.embedding_function:
                raise RuntimeError("These passages need embeddings but no valid model or embedding function was provided to the Database Object")
            else: 
                embeddings = None

            docs, metas, ids, insert_embeds = self._prepare_insertion_batch(batch_pass, embeddings, seen_ids)

            if len(insert_embeds) > 0:
                collection.add(documents=docs, metadatas=metas, ids=ids, embeddings=insert_embeds)
            else:
                collection.add(documents=docs, metadatas=metas, ids=ids)

        new_count = collection.count()
        print(f"Added {new_count - num_records} new documents. Total = {new_count}")

    def compress_embeddings(
        self,
        corpus: Corpus,
        umap_verbose: bool = True,
        **umap_args,
    ) -> ArrayLike:

        if len(corpus) == 0:
            return None

        umap = UMAP(verbose=umap_verbose, **umap_args)
        compressed = umap.fit_transform([p.embedding for p in corpus.passages])

        return compressed

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
        p = Passage(doc, meta, highlighting, unique_id=rec_id) # meta["full_word_spans"], meta["char2tokens"]
        # Assign Tokenized Text
        p.tokenized_text = meta["tokenized_text"].split()
        return p


    def get_corpus(
        self,
        collection: Collection,
        filter_words: list[str] = None,
        where_obj: dict[str, Any] = None,
        limit: int = 0,
        include_embeddings: bool = False
    ) -> Corpus:
        # pylint: disable=too-many-arguments
        # Result OBJ has these keys: dict_keys(['ids', 'embeddings', 'metadatas', 'documents', 'uris', 'data'])
        # by default only "metadatas" and "documents" are populated
        include = ["metadatas", "documents", "embeddings"] if include_embeddings else ["metadatas", "documents"]

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
            return Corpus(passages, label="; ".join(filter_words) if filter_words else None)
        return Corpus()

    def query_vector_neighbors(
        self,
        collection: Collection,
        vector: list[float],
        k_neighbors=10
    ) -> Iterable[tuple[Passage, float]]:
        include = ["metadatas", "documents", "embeddings", "distances"]

        result = collection.query(
            query_embeddings=[vector], n_results=k_neighbors, include=include
        )

        for rec_id, doc, meta, emb, dist in zip(result["ids"][0], result["documents"][0], result["metadatas"][0], 
                                                result["embeddings"][0], result["distances"][0]):
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
            print("WARN: There is no embedding function defined in this database")
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

    def _tokenize(self, sentence: str):
        encoded_input = self.tokenizer(sentence, return_tensors="pt")
        return encoded_input

    def embed_text_batch(self, text_batch: list[str]):
        batch_embeddings = None
        if self.embedding_function:
            batch_embeddings = self.embedding_function(text_batch)
        else:
            logging.warning("There is no valid embedding function in this database. Returning None")
        return batch_embeddings

    def is_in_collection(self, collection: Collection, text: str):
        results = collection.get(
            where_document={"$contains": text}, include=["documents"]
        )
        for ret_doc in results["documents"]:
            if len(text) == len(ret_doc):
                return True
        return False

        # ## This at the bottom does NOT work because the distance is VERY small but actually not 0
        # result = collection.query(
        #     query_texts=[text],
        #     n_results=1,
        #     include=["distances"]
        # )
        # if len(result) == 0: return False
        # print(result["distances"][0])
        # if result["distances"][0] == 0:
        #     return True
        # else:
        #     return False

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

    def retrieve_vectors_if_exist(
        self, collection: Collection, passages: list[Passage]
    ) -> list[float]:
        # Lookup by unique ID and retrieve the vectors if exist in the DB
        # Even if a single vector in the batch is missing then we will compute the full batch again (to avoid confusions)
        # response = {"ids": []}  # When we have uniqueID's we will query here...
        # retrieved_embeddings = []
        # if len(response["ids"]) == len(passages):
        #     retrieved_embeddings = response["embeddings"]
        # return retrieved_embeddings
        raise NotImplementedError
