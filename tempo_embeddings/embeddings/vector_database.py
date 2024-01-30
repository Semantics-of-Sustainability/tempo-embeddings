import os, json
import shutil
from abc import ABC, abstractmethod
import numpy as np
from tqdm import tqdm
from typing import Iterable, List, Callable, Dict, Any, Optional
from typing import TYPE_CHECKING
from numpy.typing import ArrayLike

from ..text.passage import Passage
if TYPE_CHECKING:
    from ..text.corpus import Corpus

from transformers import AutoTokenizer

import chromadb
import chromadb.utils.embedding_functions as embedding_functions
from chromadb.types import Collection
from chromadb.db.base import UniqueConstraintError

from umap.umap_ import UMAP


class VectorDatabaseManagerWrapper(ABC):
    """A Wrapper for different Vector Databases"""

    def __init__(
        self,
        batch_size: int
    ):
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


    def _batches(self, passages: list[Passage]) -> Iterable[list[Passage]]:
        for batch_start in tqdm(
            range(0, len(passages), self.batch_size),
            desc="Embeddings Batches",
            unit="batch",
            total=len(passages) // self.batch_size + 1,
        ):
            yield passages[batch_start : batch_start + self.batch_size]

    @abstractmethod
    def connect(self):
        pass

    @abstractmethod
    def insert_passages_embeddings(self, collection:Collection, passages: List[Passage]):
        pass

    @abstractmethod
    def retrieve_vectors_if_exist(self, collection: Collection, passages: List[Passage]):
        pass


class ChromaDatabaseManager(VectorDatabaseManagerWrapper):
    """A Chroma Database can create N collections and will always use THE SAME embedder function and tokenizer for all collections.
        To create collections with different embedders one needs to create separate databases
    """
    def __init__(self, 
                 db_path: str = "default_db",
                 embedder_name: str = "sentence-transformers/all-MiniLM-L6-v2",
                 embedder_config: Dict[str,Any] = None,
                 batch_size: int = 8):
        super().__init__(batch_size)
        self.db_path = db_path
        self.embedder_name = embedder_name
        self.embedder_config = embedder_config
        self.tokenizer = AutoTokenizer.from_pretrained(embedder_name)
        self.client =  None
        self.requires_explicit_embeddings = False
        if self.embedder_config and self.embedder_config.get("type") == "hf":
            self.embedding_function = embedding_functions.HuggingFaceEmbeddingFunction(
                api_key=self.embedder_config.get("api_key"),
                model_name=embedder_name
            )
        elif self.embedder_config and self.embedder_config.get("type") == "custom":
            self.requires_explicit_embeddings = True
            self.embedding_function = None
        else:
            self.embedding_function = None

        self.config = {
            "embedder_name": self.embedder_name,
            "embedder_type": self.embedder_config["type"],
            "requires_explicit_embeddings": self.requires_explicit_embeddings,
            "existing_collections": []
        }
    
    def _save_config(self):
        if self.config:
            json.dump(self.config, open(f"{self.db_path}/config.json", "w"), indent=2)

    def connect(self):
        if not self.client:
            self.client = chromadb.PersistentClient(path=self.db_path)
            # If the given path existed then we will load the records AND override the parameters with the config
            config_path = f"{self.db_path}/config.json"
            if os.path.exists(config_path):
                self.config = json.load(open(config_path))
                print(f"Database Path '{self.db_path}' already exists. Loading current configuration\n{self.config}")
            else:
                print(f"Creating NEW Database in {self.db_path}...")
                self._save_config()
        else:
            print(f"A connection to the client already exsists in this session: {self.client}")

    def create_new_collection(self, name: str, passages: List[Passage] = None, embeddings: ArrayLike = None) -> Optional[Collection]:
        if not self.client:
            print("Please connect to a valid database first")
            return None
        # Create NEW collection and Embeds the given passages. Do nothing otherwise
        try:
            if self.embedding_function:
                collection = self.client.create_collection(name, 
                                                            embedding_function=self.embedding_function,
                                                            metadata={"hnsw:space": "cosine"})
            else:
                collection = self.client.create_collection(name, metadata={"hnsw:space": "cosine"})
            self.config["existing_collections"].append(name)
            self.active_collection_name = name
            self._save_config()
            # If the collection is new then insert the corresponding passages already
            if passages: self.insert_passages_embeddings(collection, passages, embeddings)

            return collection
        except UniqueConstraintError:
            print(f"Collection '{name}' already exists. Not doing anything. (You can delete it or use the 'get_existing_collection' method)")
            return None
        
    def get_existing_collection(self, name: str) -> Optional[Collection]:
        try:
            collection = self.client.get_collection(name, embedding_function=self.embedding_function)
            self.active_collection_name = name
            return collection
        except:
           return None


    def delete_collection(self, name):
        try:
            self.client.delete_collection(name)
            print(f"Succesfully deleted {name}")
        except:
            print(f"Collection '{name}' does not exist in this database.")


    def insert_passages_embeddings(self, collection: Collection, passages: List[Passage], embeddings: ArrayLike = None):
        if self.requires_explicit_embeddings and embeddings is None:
            raise ValueError("This Database requires embeddings to be explicitly pre-computed and fed into this function!")
        
        if isinstance(embeddings, list):
            embeddings_list = embeddings
        else:
            embeddings_list = embeddings.tolist()
        
        # TODO: When executed in separate context it actually inserts records again (so Hash is not actually unique). Find a better UNIQUE ID!
        num_records = collection.count()
        for i, batch in enumerate(self._batches(passages)):
            docs, metas, ids = [], [], []
            for p in batch:
                if embeddings is not None: p.tokenization = self._tokenize(p.text)
                docs.append(p.text)
                metas.append(p.metadata)
                ids.append(str(hash(p)))
            if embeddings is None:
                collection.add(
                    documents=docs,
                    metadatas=metas,
                    ids=ids
                )
            else:
                start_slice = i*self.batch_size
                embeds = embeddings_list[start_slice:start_slice+self.batch_size]
                collection.add(
                    documents=docs,
                    embeddings=embeds,
                    metadatas=metas,
                    ids=ids
                )
        
        new_count = collection.count()
        print(f"Added {new_count - num_records} new documents. Total = {new_count}")
    

    def compress_embeddings(
        self,
        collection: Collection = None,
        persist_in_db: bool = False,
        umap_verbose: bool = True,
        **umap_args) -> ArrayLike:
        
        records = self.get_records(collection)

        if len(records["ids"]) == 0: return None

        full_embeddings = records["embeddings"]
        database_ids = records["ids"]
        metadatas = records["metadatas"]

        umap = UMAP(verbose=umap_verbose, **umap_args)
        compressed = umap.fit_transform(full_embeddings)

        if persist_in_db:
            for i, m in enumerate(metadatas):
                datapoint = list(compressed[i])
                m["datapoint_x"] = float(datapoint[0])
                m["datapoint_y"] = float(datapoint[1])
            collection.update(database_ids, metadatas=metadatas)
            print("Corresponding Datapoints (datapoint_x, datapoint_y) were saved in the database")
        
        return compressed
    

    def get_records(self, collection:Collection, filter_words: List[str] = [], where_obj: Dict[str, Any] = None, include: List[str] = ["metadatas", "documents", "embeddings"]):
        # Result OBJ has these keys: dict_keys(['ids', 'embeddings', 'metadatas', 'documents', 'uris', 'data']) but by default only "metadatas" and "documents" are populated
        
        # Build the WHERE_DOCUMENT Object
        if len(filter_words) == 0:
            where_doc = None
        elif len(filter_words) == 1:
            where_doc={"$contains": filter_words[0]}
        else:
            where_doc = {'$and': [{"$contains": w} for w in filter_words]}

        result = collection.get(
                where=where_obj,
                where_document=where_doc,
                include=include
            )

        return {
            "ids": result.get("ids", []),
            "metadatas": result.get("metadatas", []),
            "documents": result.get("documents", []),
            "embeddings": [] if "embeddings" not in result else np.array(result["embeddings"], dtype=np.float32)
        }
    
    def query_vector_neighbors(self, collection: Collection, vector: List[float], k_neighbors = 10, include: List[str] = ["metadatas", "documents", "embeddings", "distances"]):
        result = collection.query(
            query_embeddings=[vector],
            n_results=k_neighbors,
            include=include
        )

        res_dict =  {
            "ids": result.get("ids", [[]])[0],
            "metadatas": result.get("metadatas", [[]])[0],
            "documents": result.get("documents", [[]])[0],
            "embeddings": [] if "embeddings" not in result else np.array(result["embeddings"][0], dtype=np.float32)
        }
        if "distances" in result:
            res_dict["distances"] = result["distances"][0]
        return res_dict
    
    def query_text_neighbors(self, collection: Collection, text: str, k_neighbors = 10, include: List[str] = ["metadatas", "documents", "embeddings", "distances"]):
        if not self.embedding_function:
            print("WARN: There is no embedding function defined in this database")
            return None
        result = collection.query(
            query_texts=[text],
            n_results=k_neighbors,
            include=include
        )

        res_dict =  {
            "ids": result.get("ids", [[]])[0],
            "metadatas": result.get("metadatas", [[]])[0],
            "documents": result.get("documents", [[]])[0],
            "embeddings": [] if "embeddings" not in result else np.array(result["embeddings"][0], dtype=np.float32)
        }
        if "distances" in result:
            res_dict["distances"] = result["distances"][0]
        return res_dict
          
    def _tokenize(self, sentence: str):
        encoded_input = self.tokenizer(sentence, return_tensors='pt')
        return encoded_input

    def embed_text_batch(self, text_batch: List[str]):
        if self.embedding_function:
            return self.embedding_function(text_batch)
        else:
            print("WARN: There is no valid embedding function in this database")
            return None
        
    def is_in_collection(self, collection:Collection, text: str):
        results = collection.get(where_document={"$contains": text}, include=["documents"])
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
        results = collection.get(where_document={"$contains": text}, include=["documents", "embeddings"])
        if len(results) == 0: return None
        for ix, ret_doc in enumerate(results["documents"]):
            if len(text) == len(ret_doc):
                return results["embeddings"][ix]
        return None
    
    def retrieve_vectors_if_exist(self, collection:Collection, passages: List[Passage]) -> List[float]:
        # TODO: Lookup by unique ID and retrieve the vectors if exist in the DB
        # Even if a single vector in the batch is missing then we will compute the full batch again (to avoid confusions)
        response = {"ids": []} # When we have uniqueID's we will query here...
        retrieved_embeddings = []
        if len(response["ids"]) == len(passages):
            retrieved_embeddings = response["embeddings"]
        return retrieved_embeddings
  