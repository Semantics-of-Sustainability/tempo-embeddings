import json
import logging
import os
import platform
from typing import Any
from typing import Iterable
from typing import TypeVar
import weaviate
import weaviate.classes as wvc
from transformers import AutoTokenizer
from weaviate.classes.query import Filter
from weaviate.classes.query import MetadataQuery
from weaviate.util import generate_uuid5
from ..text.corpus import Corpus
from ..text.highlighting import Highlighting
from ..text.passage import Passage
from .vector_database import VectorDatabaseManagerWrapper


Collection = TypeVar("Collection")
logger = logging.getLogger(__name__)

class WeaviateDatabaseManager(VectorDatabaseManagerWrapper):
    """A Weaviate Database can create N collections and will always use THE SAME embedder function and tokenizer for all collections.
    To create collections with different embedders one needs to create separate databases
    """

    def __init__(
        self,
        db_path: str = "weaviate_default_db",
        embedder_name: str = None,
        embedder_config: dict[str, Any] = None,
        batch_size: int = 8,
    ):
        super().__init__(batch_size)
        self.db_path = db_path
        self.embedder_name = embedder_name
        self.embedder_config = embedder_config or {"type": "default"}
        self.tokenizer = AutoTokenizer.from_pretrained(embedder_name) if embedder_name else None
        self.model = None
        self.client = None
        self.weaviate_headers = {}
        
        if self.embedder_config.get("type") == "hf":
            self.embedding_function = wvc.config.Configure.Vectorizer.text2vec_huggingface(model=self.embedder_name)
            self.weaviate_headers = {"X-HuggingFace-Api-Key": self.embedder_config["api_key"]}
        elif self.embedder_config.get("type") == "custom_model" or self.embedder_config.get("type") == "default":
            try:
                self.model = self.embedder_config["model"]
                self.model.batch_size = self.batch_size
                self.embedding_function = wvc.config.Configure.Vectorizer.none()
            except KeyError as e:
                logger.error("If the type is 'custom_model' or 'default' you should pass the model object under Key 'model': %s", str(e))
        else:
            raise ValueError(f"Malformed embedder_config {self.embedder_config}. Check that 'type', 'api_key' and 'model' keys are properly populated.")

        if os.path.exists(self.db_path):
            self._load_config()
        else:
            os.makedirs(self.db_path)
            self.config = {
                "embedder_name": self.embedder_name,
                "embedder_type": self.embedder_config["type"],
                "model": self.model,
                "existing_collections": [],
            }
            self._save_config()
        
        
    def connect(self):
        with weaviate.connect_to_local() as client:
            # TODO: add functionality for connecting to remote database
            logger.info(type(client))
            logger.info("Weaviate Server Is Up: %s", client.is_ready())
            return client.is_ready()


    def _save_config(self):
        if self.config:
            with open(f"{self.db_path}/config.json", "w", encoding="utf-8") as f:
                json.dump(self.config, f, indent=2)
    

    def _load_config(self):
        with open(f"{self.db_path}/config.json", encoding="utf-8") as f:
            self.config = json.load(f)


    def get_available_collections(self):
        if self.config:
            return self.config.get("existing_collections", [])
        return []


    def create_new_collection(
        self,
        name: str,
        corpus: Corpus = None,
        # collection_metadata = None, # This Dict can be used in the future to give a precise SCHEMA. NOT IMPLEMENTED YET!
    ) -> None:
        """
        Create NEW collection and Embeds the given passages. Do nothing otherwise
        Args:
            name (str): name of the collection to be created in Weaviate
            corpus (Corpus, optional): Insert the provided corpus after creating the collection. Defaults to None.

        Raises:
            ValueError: If the given collection name already exists
        """
        # 
        if name in self.config["existing_collections"]:
            raise ValueError(f"Collection '{name}' has been created already! Try 'insert_corpus()' if you want to add more items to the collection")
        
        # If the collection is new then insert the corresponding passages already
        if corpus:
            self.insert_corpus(name, corpus)
            self.config["existing_collections"].append(name)
            self._save_config()
            logger.info("Created NEW collection '%s'", name)
        

    def delete_collection(self, name):
        with weaviate.connect_to_local() as client:
            try:
                client.collections.delete(name)
                self.config["existing_collections"] = [c for c in self.config["existing_collections"] if c != name]
                self._save_config()
                logger.info("Succesfully deleted %s", name)
            except Exception as e:
                logger.error("delete_collection() with name '%s' caused exception %s", name, e)


    def get_collection_count(self, name):
        with weaviate.connect_to_local() as client:
            collection = client.collections.get(name)
            response = collection.aggregate.over_all(total_count=True)
            return response.total_count

    def _insert_using_custom_model(self, corpus, collection):
        num_records = 0
        embeddings = self.model.embed_corpus(corpus, store_tokenizations=True, batch_size=self.batch_size)
        with weaviate.connect_to_local(headers=self.weaviate_headers) as client:
            collection_obj = client.collections.get(collection)
            for batch_pass, batch_embeds in zip(corpus.batches(self.batch_size), embeddings):
                logger.debug("Batch pass... %s | %s", type(batch_pass), type(batch_embeds))
                logger.debug("NODE: %s", platform.node())
                data_objects = []
                # Prepare Insertion Batch...
                for p, emb in zip(batch_pass, [tensor.tolist() for tensor in batch_embeds]):
                    props = p.metadata
                    props['passage'] = p.text
                    props['highlighting'] = str(p.highlighting)
                    #properties = {"passage": p.text, "title": p.metadata['title'], "date": datetime.strptime(p.metadata['date'],'%Y-%m-%d'), 
                    #               "issuenumber": int(p.metadata['issuenumber'])}
                    data_object = wvc.data.DataObject(
                        properties=props,
                        uuid=generate_uuid5(props),
                        vector=emb
                    )
                    data_objects.append(data_object)
                # Make the Insertion
                response = collection_obj.data.insert_many(data_objects)
                num_records += len(response.all_responses)
        return num_records

    def _insert_using_huggingface_api(self, corpus, collection):
        num_records = 0
        with weaviate.connect_to_local(headers=self.weaviate_headers) as client:
            collection_obj = client.collections.get(collection)
            with collection_obj.batch.dynamic() as batch:
                for p in corpus.passages:
                    props = p.metadata
                    props['passage'] = p.text
                    props['highlighting'] = str(p.highlighting)
                    batch.add_object(
                        properties=props,
                        uuid=generate_uuid5(props),
                        # vector=self.model.embed_passage(p)
                    )
                num_records += len(corpus.passages)
        return num_records

    def insert_corpus(self, collection: str, corpus: Corpus):

        if len(corpus.passages) == 0:
            raise ValueError("There should be at least one passage to insert.") 

        if self.model:
            num_records = self._insert_using_custom_model(corpus, collection)      
        elif self.embedder_config.get("type") == "hf":
            num_records = self._insert_using_huggingface_api(corpus, collection)
        else:
            raise ValueError("There is no valid way to vectorize the texts. Provide a custom model or a valid weaviate embedder")

        logger.info("Added %d new documents.", num_records)

    def _create_passage_from_record(self, rec_id, meta, vector):
        doc = meta.pop("passage")
        # Get Highlighting
        highlighting = None
        hl = meta["highlighting"]
        if "_" in hl:
            start, end = [int(x) for x in hl.split("_")]
            highlighting = Highlighting(start, end)
            # filter_terms.add(doc[start:end])
        # Create Passage
        p = Passage(doc, meta, highlighting, unique_id=rec_id, embedding=vector)
        p.tokenized_text = doc.split()
        return p

    # pylint: disable-next=too-many-arguments
    def get_corpus(self,
                   collection: str, 
                   filter_words: list[str] = None, 
                   where_obj: dict[str, Any] = None, 
                   include_embeddings: bool = False, 
                   limit: int = 10000):
        with weaviate.connect_to_local() as client:
            my_collection = client.collections.get(collection)

            db_filter_words = Filter.by_property("passage").contains_any(filter_words) if filter_words and len(filter_words) > 0 else None
            # TODO: How can we generalize the filtering?
            if where_obj and 'year_from' in where_obj and 'year_to' in where_obj:
                db_prop_filters = Filter.by_property("year").greater_or_equal(str(where_obj['year_from'])) & \
                                    Filter.by_property("year").less_or_equal(str(where_obj['year_to']))
                db_filters_all = db_filter_words & db_prop_filters
            else:
                db_filters_all = db_filter_words
            
            limit = limit if limit > 0 else None

            response = my_collection.query.fetch_objects(
                limit=limit,
                filters=db_filters_all, 
                include_vector=include_embeddings
            )

            if len(response.objects) > 0:
                passages = [self._create_passage_from_record(o.uuid, o.properties, o.vector['default'] 
                                                             if include_embeddings else None) 
                                                             for o in response.objects]
                return Corpus(passages, label="; ".join(filter_words) if filter_words else None) 
            
            return Corpus()
    
    
    def query_vector_neighbors(
        self,
        collection: Collection,
        vector: list[float],
        k_neighbors=10
    ) -> Iterable[tuple[Passage, float]]:
        
        with weaviate.connect_to_local(headers=self.weaviate_headers) as client:
            wv_collection = client.collections.get(collection)
            response = wv_collection.query.near_vector(
                near_vector=vector,
                limit=k_neighbors,
                include_vector=True,
                return_metadata=MetadataQuery(distance=True)
            )

            for o in response.objects:
                text = o.properties.pop("passage")
                yield (Passage(text, o.properties, embedding=o.vector['default'], unique_id=o.uuid), o.metadata.distance)
    

    def query_text_neighbors(
        self,
        collection: Collection,
        text: list[float],
        k_neighbors=10
    ) -> Iterable[tuple[Passage, float]]:
        
        with weaviate.connect_to_local(headers=self.weaviate_headers) as client:
            wv_collection = client.collections.get(collection)
            response = wv_collection.query.near_text(
                query=text,
                limit=k_neighbors,
                include_vector=True,
                return_metadata=MetadataQuery(distance=True)
            )

            for o in response.objects:
                text = o.properties.pop("passage")
                yield (Passage(text, o.properties, embedding=o.vector['default'], unique_id=o.uuid), o.metadata.distance)
        