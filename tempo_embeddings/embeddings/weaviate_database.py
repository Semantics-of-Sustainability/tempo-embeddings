import gzip
import json
import logging
import platform
import uuid
from typing import Any, Iterable, Optional, TypeVar

from tqdm import tqdm

import weaviate
import weaviate.classes as wvc
from weaviate.classes.query import Filter, MetadataQuery
from weaviate.exceptions import UnexpectedStatusCodeError, WeaviateQueryError
from weaviate.util import generate_uuid5

from ..settings import WEAVIATE_CONFIG_COLLECTION
from ..text.corpus import Corpus
from ..text.highlighting import Highlighting
from ..text.passage import Passage
from .model import TransformerModelWrapper
from .vector_database import VectorDatabaseManagerWrapper

Collection = TypeVar("Collection")
logger = logging.getLogger(__name__)


class WeaviateConfigDb:
    # TODO: When this is implemented for other database backends, move common parts to an abstract class

    _CORPUS_NAME_FIELD = "corpus"

    def __init__(
        self,
        client: weaviate.Client,
        *,
        collection_name: str = WEAVIATE_CONFIG_COLLECTION,
    ) -> None:
        self._client: weaviate.Client = client
        self._collection_name = collection_name

        self._collection = (
            self._client.collections.get(self._collection_name)
            if self._exists()
            else self._create()
        )

    def __contains__(self, corpus: str):
        return corpus in self.get_corpora()

    def __getitem__(self, corpus: str) -> dict[str, Any]:
        uuid = generate_uuid5(corpus)
        if config := self._collection.query.fetch_object_by_id(uuid):
            return {"uuid": str(uuid)} | config.properties
        else:
            raise KeyError(f"Corpus '{corpus}' not found.")

    def __setitem__(self, corpus: str, properties: dict[str, Any]):
        _uuid = generate_uuid5(corpus)
        uuid = properties.pop("uuid", _uuid)

        if uuid != _uuid:
            raise ValueError(
                f"UUID '{uuid}' does not match expected UUID for corpus '{corpus}'"
            )

        try:
            return self._collection.data.insert(
                properties={WeaviateConfigDb._CORPUS_NAME_FIELD: corpus} | properties,
                uuid=uuid,
                vector={},
            )
        except UnexpectedStatusCodeError as e:
            raise ValueError(f"Error inserting corpus '{corpus}': {e}") from e

    def _exists(self):
        return self._client.collections.exists(self._collection_name)

    def _create(self) -> weaviate.collections.Collection:
        if self._exists():
            raise ValueError(f"Collection '{self._collection_name}' already exists.")
        return self._client.collections.create(
            self._collection_name,
            vectorizer_config=wvc.config.Configure.Vectorizer.none(),
        )

    def _delete(self) -> None:
        if self._exists():
            return self._client.collections.delete(self._collection_name)
        else:
            raise ValueError(f"Collection '{self._collection_name}' does not exist.")

    def add_corpus(
        self, corpus: str, embedder: str, properties: Optional[dict[str, Any]] = None
    ) -> uuid.UUID:
        properties = {
            WeaviateConfigDb._CORPUS_NAME_FIELD: corpus,
            "embedder": embedder,
        } | (properties or {})

        self[corpus] = properties

    def delete_corpus(self, corpus: str) -> bool:
        uuid = generate_uuid5(corpus)
        return self._collection.data.delete_by_id(uuid)

    def get_corpora(self) -> Iterable[str]:
        """Get all registered corpora.

        Returns:
            Iterable[str]: The names of all registered corpora
        """
        for group in self._collection.aggregate.over_all(
            group_by=wvc.aggregate.GroupByAggregate(
                prop=WeaviateConfigDb._CORPUS_NAME_FIELD
            )
        ).groups:
            yield group.grouped_by.value


class WeaviateDatabaseManager(VectorDatabaseManagerWrapper):
    """A Weaviate Database can create N collections and will always use THE SAME embedder function and tokenizer for all collections.
    To create collections with different embedders one needs to create separate databases

    A database is defined as a set of collections, comprising the metadata collection and one collection per corpus.
    A corpus is defined as a set of passages
    """

    def __init__(
        self,
        model: TransformerModelWrapper,
        *,
        # TODO hf_embedder=None ,hf_api_key:str,
        client: Optional[weaviate.Client] = None,
        config_collection_name: str = WEAVIATE_CONFIG_COLLECTION,
        batch_size: int = 8,
    ):
        super().__init__(batch_size)

        self.model = model
        # FIXME: allow for model=None, fix ingest()/_insert_using_custom_model()
        # TODO: add support for HF/Weaviate embedder
        weaviate_headers = {}

        self._client = client or weaviate.connect_to_local(headers=weaviate_headers)
        self._config = WeaviateConfigDb(
            self._client, collection_name=config_collection_name
        )

    def __del__(self):
        try:
            self._client.close()
        except:  # noqa: E722
            pass

    def __contains__(self, corpus: str):
        return corpus in self._config

    @property
    def client(self):
        return self._client

    def get_available_collections(self) -> Iterable[str]:
        return self._config.get_corpora()

    def provenances(
        self, collection: str, *, metadata_field: str = "provenance"
    ) -> Iterable[str]:
        """Return the filenames in the collection."""
        if collection in self._config:
            for group in (
                self._client.collections.get(collection)
                .aggregate.over_all(
                    group_by=wvc.aggregate.GroupByAggregate(prop=metadata_field)
                )
                .groups
            ):
                yield group.grouped_by.value
        else:
            logger.info(f"Collection '{collection}' not found, no files ingested.")

    def ingest(
        self,
        corpus: Corpus,
        name: Optional[str] = None,
        *,
        embedder: Optional[str] = None,
        properties: Optional[dict[str, Any]] = None,
    ):
        """Ingest a corpus into the database.

        Args:
            corpus (Corpus): The corpus to ingest
            name (str, optional): The name of the collection. Defaults to the corpus label.
            embedder (str, optional): The name of the embedder to use. Defaults to the model name.
            properties (dict[str, Any], optional): Additional properties to store in the database. Defaults to None.
        """
        if embedder is None and self.model is None:
            raise ValueError(
                "No embedder specified and no default model set. Either set a model or specify an embedder name."
            )

        name = name or corpus.label

        if len(corpus) > 0:
            if not self._client.collections.exists(name):
                self._config.add_corpus(
                    corpus=name,
                    embedder=embedder or self.model.name,
                    properties=properties or {},
                )
                # TODO: allow for embedded vectorizers
                self._client.collections.create(
                    name, vectorizer_config=wvc.config.Configure.Vectorizer.none()
                )
            collection = self._client.collections.get(name=name)
            # TODO: implement other model type
            try:
                self._insert_using_custom_model(corpus, collection)
            except WeaviateQueryError as e:
                logger.error("Error while ingesting corpus '%s': %s", name, e)
                if not self._client.collections.exists(name):
                    logger.info(f"Removing collection '{name}' to config database")
                    self._config.delete_corpus(name)

    def delete_collection(self, name):
        self._client.collections.delete(name)
        self._config.delete_corpus(name)

    def reset(self):
        """Delete all collections and reset the configuration database."""
        for collection in self._config.get_corpora():
            self.delete_collection(collection)

    def get_collection_count(self, name) -> int:
        """Returns the size of a given collection"""
        collection = self._client.collections.get(name)
        response = collection.aggregate.over_all(total_count=True)
        return response.total_count

    def _insert_using_custom_model(
        self, corpus, collection: weaviate.collections.Collection
    ) -> int:
        """Compute embeddings for all passages in the corpus and insert them into the collection.

        Args:
            corpus (Corpus): The corpus to insert
            collection (weaviate.collections.Collection): The collection to insert into
        Returns:
            int: The number of records inserted
        Raises:
            WeaviateQueryError: If a Weaviate error occurs during insertion
        """
        num_records = 0
        embeddings = self.model.embed_corpus(
            corpus, store_tokenizations=True, batch_size=self.batch_size
        )
        for batch_pass, batch_embeds in zip(
            corpus.batches(self.batch_size), embeddings
        ):
            logger.debug("Batch pass... %s | %s", type(batch_pass), type(batch_embeds))
            logger.debug("NODE: %s", platform.node())
            data_objects = []
            # Prepare Insertion Batch...
            for p, emb in zip(batch_pass, [tensor.tolist() for tensor in batch_embeds]):
                props = p.metadata
                props["passage"] = p.text
                props["highlighting"] = str(p.highlighting)
                data_object = wvc.data.DataObject(
                    properties=props, uuid=generate_uuid5(props), vector=emb
                )

                # TODO: allow for named vectors
                # see https://weaviate.io/developers/weaviate/manage-data/create#create-an-object-with-named-vectors
                data_objects.append(data_object)

            response = collection.data.insert_many(data_objects)
            num_records += len(response.all_responses)

        return num_records

    def _insert_using_huggingface_api(self, corpus, collection):
        num_records = 0
        collection_obj = self._client.collections.get(collection)
        with collection_obj.batch.dynamic() as batch:
            for p in corpus.passages:
                props = p.metadata
                props["passage"] = p.text
                props["highlighting"] = str(p.highlighting)
                batch.add_object(
                    properties=props,
                    uuid=generate_uuid5(props),
                    # vector=self.model.embed_passage(p)
                )
            num_records += len(corpus.passages)
        return num_records

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
    def get_corpus(
        self,
        collection: str,
        filter_words: list[str] = None,
        where_obj: dict[str, Any] = None,
        include_embeddings: bool = False,
        limit: int = 10000,
    ) -> Corpus:
        my_collection = self._client.collections.get(collection)

        db_filter_words = (
            Filter.by_property("passage").contains_any(filter_words)
            if filter_words and len(filter_words) > 0
            else None
        )
        # TODO: How can we generalize the filtering?
        if where_obj and "year_from" in where_obj and "year_to" in where_obj:
            db_prop_filters = Filter.by_property("year").greater_or_equal(
                str(where_obj["year_from"])
            ) & Filter.by_property("year").less_or_equal(str(where_obj["year_to"]))
            db_filters_all = db_filter_words & db_prop_filters
        else:
            db_filters_all = db_filter_words

        limit = limit if limit > 0 else None

        response = my_collection.query.fetch_objects(
            limit=limit, filters=db_filters_all, include_vector=include_embeddings
        )

        if len(response.objects) > 0:
            passages = [
                self._create_passage_from_record(
                    o.uuid,
                    o.properties,
                    o.vector["default"] if include_embeddings else None,
                )
                for o in response.objects
            ]
            return Corpus(
                passages, label="; ".join(filter_words) if filter_words else None
            )

        return Corpus()

    def query_vector_neighbors(
        self, collection: Collection, vector: list[float], k_neighbors=10
    ) -> Iterable[tuple[Passage, float]]:
        wv_collection = self._client.collections.get(collection)
        response = wv_collection.query.near_vector(
            near_vector=vector,
            limit=k_neighbors,
            include_vector=True,
            return_metadata=MetadataQuery(distance=True),
        )

        for o in response.objects:
            text = o.properties.pop("passage")
            yield (
                Passage(
                    text, o.properties, embedding=o.vector["default"], unique_id=o.uuid
                ),
                o.metadata.distance,
            )

    def query_text_neighbors(
        self, collection: Collection, text: list[float], k_neighbors=10
    ) -> Iterable[tuple[Passage, float]]:
        wv_collection = self._client.collections.get(collection)
        response = wv_collection.query.near_text(
            query=text,
            limit=k_neighbors,
            include_vector=True,
            return_metadata=MetadataQuery(distance=True),
        )

        for o in response.objects:
            text = o.properties.pop("passage")
            yield (
                Passage(
                    text, o.properties, embedding=o.vector["default"], unique_id=o.uuid
                ),
                o.metadata.distance,
            )

    def collection_config(self, collection_name: str) -> dict[str, Any]:
        """Get a dictionary with the configuration of a collection, including total_count."""
        # TODO: implement Pydantic config object; inherit to/from CorpusConfig class

        total_count = (
            self._client.collections.get(collection_name)
            .aggregate.over_all(total_count=True)
            .total_count
        )

        return self._config[collection_name] | {"total_count": total_count}

    def collection_objects(
        self, collection_name, *, include_vector: bool = True
    ) -> Iterable[dict[str, Any]]:
        """Get all objects from the collection serialized as dictionaries.

        - Returns all the properties of the object
        - Adds the "uuid" field as a string
        - Adds the "vector" field from the .vector["default"] object field

        Args:
            collection_name (str): The name of the collection
            include_vector (bool, optional): Whether to include the vector in the output. Defaults to True.
        Yields:
            Iterable[dict[str, Any]]: The objects in the collection serialized as dictionaries
        """

        for obj in self._client.collections.get(collection_name).iterator(
            include_vector=include_vector
        ):
            yield obj.properties | {
                "uuid": str(obj.uuid),
                "vector": obj.vector["default"] if include_vector else None,
            }

    def export_from_collection(
        self, collection_name: str, filepath: str = None
    ) -> None:
        """Export entire collection in jsonl format.

        The first line is the configuration, the rest are the records.

        Args:
            collection_name (str): The name of the collection to export
            filepath (str, optional): The file to write to. Defaults to None.
        """
        filename_tgt = filepath or f"{collection_name}.json.gz"

        with gzip.open(filename_tgt, "wt", encoding="utf-8") as fileout:
            logger.info("Writing into '%s' file...", filename_tgt)

            config = self.collection_config(collection_name)

            json.dump(config, fileout)
            fileout.write("\n")

            for _object in tqdm(
                self.collection_objects(collection_name),
                total=config["total_count"],
                unit="record",
                desc=f"Exporting '{collection_name}'",
            ):
                json.dump(_object, fileout)
                fileout.write("\n")

    def import_config(
        self, config: dict[str, Any], *, skip_keys={"corpus", "total_count"}
    ):
        """Import a collection configuration into the configuration database.

        Args:
            config (dict[str, Any]): The configuration dictionary
            skip_keys (set[str], optional): Keys to skip when importing. Defaults to {"corpus", "total_count"}.
        """
        self._config[config["corpus"]] = {
            key: config[key] for key in config if key not in skip_keys
        }

    def import_objects(
        self,
        objects: Iterable[dict[str, Any]],
        collection_name: str,
        *,
        total_count: int = None,
        batch_size: int = 100,
    ) -> int:
        """Import a collection of objects into the database.

        Modifies the objects in place by removing the "vector" and "uuid" keys.

        Args:
            objects (Iterable[dict[str, Any]]): The objects to import
            collection_name (str): The name of the collection
            total_count (int, optional): The total number of objects to import. Defaults to None.
            batch_size (int, optional): The batch size for importing. Defaults to 100.
        Returns:
            int: The number of records imported
        """
        count = 0

        with self._client.collections.get(collection_name).batch.fixed_size(
            batch_size=batch_size
        ) as batch:
            for _object in tqdm(
                objects,
                desc=f"Importing {collection_name}",
                unit="record",
                total=total_count,
            ):
                batch.add_object(
                    vector=_object.pop("vector"),
                    uuid=uuid.UUID(_object.pop("uuid")),
                    properties=_object,
                )

                count += 1

        if total_count is not None and count != total_count:
            logger.warning(
                "Total count mismatch: expected %d, but imported %d records",
                total_count,
                count,
            )

        return count

    def import_from_file(self, filename_src: str):
        """Import a collection from a jsonl file.

        The first line is expected to be the configuration, the rest are the records.

        Args:
            filename_src (str): The file to read from
        Raises:
            ValueError: If the corpus is already registered in the configuration

        """
        with gzip.open(filename_src, "rt", encoding="utf-8") as f:
            logger.info("Importing into Weaviate '%s' collection...", filename_src)

            config = json.loads(f.readline())
            collection_name = config["corpus"]
            total_count = config["total_count"]
            self.import_config(config)

            self.import_objects(
                (json.loads(line) for line in f),
                collection_name,
                total_count=total_count,
            )

    def validate_config(self) -> None:
        """Validate that the configuration database entries are present as database collections.

        Raises:
            ValueError: If a corpus is registered in the configuration database but the collection does not exist in the database
        """
        for corpus in self._config.get_corpora():
            if not self._client.collections.exists(corpus):
                raise ValueError(
                    "Corpus '%s' is registered in the configuration database but no collection of that name exists in the database.",
                    corpus,
                )
        for collection in self.client.collections.list_all():
            if (
                collection != self._config._collection_name
                and collection not in self._config
            ):
                logger.warning(
                    "Collection '%s' exists in the database but is not registered in the configuration database.",
                    collection,
                )
