import gzip
import json
import logging
import uuid
from functools import lru_cache
from typing import Any, Iterable, Optional, TypeVar

import cachetools
from cachetools.keys import hashkey
from shelved_cache import PersistentCache
from tqdm import tqdm

import weaviate
import weaviate.classes as wvc
from weaviate.auth import Auth
from weaviate.classes.config import DataType, Property
from weaviate.classes.query import Filter, MetadataQuery
from weaviate.exceptions import UnexpectedStatusCodeError, WeaviateQueryError
from weaviate.util import generate_uuid5

from ..settings import DOC_FREQUENCY_CACHE_FILE, STRICT, WEAVIATE_CONFIG_COLLECTION
from ..text.corpus import Corpus
from ..text.passage import Passage
from ..text.year_span import YearSpan
from .model import TransformerModelWrapper
from .vector_database import VectorDatabaseManagerWrapper

Collection = TypeVar("Collection")


class WeaviateConfigDb:
    # TODO: When this is implemented for other database backends, move common parts to an abstract class

    _CORPUS_NAME_FIELD = "corpus"

    def __init__(
        self,
        client: weaviate.Client,
        *,
        collection_name: str = WEAVIATE_CONFIG_COLLECTION,
        logger: Optional[logging.Logger] = None,
        doc_frequency_cache_file: str = DOC_FREQUENCY_CACHE_FILE,
    ) -> None:
        self._client: weaviate.Client = client
        self._collection_name = collection_name

        self._collection = (
            self._client.collections.get(self._collection_name)
            if self._exists()
            else self._create()
        )

        self._logger = logger or logging.getLogger(self.__class__.__name__)

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
        logger: Optional[logging.Logger] = None,
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
        self._logger = logger or logging.getLogger(self.__class__.__name__)

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

    @lru_cache
    def get_metadata_values(self, collection: str, field: str) -> list[str]:
        """Get the unique values for a metadata field in a collection.

        Args:
            collection (str): The collection name
            field (str): The metadata field name
        Returns:
            Iterable[str]: The unique values for the metadata field
        Raises:
            ValueError: If the collection does not exist
        """
        try:
            response = self._client.collections.get(collection).aggregate.over_all(
                group_by=wvc.aggregate.GroupByAggregate(prop=field)
            )
        except WeaviateQueryError as e:
            raise ValueError(
                f"Could not retrieve values for field '{field}' in collection '{collection}'."
            ) from e

        return [group.grouped_by.value for group in response.groups]

    def properties(self, collection: str) -> set[str]:
        """Get the properties of a collection.

        Args:
            collection (str): The collection name
        Returns:
            set[str]: The property names
        Raises:
            ValueError: If the collection does not exist
        """
        try:
            return {
                property.name
                for property in self._client.collections.get(collection)
                .config.get()
                .properties
            }
        except UnexpectedStatusCodeError as e:
            raise ValueError(
                f"Error retrieving properties for collection '{collection}': {e}"
            ) from e

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

        collection_name = name or corpus.label

        if self._client.collections.exists(collection_name):
            collection: Collection = self._client.collections.get(name=collection_name)
        else:
            self._config.add_corpus(
                corpus=collection_name,
                embedder=embedder or self.model.name,
                properties=properties or {},
            )
            collection = self._init_collection(collection_name, embedder, properties)

        # TODO: implement other model type
        try:
            return self._insert_using_custom_model(corpus, collection)
        except WeaviateQueryError as e:
            self._logger.error(
                "Error while ingesting corpus '%s': %s", collection_name, e
            )
            if not self._client.collections.exists(collection_name):
                self._logger.debug(
                    f"Removing collection '{collection_name}' from config database"
                )
                self._config.delete_corpus(collection_name)

    def _init_collection(
        self,
        collection_name: str,
        embedder: Optional[str] = None,
        properties: Optional[dict[str, Any]] = None,
    ) -> Collection:
        """Initialise a collection in the database.

        Note: this does not add it to the configuration database, so WeaviateConfigDB.add_corpus() should be called separately.

        Args:
            collection_name (str): The name of the collection. Defaults to the corpus label.
            embedder (str, optional): The name of the embedder to use. Defaults to the model name.
            properties (dict[str, Any], optional): Additional properties to store in the database. Defaults to None.

        Returns:
            Collection: the collection object

        """

        # TODO: allow for embedded vectorizers
        collection: Collection = self._client.collections.create(
            collection_name, vectorizer_config=wvc.config.Configure.Vectorizer.none()
        )
        for field, type_name in Passage.Metadata.model_field_names():
            try:
                collection.config.add_property(
                    Property(name=field, data_type=DataType(type_name))
                )
            except ValueError as e:
                self._logger.warning(
                    "Could not derive Weaviate data type for field '%s': %s", field, e
                )

        return collection

    def delete_collection(self, name):
        self._client.collections.delete(name)
        self._config.delete_corpus(name)

    def reset(self):
        """Delete all collections and reset the configuration database."""
        for collection in self._config.get_corpora():
            self.delete_collection(collection)

    def get_collection_count(
        self,
        name,
        metadata: Optional[dict[str, Any]] = None,
        metadata_not: Optional[dict[str, Any]] = None,
    ) -> int:
        """Returns the total size of a given collection, optionally with metadata filters applied.

        Args:
            name (str): The collection name
            metadata (dict[str, Any], optional): Additional metadata filters. Defaults to None.
            metadata_not (dict[str, Any], optional): Additional metadata filters to exclude. Defaults to None.
        """
        return self.doc_frequency(
            "", name, metadata=metadata, metadata_not=metadata_not, normalize=False
        )

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
        for passages_batch, embeddings_batch in zip(
            corpus.batches(self.batch_size),
            self.model.embed_corpus(
                corpus, store_tokenizations=True, batch_size=self.batch_size
            ),
            **STRICT,
        ):
            data_objects = []
            # Prepare Insertion Batch...
            for passage, embedding in zip(
                passages_batch,
                # split the tensor into a list of lists:
                [tensor.tolist() for tensor in embeddings_batch],
                **STRICT,
            ):
                props = passage.metadata | {
                    "passage": passage.text,
                    "highlighting": str(passage.highlighting),
                }
                data_object = wvc.data.DataObject(
                    properties=props, uuid=generate_uuid5(props), vector=embedding
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

    def get_corpus(
        self,
        collection: str,
        filter_words: list[str] = None,
        year_from: Optional[int] = None,
        year_to: Optional[int] = None,
        metadata_filters: Optional[dict[str, Any]] = None,
        metadata_not: Optional[dict[str, Any]] = None,
        include_embeddings: bool = False,
        limit: int = 10000,
        filter_duplicates: bool = True,
    ) -> Corpus:
        # TODO: replace year_from, year_to with YearSpan object

        """Get a corpus from the database with optional filters.

        Args:
            collection (str): the collection name
            filter_words (list[str], optional): Only include passage that contain any of these words. Defaults to None.
            year_from (int, optional): Only include passages from this year or later. Defaults to None.
            year_to (int, optional): Only include passages up to this year. Defaults to None.
            metadata_filters (dict[str, Any], optional): Additional filter criteria for exact matching in the form <field, term>.
            metadata_not (dict[str, Any], optional): Additional filter criteria to _exclude_ exact matching in the form <field, term>.
            include_embeddings (bool, optional): If true, include the embeddings to the output. Defaults to False.
            limit (int, optional): The maximum number of passages in the initial corpus. Defaults to 10000.
            filter_duplicates (bool, optional): If True, filter out duplicate passages. Defaults to True.
        Returns:
            Corpus: A corpus object

        Raises:
            RuntimeError: If an error occurs during the query
        """

        try:
            response = self._client.collections.get(collection).query.fetch_objects(
                limit=limit or None,
                filters=QueryBuilder.build_filter(
                    filter_words,
                    YearSpan(year_from, year_to),
                    metadata_filters,
                    QueryBuilder.clean_metadata(
                        metadata_not, self.properties(collection)
                    ),
                ),
                include_vector=include_embeddings,
            )
        except WeaviateQueryError as e:
            raise RuntimeError(
                f"Error while fetching corpus '{collection}'. Try a lower limit (was: {limit})."
            ) from e
        else:
            passages: tuple[Passage] = tuple(
                [
                    Passage.from_weaviate_record(o, collection=collection)
                    for o in response.objects
                ]
            )
            label = collection
            if passages and filter_words:
                label += ": '" + "; ".join(filter_words) + "'"

            return self._response_to_corpus(
                response.objects, collection, label, filter_duplicates
            )

    @staticmethod
    def _response_to_corpus(
        objects, collection_name: str, label: str, filter_duplicates: bool
    ) -> Corpus:
        passages = []

        if filter_duplicates:
            seen_texts = set()

            for object in objects:
                text = object.properties["passage"]
                if text not in seen_texts:
                    passages.append(
                        Passage.from_weaviate_record(object, collection=collection_name)
                    )
                    seen_texts.add(text)

            logging.info(
                f"Found {len(passages)} unique passages in {len(objects)} objects for collection '{collection_name}'."
            )
        else:
            passages = [
                Passage.from_weaviate_record(o, collection=collection_name)
                for o in objects
            ]

        return Corpus(passages, label)

    def doc_frequencies_per_year(
        self,
        term: str,
        collection: str,
        start_year: int,
        end_year: int,
        metadata: Optional[dict[str, Any]] = None,
        metadata_not: Optional[dict[str, Any]] = None,
        normalize: bool = False,
    ) -> dict[int, float]:
        """Get the number of documents that contain a term in the collection per year.

        Args:
            term (str): The term to count
            collection (str): collection to query
            start_year (int): The start year
            end_year (int): The end year
            metadata (dict[str, Any]): Additional metadata filters
            metadata_not (dict[str, Any]): Additional metadata filters to exclude
            normalize: If True, normalize the number of matching documents
        Returns:
            dict[int, float]: The (relative) number of occurrences of the term per year
        """

        return {
            year: self.doc_frequency(
                term,
                collection,
                metadata=metadata,
                metadata_not=metadata_not,
                normalize=normalize,
                year_span=YearSpan(year, year),
            )
            for year in range(start_year, end_year)
        }

    @staticmethod
    def __doc_frequency_hashkey(term, collection, metadata, metadata_not, year_span):
        def dict_to_tuple(d):
            return tuple(sorted(d.items())) if d else None

        (start, end) = (year_span.start, year_span.end) if year_span else (None, None)

        return hashkey(
            tuple(term),
            collection.name,
            dict_to_tuple(metadata),
            dict_to_tuple(metadata_not),
            start,
            end,
        )

    @cachetools.cached(
        PersistentCache(
            cachetools.TTLCache,
            filename=DOC_FREQUENCY_CACHE_FILE,
            maxsize=1024,
            ttl=2628000,  # 1 month
        ),
        key=__doc_frequency_hashkey,
    )
    @staticmethod
    def _doc_frequency(
        terms: list[str],
        collection: Collection,
        metadata: Optional[dict[str, Any]] = None,
        metadata_not: Optional[dict[str, Any]] = None,
        year_span: Optional[YearSpan] = None,
    ):
        response = collection.aggregate.over_all(
            filters=QueryBuilder.build_filter(
                terms, year_span, metadata=metadata, metadata_not=metadata_not
            ),
            total_count=True,
        )
        return response.total_count

    def doc_frequency(
        self,
        term: str,
        collection: str,
        metadata: Optional[dict[str, Any]] = None,
        metadata_not: Optional[dict[str, Any]] = None,
        normalize: bool = False,
        year_span: Optional[YearSpan] = None,
    ) -> float:
        """Get the number of documents that contain a term in the collection.

        If 'term' is empty, return the total number of documents in the collection.

        If 'normalize' is True, normalize the number of documents that contain the term by the number of documents
            in the collection matching the filters, but regardless of the term.

        Args:
            term (str): The term to count
            collection (str): collection to query
            metadata (dict[str, Any]): Additional metadata filters
            metadata_not (dict[str, Any]): Additional metadata filters to exclude
            normalize: If True, normalize the number of matching documents
            year_span (Optional[YearSpan], optional): The year range to consider. Defaults to None.

        Returns:
            float: The (relative) number of occurrences of the term
        """
        search_terms: list[str] = [term] if term.strip() else []
        if normalize and not search_terms:
            self._logger.warning("Did not provide a term to normalize.")
            return 1.0

        _metadata_not = QueryBuilder.clean_metadata(
            metadata_not, self.properties(collection)
        )
        _collection = self._client.collections.get(collection)
        freq: int = WeaviateDatabaseManager._doc_frequency(
            search_terms, _collection, metadata, _metadata_not, year_span
        )

        if freq and normalize:
            total: int = self.doc_frequency(
                "",
                collection,
                metadata,
                metadata_not,
                normalize=False,
                year_span=year_span,
            )
            freq /= total

        return freq

    def neighbours(
        self,
        corpus: Corpus,
        k: int,
        *,
        distance: Optional[float] = None,
        collections: Optional[list[str]] = None,
        year_span: Optional[YearSpan] = None,
        metadata_not: Optional[dict[str, Any]] = None,
        exclude_passages: Optional[set[Passage]] = None,
    ) -> Corpus:
        """Find passages to expand a corpus with the k-nearest neighbors of the centroid of the corpus.

        Passages in the new corpus are sorted by distance to the centroid.

        Args:
            corpus (Corpus): The corpus to expand, edited in-place
            k (int): The maximum number of neighbors to add per collection
            distance (Optional[float], optional): The maximum distance to consider. Defaults to 0.2.
            collections (Optional[list[str]], optional): The collections to query. Defaults to all available collections.
            year_range (Optional[YearSpan], optional): The year range to consider. Defaults to None.
            metadata_not (Optional[dict[str, Any]], optional): Additional metadata filters to exclude. Defaults to None.
            exclude_passages (Optional[set[Passage]]): Passages to exclude from the search. If not specified, exclude passages from the original corpus.

        Returns:
            A new corpus with passages close to the input corpus
        """
        passages: dict[Passage, float] = dict()
        if len(corpus) == 0:
            self._logger.warning("Empty corpus, no neighbors to find.")
        else:
            exclude_passages = exclude_passages or set(corpus.passages)

            for collection in collections or self.get_available_collections():
                centroid = corpus.centroid(use_2d_embeddings=False).tolist()

                try:
                    for passage, distance in self._query_vector_neighbors(
                        collection,
                        centroid,
                        k + len(exclude_passages),  # account for excluded passages
                        max_distance=distance,
                        year_span=year_span,
                        metadata_not=metadata_not,
                    ):
                        if passage not in exclude_passages:
                            passages[passage] = min(
                                distance, passages.get(passage, float("inf"))
                            )
                except WeaviateQueryError as e:
                    self._logger.error(
                        "Error while querying collection '%s': %s", collection, e
                    )

        _sorted = sorted(passages.items(), key=lambda x: x[1])

        # FIXME: the cosine distances from Weaviate are not the same as the ones from Corpus.distances()
        passages = [passage for passage, _ in _sorted[:k]]

        label = f"{corpus.label} {k} neighbours"

        return Corpus(passages, label, umap_model=corpus.umap)

    def _query_vector_neighbors(
        self,
        collection: str,
        vector: list[float],
        max_neighbors=10,
        max_distance: Optional[float] = None,
        *,
        year_span: Optional[YearSpan] = None,
        metadata_not: Optional[dict[str, Any]] = None,
    ) -> Iterable[tuple[Passage, float]]:
        # TODO: use autocut: https://weaviate.io/developers/weaviate/api/graphql/additional-operators#autocut
        # TODO: add excluded passages parameter and account for them in max_neighbours

        if max_neighbors > 10000:
            self._logger.warning(
                "Limiting maximum number of neighbors to 10000 (was: %d) while querying '%s'.",
                max_neighbors,
                collection,
            )
            max_neighbors = 10000
        wv_collection = self._client.collections.get(collection)
        response = wv_collection.query.near_vector(
            near_vector=vector,
            distance=max_distance,
            limit=max_neighbors,
            include_vector=True,
            return_metadata=MetadataQuery(distance=True),
            filters=QueryBuilder.build_filter(
                year_span=year_span,
                metadata_not=QueryBuilder.clean_metadata(
                    metadata_not, self.properties(collection)
                ),
            ),
        )

        for o in response.objects:
            yield (
                Passage.from_weaviate_record(o, collection=collection),
                o.metadata.distance,
            )

    def query_text_neighbors(
        self, collection: Collection, text: list[float], k_neighbors=10
    ) -> Iterable[tuple[Passage, float]]:
        # TODO: add filters for metadata, year_span, metadata_not

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
            self._logger.info("Writing into '%s' file...", filename_tgt)

            config = self.collection_config(collection_name)

            json.dump(config, fileout)
            fileout.write("\n")

            for _object in tqdm(
                self.collection_objects(collection_name),
                total=config["total_count"],
                unit="record",
                desc=f"Exporting '{collection_name}'",
            ):
                json.dump(_object, fileout, default=str)
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
        total_count: Optional[int] = None,
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

        collection: Collection = (
            self._client.collections.get(name=collection_name)
            if self._client.collections.exists(collection_name)
            else self._init_collection(
                collection_name, self._config[collection_name]["embedder"]
            )
        )
        with collection.batch.fixed_size(batch_size=batch_size) as batch:
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

        if collection.batch.failed_objects:
            self._logger.error(collection.batch.failed_objects)

        if total_count is not None and count != total_count:
            self._logger.warning(
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
            self._logger.info(
                "Importing into Weaviate '%s' collection...", filename_src
            )

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
                self._logger.warning(
                    "Collection '%s' exists in the database but is not registered in the configuration database.",
                    collection,
                )

    @classmethod
    def from_args(
        cls,
        *,
        model_name: Optional[str],
        ### weaviate arguments:
        http_host: str,
        http_port: int = 8087,
        http_secure: bool = False,
        grpc_host: str = None,
        grpc_port: int = 50051,
        api_key: Optional[str] = None,
        ###
        batch_size: int = 8,
    ):
        """Initialize a WeaviateDatabaseManager by initializing a model wrapper and a Weaviate client.

        Note: the model_name is currently expected to be a SentenceTransformer model.

        Args:
            model_name (str): The name of the model to use for a SentenceTransformerModelWrapper
            http_host (str): The Weaviate server host
            http_port (int, optional): The Weaviate server port. Defaults to 8087.
            http_secure (bool, optional): Use SSL. Defaults to False.
            grpc_host (str, optional): The Weaviate gRPC host. Defaults to None.
            grpc_port (int, optional): The Weaviate gRPC port. Defaults to 50051.
            api_key (Optional[str], optional): The Weaviate API key. Defaults to None.
            batch_size (int, optional): The batch size. Defaults to 8.
        Returns:
            WeaviateDatabaseManager: The initialized database manager
        """
        weaviate_client = weaviate.connect_to_custom(
            http_host,
            http_port,
            http_secure,
            grpc_host=grpc_host or http_host,
            grpc_port=grpc_port,
            grpc_secure=http_secure,
            auth_credentials=Auth.api_key(api_key) if api_key else None,
        )
        model = (
            TransformerModelWrapper.from_model_name(model_name) if model_name else None
        )
        return cls(model, client=weaviate_client, batch_size=batch_size)


class QueryBuilder:
    @staticmethod
    def build_filter(
        filter_words: list[str] = None,
        year_span: Optional[YearSpan] = None,
        metadata: Optional[dict[str, Any]] = None,
        metadata_not: Optional[dict[str, Any]] = None,
        *,
        text_field: str = "passage",
        year_field: str = "date",
    ) -> Optional[Filter]:
        """Generic method to build a Weaviate Filter.

        All filters are combined with AND.

        Args:
            filter_words (list[str], optional): Only include passage that contain any of these words. Defaults to None.
            year_span (YearSpan, optional): Only include passages from this year span. Defaults to None.
            metadata (dict[str, Any], optional): Additional filter criteria for exact matches in the form <field, term>.
            metadata_not(dict[str, Any], optional): Additional filter criteria to _exclude_ exact matches in the form <field, term> or <field, [terms]>.
            text_field: The field name for the text. Defaults to "passage".
            year_field (str, optional): The field name for the year. Defaults to "year".
        Returns:
            Optional[Filter]: The filter object or None if none of the input arguments are set.

        """
        # TODO: contains_any should be a list of tuples to allow for multiple values per field

        filters: list[Filter] = []

        if filter_words:
            filters.append(Filter.by_property(text_field).contains_any(filter_words))

        if year_span is not None:
            filters.extend(year_span.to_weaviate_filter(field_name=year_field))

        if metadata:
            filters.extend(
                [
                    Filter.by_property(field).equal(value)
                    for field, value in metadata.items()
                ]
            )
        if metadata_not:
            for field, value in metadata_not.items():
                if isinstance(value, list):
                    filters.extend(
                        [Filter.by_property(field).not_equal(v) for v in value]
                    )
                else:
                    filters.append(Filter.by_property(field).not_equal(value))

        return Filter.all_of(filters) if filters else None

    @staticmethod
    def clean_metadata(
        metadata: Optional[dict[str, Any]], collection_properties: set[str]
    ) -> dict[str, Any]:
        """Remove metadata fields that are not in the collection properties.

        Args:
            metadata (dict[str, Any]): A metadata dictionary containing field-value pairs
            collection_properties (set[str]): The collection properties
        Returns:
            dict[str, Any]: A new dictionary containing only fields that are in the collection properties
        """
        if metadata:
            return {
                key: value
                for key, value in metadata.items()
                if key in collection_properties
            }
        else:
            return {}
