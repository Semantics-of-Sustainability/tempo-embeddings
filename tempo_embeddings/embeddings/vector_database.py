from abc import ABC, abstractmethod
from typing import Any, TypeVar, Union

from ..text.corpus import Corpus

Collection = TypeVar("Collection")


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

    @abstractmethod
    def ingest(self, collection: Union[Collection, str], corpus: Corpus):
        return NotImplemented

    @abstractmethod
    def get_corpus(
        self,
        collection: Union[Collection, str],
        filter_words: list[str],
        where_obj: dict[str, Any],
    ):
        return NotImplemented
