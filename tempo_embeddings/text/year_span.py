from typing import Iterable, Optional

from pydantic import model_validator
from pydantic.dataclasses import dataclass

from weaviate.classes.query import Filter


@dataclass
class YearSpan:
    """A span of years.

    Args:
        start: The start year of the span. If None, the span starts at the beginning of time.
        end: The end year of the span. If None, the span ends at the end of time.
    Raises:
        ValueError: if end is before start.
    """

    start: Optional[int] = None
    end: Optional[int] = None

    @model_validator(mode="after")
    def validate_year_span(self):
        if self.start is not None and self.end is not None and self.end < self.start:
            raise ValueError(
                f"End year {self.end} must be greater than or equal to start year {self.start}"
            )

        return self

    def to_weaviate_filter(
        self, *, field_name: str, field_type=int
    ) -> Iterable[Filter]:
        """Generate filters for querying Weaviate based on the year span.

        Yields one filter for each boundary of the year span which is not None. Can hence yield 0, 1, or 2 filters.

        Args:
            field_name: The name of the field to filter on, e.g. "year".
            field_type: the type of the field, e.g. int. Can be e.g. str and should match the type of the field as stored in the database.
        Yield:
            Filters for querying Weaviate.
        """
        if self.start is not None:
            yield Filter.by_property(field_name).greater_or_equal(
                field_type(self.start)
            )
        if self.end is not None:
            yield Filter.by_property(field_name).less_or_equal(field_type(self.end))
