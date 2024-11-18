import datetime
from typing import Any, Iterable, Optional

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

    def _to_types(self, field_type: type) -> tuple[Any, Any]:
        """Convert the start and end years to the given field type.

        Args:
            field_type: type; this can be any Callable, including int, str, etc.
                if it is datetime.datetime, the start and end years are converted to datetime objects with UTC timezone.
        Returns:
            tuple[Any, Any]: The start and end years converted to the given field type.
        """
        if field_type == datetime.datetime:
            # Special case: datetime
            if self.start is None:
                start_date = None
            else:
                start_date = datetime.datetime(
                    year=self.start, month=1, day=1, hour=0, minute=0, second=0
                )
                if not start_date.tzinfo:
                    start_date = start_date.replace(tzinfo=datetime.timezone.utc)
            if self.end is None:
                end_date = None
            else:
                end_date = datetime.datetime(
                    year=self.end, month=12, day=31, hour=23, minute=59, second=59
                )
                if not end_date.tzinfo:
                    end_date = end_date.replace(tzinfo=datetime.timezone.utc)
        else:
            # default case:
            start_date = field_type(self.start) if self.start is not None else None
            end_date = field_type(self.end) if self.end is not None else None

        return start_date, end_date

    def to_weaviate_filter(
        self, *, field_name: str, field_type=datetime.datetime
    ) -> Iterable[Filter]:
        """Generate filters for querying Weaviate based on the year span.

        Yields one filter for each boundary of the year span which is not None. Can hence yield 0, 1, or 2 filters.

        Args:
            field_name: The name of the field to filter on, e.g. "year".
            field_type: the type of the field, e.g. int. Can be e.g. str and should match the type of the field as stored in the database.
        Yield:
            Filters for querying Weaviate.
        """

        start, end = self._to_types(field_type)
        if self.start is not None:
            yield Filter.by_property(field_name).greater_or_equal(start)
        if self.end is not None:
            # TODO: make end of range exclusive (less_than(end))
            yield Filter.by_property(field_name).less_or_equal(end)
