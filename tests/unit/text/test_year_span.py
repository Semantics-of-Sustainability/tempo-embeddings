import datetime
from contextlib import nullcontext as does_not_raise

import pytest

from tempo_embeddings.settings import STRICT
from tempo_embeddings.text.year_span import YearSpan
from weaviate.classes.query import Filter

from ..embeddings.test_weaviate_database import TestQueryBuilder


class TestYearSpan:
    @pytest.mark.parametrize(
        "start, end, exception",
        [
            (None, None, None),
            (2000, None, None),
            (None, 2000, None),
            (2000, 2000, None),
            (2000, 1999, pytest.raises(ValueError)),
        ],
    )
    def test_validate_year_span(self, start, end, exception):
        with exception or does_not_raise():
            YearSpan(start=start, end=end).validate_year_span()

    @pytest.mark.parametrize(
        "start, end, field_name, expected",
        [
            (
                2000,
                None,
                "year",
                [
                    Filter.by_property("year").greater_or_equal(
                        datetime.datetime(2000, 1, 1, 0, 0, 0)
                    )
                ],
            ),
            (
                None,
                2000,
                "year",
                [
                    Filter.by_property("year").less_or_equal(
                        datetime.datetime(2000, 12, 31, 23, 59, 59)
                    )
                ],
            ),
            (
                2000,
                2020,
                "year",
                [
                    Filter.by_property("year").greater_or_equal(
                        datetime.datetime(2000, 1, 1, 0, 0, 0)
                    ),
                    Filter.by_property("year").less_or_equal(
                        datetime.datetime(2020, 12, 31, 23, 59, 59)
                    ),
                ],
            ),
            (
                2000,
                None,
                "field",
                [
                    Filter.by_property("field").greater_or_equal(
                        datetime.datetime(2000, 1, 1, 0, 0, 0)
                    )
                ],
            ),
        ],
    )
    def test_to_weaviate_filter(self, start, end, field_name, expected):
        year_span = YearSpan(start=start, end=end)

        for filter, _expected in zip(
            year_span.to_weaviate_filter(field_name=field_name), expected, **STRICT
        ):
            TestQueryBuilder.assert_filter_equals(filter, _expected)
