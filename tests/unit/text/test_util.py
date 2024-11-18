import datetime
from contextlib import nullcontext as does_not_raise

import pytest

from tempo_embeddings.text.util import any_to_int


@pytest.mark.parametrize(
    "input_value,expected,expected_exception",
    [
        (5, 5, None),
        ("10", 10, None),
        (datetime.datetime(2020, 1, 1), 2020, None),
        (datetime.date(2024, 11, 18), 2024, None),
        ("invalid", None, pytest.raises(ValueError)),
        (None, None, pytest.raises(ValueError)),
    ],
)
def test_any_to_int(input_value, expected, expected_exception):
    with expected_exception or does_not_raise():
        assert any_to_int(input_value) == expected
