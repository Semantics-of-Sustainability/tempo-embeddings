import bz2
import gzip
import lzma
from contextlib import nullcontext as does_not_raise

import pytest

from tempo_embeddings.io.util import get_open_func


@pytest.mark.parametrize(
    "compression, expected, expected_exception",
    [
        (None, open, does_not_raise()),
        ("gzip", gzip.open, does_not_raise()),
        ("bz2", bz2.open, does_not_raise()),
        ("lzma", lzma.open, does_not_raise()),
        ("unknown", None, pytest.raises(ValueError)),
    ],
)
def test_get_open_func(compression, expected, expected_exception):
    with expected_exception:
        assert get_open_func(compression) == expected
