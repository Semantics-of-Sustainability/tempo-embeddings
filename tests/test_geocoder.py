import os
import time
from unittest.mock import MagicMock

import pytest

from tempo_embeddings.io.geocoder import Geocoder


@pytest.fixture
def location():
    return 12.34, 56.78


@pytest.fixture
def geocoder(tmp_path):
    test_db_path = tmp_path / "test_geocode_cache.db"
    yield Geocoder(db_path=str(test_db_path))


@pytest.fixture
def mock_geocoder(mocker, location):
    mock_geocoder = mocker.patch("tempo_embeddings.io.geocoder.Nominatim.geocode")
    mock_geocoder.return_value = MagicMock(latitude=location[0], longitude=location[1])
    return mock_geocoder


class TestGeocoder:
    def test_init(self, geocoder):
        assert os.path.exists(geocoder.db_path)

    def test_cache_location_and_get_cached_location(self, geocoder, location):
        place_name = "Test Place"

        geocoder._cache_location(place_name, location[0], location[1])
        cached_location = geocoder._get_cached_location(place_name)
        assert cached_location == location

    def test_geocode_place(self, mock_geocoder, geocoder, location):
        place_name = "Test Place"

        assert geocoder.geocode_place(place_name) == location

        # test caching:
        geocoder.geocode_place(place_name)
        (
            mock_geocoder.assert_called_once_with(place_name),
            "Should have called the remote service exactly once.",
        )

    def test_geocode_place_with_rate_limit(self, mock_geocoder, geocoder, location):
        place_name = "Test Place"

        geocoder.last_request_time = time.time()
        start_time = time.time()
        assert geocoder.geocode_place(place_name) == location

        assert time.time() - start_time >= 1, "Should have delayed the remote call"
        assert (
            geocoder.last_request_time >= start_time + 1
        ), "Should have updated last_request_time"

        (
            mock_geocoder.assert_called_once_with(place_name),
            "Should have called the remote service exactly once.",
        )
