import logging
import sqlite3
import time
from typing import Optional, Tuple

from geopy.exc import GeocoderTimedOut
from geopy.geocoders import Nominatim


class Geocoder:
    def __init__(
        self,
        db_path: str = "geocode_cache.db",
        user_agent: str = "place_mapper",
        timeout: int = 10,
    ) -> None:
        """
        Initializes the Geocoder class.

        Args:
            db_path (str): Path to the SQLite database file.
            user_agent (str): User agent for the Nominatim geolocator.
            timeout (int): Timeout for the geolocator requests.
        """
        self.db_path = db_path
        self.geolocator = Nominatim(user_agent=user_agent, timeout=timeout)
        self.init_db()
        self.last_request_time = 0

    def init_db(self) -> None:
        """
        Initializes the SQLite database.
        """
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        cursor.execute(
            """
            CREATE TABLE IF NOT EXISTS geocode_cache (
                place_name TEXT PRIMARY KEY,
                latitude REAL,
                longitude REAL
            )
            """
        )
        conn.commit()
        conn.close()

    def get_cached_location(self, place_name: str) -> Optional[Tuple[float, float]]:
        """
        Retrieves a cached location from the SQLite database.

        Args:
            place_name (str): Name of the place to retrieve the location for.

        Returns:
            Optional[Tuple[float, float]]: A tuple containing the latitude and longitude, or None if not found.
        """
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        cursor.execute(
            "SELECT latitude, longitude FROM geocode_cache WHERE place_name = ?",
            (place_name,),
        )
        result = cursor.fetchone()
        conn.close()
        return result

    def cache_location(
        self, place_name: str, latitude: Optional[float], longitude: Optional[float]
    ) -> None:
        """
        Caches a location in the SQLite database.

        Args:
            place_name (str): Name of the place to cache.
            latitude (float): Latitude of the place.
            longitude (float): Longitude of the place.
        """
        if latitude is None or longitude is None:
            latitude, longitude = float("nan"), float("nan")
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        cursor.execute(
            "INSERT OR REPLACE INTO geocode_cache (place_name, latitude, longitude) VALUES (?, ?, ?)",
            (place_name, latitude, longitude),
        )
        conn.commit()
        conn.close()

    def geocode_place(self, place_name: str) -> Optional[Tuple[float, float]]:
        """
        Geocodes a place name using the Nominatim geolocator.

        Args:
            place_name (str): Name of the place to geocode.

        Returns:
            Optional[Tuple[float, float]]: A tuple containing the latitude and longitude, or None if not found.
        """
        cached_location = self.get_cached_location(place_name)
        if cached_location:
            return cached_location

        current_time = time.time()
        elapsed_time = current_time - self.last_request_time
        if elapsed_time < 1:
            time.sleep(1)  # Respect the rate limit of 1 request per second
        self.last_request_time = time.time()

        try:
            location = self.geolocator.geocode(place_name)
        except GeocoderTimedOut as e:
            logging.error(f"Geocoding request for '{place_name}' timed out: {e}")
            lat, long = None, None
        else:
            if location:
                lat, long = location.latitude, location.longitude
            else:
                lat, long = None, None
            self.cache_location(place_name, lat, long)

        return lat, long