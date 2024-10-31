import sqlite3
import time

from geopy.geocoders import Nominatim


class Geocoder:
    def __init__(
        self, db_path="geocode_cache.db", user_agent="place_mapper", timeout=10
    ):
        self.db_path = db_path
        self.geolocator = Nominatim(user_agent=user_agent, timeout=timeout)
        self.init_db()
        self.last_request_time = (
            0  # Global variable to store the time of the last request
        )

    def init_db(self):
        """Initialize the SQLite database."""
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

    def get_cached_location(self, place_name):
        """Retrieve a cached location from the SQLite database."""
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        cursor.execute(
            "SELECT latitude, longitude FROM geocode_cache WHERE place_name = ?",
            (place_name,),
        )
        result = cursor.fetchone()
        conn.close()
        return result

    def cache_location(self, place_name, latitude, longitude):
        """Cache a location in the SQLite database."""
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

    def geocode_place(self, place_name):
        cached_location = self.get_cached_location(place_name)
        if cached_location:
            return cached_location

        current_time = time.time()
        elapsed_time = current_time - self.last_request_time
        if elapsed_time < 1:
            time.sleep(1)  # Respect the rate limit of 1 request per second
        self.last_request_time = time.time()

        location = self.geolocator.geocode(place_name)
        if location:
            self.cache_location(place_name, location.latitude, location.longitude)
        else:
            self.cache_location(place_name, None, None)  # Cache place with no location
        return location
