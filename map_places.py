import csv
import logging
import re
import time
from functools import lru_cache

import folium
from folium.plugins import HeatMapWithTime
from geopy.geocoders import Nominatim
from tqdm import tqdm

last_request_time = 0  # Global variable to store the time of the last request


@lru_cache(maxsize=1000000)
def geocode_place(geolocator, place_name):
    global last_request_time
    current_time = time.time()
    elapsed_time = current_time - last_request_time
    if elapsed_time < 1:
        time.sleep(1)  # Respect the rate limit of 1 request per second
    last_request_time = time.time()
    return geolocator.geocode(place_name)


def is_valid_place_name(place_name: str) -> bool:
    """Check if the place name is valid (not too short and not mostly special characters)."""
    if len(place_name) < 3:
        return False
    if re.fullmatch(r"\W+", place_name):
        return False
    return True


def create_map(input_csv, output, limit=1000):
    geolocator = Nominatim(timeout=10, user_agent="place_mapper")
    map_ = folium.Map(location=[52.3676, 4.9041], zoom_start=6)  # Centered on Amsterdam
    heat_data = defaultdict(list)

    with open(input_csv, mode="r", encoding="utf-8") as csvfile:
        reader = csv.DictReader(csvfile)
        total_lines = min(
            sum(1 for _ in csvfile) - 1, limit
        )  # Count lines excluding header and adapt to limit
        csvfile.seek(0)  # Reset file pointer to the beginning
        next(reader)  # Skip header

        for i, row in enumerate(
            tqdm(reader, desc="Processing places", total=total_lines)
        ):
            if i >= limit:
                break
            place_name = row["place_name"]
            if not is_valid_place_name(place_name):
                continue
            location = geocode_place(geolocator, place_name)
            if location:
                # Add pin for each location:
                # folium.Marker(
                #     [location.latitude, location.longitude],
                #     popup=f"{place_name} ({row['date']})",
                # ).add_to(map_)
                date = row["date"][:10]  # Extract the date part (YYYY-MM-DD)
                heat_data[date].append([location.latitude, location.longitude])

            if (i + 1) % 10 == 0:
                logging.debug(
                    f"Cache info after {i + 1} iterations: {geocode_place.cache_info()}"
                )

    heat_data_sorted = [heat_data[date] for date in sorted(heat_data)]
    HeatMapWithTime(heat_data_sorted).add_to(map_)
    map_.save(output)  # Save the map to the file


if __name__ == "__main__":
    import argparse
    from collections import defaultdict

    parser = argparse.ArgumentParser(
        description="Create a map of places from a CSV file."
    )
    parser.add_argument("--input-csv", help="Input CSV file with place names")
    parser.add_argument(
        "--output",
        "-o",
        type=argparse.FileType("x"),
        required=True,
        help="Output HTML file for the map",
    )
    parser.add_argument(
        "--limit", type=int, default=1000, help="Limit the number of places to process"
    )
    args = parser.parse_args()

    create_map(args.input_csv, args.output, args.limit)
