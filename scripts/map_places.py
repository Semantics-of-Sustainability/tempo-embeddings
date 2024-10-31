import csv
import re

import folium
from folium.plugins import HeatMapWithTime
from tqdm import tqdm

from tempo_embeddings.io.geocoder import Geocoder


def is_valid_place_name(place_name: str) -> bool:
    """Check if the place name is valid (not too short and not mostly special characters)."""
    if len(place_name) < 3:
        return False
    if re.fullmatch(r"\W+", place_name):
        return False
    return True


def create_map(input_csv, output, limit=1000):
    geocoder = Geocoder()  # Initialize the Geocoder
    map_ = folium.Map(location=[52.3676, 4.9041], zoom_start=6)  # Centered on Amsterdam
    heat_data = defaultdict(list)

    # Create a feature group for the location pins
    pins_group = folium.FeatureGroup(name="Location Pins")

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
            latitude, longitude = geocoder.geocode_place(place_name)
            if latitude and longitude:
                # Add pin for each location:
                folium.Marker(
                    [latitude, longitude],
                    popup=f"{place_name} ({row['date']})",
                ).add_to(pins_group)
                date = row["date"][:10]  # Extract the date part (YYYY-MM-DD)
                heat_data[date].append([latitude, longitude])

    heat_data_sorted = [heat_data[date] for date in sorted(heat_data)]
    HeatMapWithTime(heat_data_sorted).add_to(map_)
    pins_group.add_to(map_)
    folium.LayerControl().add_to(map_)  # Add layer control to toggle pins
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

    create_map(args.input_csv, args.output.name, args.limit)
