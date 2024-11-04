import csv
import re
from collections import defaultdict, deque

import folium
import pandas as pd
from folium.plugins import HeatMapWithTime
from tqdm import tqdm

from tempo_embeddings.io.geocoder import Geocoder


def read_data_list(input_csv, limit, geocoder):
    data = []
    heat_data = defaultdict(list)
    with open(input_csv, mode="r", encoding="utf-8") as csvfile:
        reader = csv.DictReader(csvfile)
        total_lines = min(sum(1 for _ in csvfile) - 1, limit)
        csvfile.seek(0)  # Reset file pointer to the beginning
        next(reader)  # Skip header

        for i, row in enumerate(
            tqdm(reader, unit="row", desc="Processing places", total=total_lines)
        ):
            if i >= limit:
                break
            place_name = row["place_name"]
            if len(re.findall(r"[a-zA-Z]", place_name)) >= 3:
                # skip invalid place names
                latitude, longitude = geocoder.geocode_place(place_name)
                if latitude and longitude:
                    date = row["date"][:10]  # Extract the date part (YYYY-MM-DD)
                    data.append([place_name, latitude, longitude, date])
                    heat_data[date].append([latitude, longitude])
    return data, heat_data


def add_markers(data, pins_group):
    df = pd.DataFrame(data, columns=["place_name", "latitude", "longitude", "date"])
    grouped = (
        df.groupby(["latitude", "longitude"])
        .agg(
            {
                "place_name": lambda x: list(set(x)),
                "date": lambda x: list(sorted(set(x))),
            }
        )
        .reset_index()
    )

    for _, row in grouped.iterrows():
        table_html = """
        <div style="width: 300px;">
            <table style="width: 100%;">
                <tr><th>Place Name</th><th>Dates</th></tr>
        """
        for place_name in row["place_name"]:
            place_dates = df[
                (df["latitude"] == row["latitude"])
                & (df["longitude"] == row["longitude"])
                & (df["place_name"] == place_name)
            ]["date"].tolist()
            table_html += f"<tr><td>{place_name}</td><td>{', '.join(sorted(set(place_dates)))}</td></tr>"
        table_html += "</table></div>"
        folium.Marker([row["latitude"], row["longitude"]], popup=table_html).add_to(
            pins_group
        )


def create_smoothed_heat_data(heat_data, window_size):
    sorted_dates = sorted(heat_data)
    smoothed_heat_data = []
    window = deque(maxlen=window_size)

    for date in sorted_dates:
        window.append(heat_data[date])
        combined_data = [coord for day_data in window for coord in day_data]
        smoothed_heat_data.append(combined_data)

    return smoothed_heat_data, sorted_dates


def create_map(input_csv, output, title=None, limit=1000, window_size=7):
    geocoder = Geocoder()  # Initialize the Geocoder
    map_ = folium.Map(location=[52.3676, 4.9041], zoom_start=6)  # Centered on Amsterdam

    # Add a title to the map if provided
    if title:
        title_html = f"""
            <div style="position: fixed;
                        top: 10px; left: 50%; transform: translateX(-50%); width: auto; height: 50px;
                        background-color: white; z-index: 9999; font-size: 24px;">
                <center>{title}</center>
            </div>
        """
        map_.get_root().html.add_child(folium.Element(title_html))

    # Create a feature group for the location pins
    pins_group = folium.FeatureGroup(name="Location Pins", show=False)

    data, heat_data = read_data_list(input_csv, limit, geocoder)

    add_markers(data, pins_group)

    smoothed_heat_data, sorted_dates = create_smoothed_heat_data(heat_data, window_size)

    HeatMapWithTime(
        smoothed_heat_data, index=sorted_dates, name="Time-Space Heat Map"
    ).add_to(map_)
    pins_group.add_to(map_)
    folium.LayerControl().add_to(map_)  # Add layer control to toggle pins
    map_.save(output)  # Save the map to the file


if __name__ == "__main__":
    import argparse

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
        "--title", help="Title to be included in the map", required=False
    )
    parser.add_argument(
        "--limit", type=int, default=1000, help="Limit the number of places to process"
    )
    parser.add_argument(
        "--window-size",
        type=int,
        default=7,
        help="Window size for smoothing the heatmap",
    )
    args = parser.parse_args()

    create_map(
        args.input_csv, args.output.name, args.title, args.limit, args.window_size
    )
