"""
This script visualizes random vessel paths from parquet files in a specified directory.
It uses Folium to create an interactive map with plotted paths, start and end markers,
and saves the output as an HTML file which is then opened in the default web browser.
"""

import os
import random
import argparse
import folium
import pyarrow.parquet as pq
import pandas as pd
import webbrowser
import numpy as np

def extract_mmsi_from_path(path):
    parts = path.replace("\\", "/").split("/")
    for p in parts:
        if p.startswith("MMSI="):
            return p.split("=", 1)[1]
    return "Unknown"

def plot_random_paths(parquet_dir, output_html, num_paths=50):
    parquet_files = []
    for root, _, files in os.walk(parquet_dir):
        for f in files:
            if f.lower().endswith('.parquet'):
                parquet_files.append(os.path.join(root, f))

    if not parquet_files:
        print(f"No parquet files found in {parquet_dir}")
        return

    selected_files = random.sample(parquet_files, min(num_paths, len(parquet_files)))

    # Read first file to determine map center
    table = pq.read_table(selected_files[0])
    df_first = table.to_pandas()

    # Convert Latitude/Longitude to float just in case
    df_first['Latitude'] = pd.to_numeric(df_first['Latitude'], errors='coerce')
    df_first['Longitude'] = pd.to_numeric(df_first['Longitude'], errors='coerce')

    df_first = df_first.dropna(subset=['Latitude', 'Longitude'])
    center_lat = df_first['Latitude'].mean()
    center_lon = df_first['Longitude'].mean()

    m = folium.Map(location=[center_lat, center_lon], zoom_start=10)

    colors = [
        'blue', 'red', 'green', 'purple', 'orange', 'darkred', 'lightred', 'beige',
        'darkblue', 'darkgreen', 'cadetblue', 'darkpurple', 'white', 'pink',
        'lightblue', 'lightgreen', 'gray', 'black', 'lightgray'
    ]

    for i, file in enumerate(selected_files):
        try:
            mmsi = extract_mmsi_from_path(file)

            table = pq.read_table(file)
            df = table.to_pandas()

            # Convert coords to float and drop invalid rows
            df['Latitude'] = pd.to_numeric(df['Latitude'], errors='coerce')
            df['Longitude'] = pd.to_numeric(df['Longitude'], errors='coerce')
            df = df.dropna(subset=['Latitude', 'Longitude'])

            # Sort by Timestamp if exists, otherwise skip
            if 'Timestamp' in df.columns:
                df['Timestamp'] = pd.to_datetime(df['Timestamp'], errors='coerce')
                df = df.dropna(subset=['Timestamp'])
                df = df.sort_values('Timestamp')

            coords = list(zip(df['Latitude'], df['Longitude']))

            # Skip if less than 2 unique points
            if len(coords) < 2:
                continue
            if len(set(coords)) < 2:
                continue

            color = colors[i % len(colors)]

            folium.PolyLine(
                coords,
                color=color,
                weight=3,
                opacity=0.8,
                tooltip=f"MMSI: {mmsi}"
            ).add_to(m)

            # Start marker
            folium.CircleMarker(
                coords[0],
                radius=5,
                color=color,
                fill=True,
                fill_opacity=0.9,
                tooltip=f"Start – MMSI {mmsi}"
            ).add_to(m)

            # End marker
            folium.CircleMarker(
                coords[-1],
                radius=5,
                color=color,
                fill=True,
                fill_opacity=0.9,
                tooltip=f"End – MMSI {mmsi}"
            ).add_to(m)

        except Exception as e:
            print(f"Skipping {file} due to error: {e}")

    m.save(output_html)
    webbrowser.open(f'file://{os.path.realpath(output_html)}')
    print(f"Map saved to {output_html}")

def main():
    parser = argparse.ArgumentParser(description="Visualize random vessel paths from parquet files.")
    parser.add_argument("parquet_dir", help="Directory containing parquet files")
    parser.add_argument("--output", default="random_paths_map.html", help="Output HTML file name")
    parser.add_argument("--num", type=int, default=50, help="Number of random paths to plot")
    args = parser.parse_args()

    if not os.path.isdir(args.parquet_dir):
        print(f"Directory not found: {args.parquet_dir}")
        return

    plot_random_paths(args.parquet_dir, args.output, args.num)

if __name__ == "__main__":
    main()
