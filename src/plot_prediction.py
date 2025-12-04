"""
This script visualizes ship trajectory predictions on a folium map. It reads
a JSON file containing known paths, ground truth, and predicted paths. For the plotting the data is denormalized using
provided Z normalization parameters. The resulting map is saved as an HTML file.
- Blue paths represent the known trajectories from which the predictions were made.
- Red paths represent the predicted trajectories.
- Green paths represent the ground truth trajectories for comparison.
"""

import json
import folium
import os
import argparse

# Argument parser for bsub flag (for jobs on cluster)
parser = argparse.ArgumentParser()
parser.add_argument("--isbsub", action="store_true")
args = parser.parse_args()

postifx = "_bsub" if args.isbsub else ""

# Z normalization parameters initialization
lat_mean = 0.0
lat_std = 0.0
lon_mean = 0.0
lon_std = 0.0

# Denormalize latitude
def denormalize_lat(lat_norm):
    return lat_norm * lat_std + lat_mean

# Denormalize longitude
def denormalize_lon(lon_norm):
    return lon_norm * lon_std + lon_mean

# Denormalize list of coordinates
def denormalize_coords(coord_list):
    return [(denormalize_lat(p["lat"]), denormalize_lon(p["lon"])) for p in coord_list]

# Visualize sequences on a folium map
def visualize_sequences(json_filepath, output_html="sequences_map.html"):
    global lat_mean, lat_std, lon_mean, lon_std

    # Open the JSON file and read data
    with open(json_filepath, "r") as f:
        data = json.load(f)
    
    # Get normalization parameters
    lat_mean = data["columns"]["Latitude"]["mean"]
    lat_std  = data["columns"]["Latitude"]["std"]
    lon_mean = data["columns"]["Longitude"]["mean"]
    lon_std  = data["columns"]["Longitude"]["std"]
    
    # Collect all coords for map centering
    all_coords = []
    for seq in data:
        all_coords.extend(denormalize_coords(seq["known_path"]))
        all_coords.extend(denormalize_coords(seq["ground_truth"]))
        all_coords.extend(denormalize_coords(seq["prediction"]))

    # Center map around average location
    avg_lat = sum(lat for lat, lon in all_coords) / len(all_coords)
    avg_lon = sum(lon for lat, lon in all_coords) / len(all_coords)

    # Create folium map
    m = folium.Map(location=[avg_lat, avg_lon], zoom_start=6)

    # Plot each sequence
    for seq in data:
        seq_id = seq.get("sequence_id", "N/A")
        
        known_coords = denormalize_coords(seq["known_path"])
        gt_coords = denormalize_coords(seq["ground_truth"])
        pred_coords = denormalize_coords(seq["prediction"])

        folium.PolyLine(
            known_coords, color="blue", weight=3, opacity=0.7,
            tooltip=f"Sequence {seq_id} Known Path"
        ).add_to(m)

        folium.PolyLine(
            gt_coords, color="green", weight=3, opacity=0.7,
            tooltip=f"Sequence {seq_id} Ground Truth"
        ).add_to(m)

        folium.PolyLine(
            pred_coords, color="red", weight=3, opacity=0.7,
            tooltip=f"Sequence {seq_id} Prediction"
        ).add_to(m)

    # Save map to HTML
    m.save(output_html)
    print(f"Map saved to {output_html}")

# Main execution
if __name__ == "__main__":
    # Call the visualization function
    visualize_sequences(
        os.path.join(os.getcwd(), "test_predictions" + postifx + ".json"),
        "sequences_map" + postifx + ".html"
    )
