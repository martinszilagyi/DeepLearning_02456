import json
import folium
import os

# Constants for scaling
LAT_MIN, LAT_MAX = 51.1395, 59.307808
LON_MIN, LON_MAX = 0.0002, 19.6405

def denormalize_lat(lat_norm):
    return LAT_MIN + lat_norm * (LAT_MAX - LAT_MIN)

def denormalize_lon(lon_norm):
    return LON_MIN + lon_norm * (LON_MAX - LON_MIN)

def denormalize_coords(coord_list):
    return [(denormalize_lat(p["lat"]), denormalize_lon(p["lon"])) for p in coord_list]

def visualize_sequences(json_filepath, output_html="sequences_map.html"):
    # Load JSON data
    with open(json_filepath, "r") as f:
        data = json.load(f)
    
    # Compute overall average location for initial map center
    all_coords = []
    for seq in data:
        all_coords.extend(denormalize_coords(seq["known_path"]))     # include known_path here for centering
        all_coords.extend(denormalize_coords(seq["ground_truth"]))
        all_coords.extend(denormalize_coords(seq["prediction"]))
    avg_lat = sum(lat for lat, lon in all_coords) / len(all_coords)
    avg_lon = sum(lon for lat, lon in all_coords) / len(all_coords)
    
    # Create folium map centered at average location
    m = folium.Map(location=[avg_lat, avg_lon], zoom_start=6)
    
    # Plot each sequence
    for seq in data:
        seq_id = seq.get("sequence_id", "N/A")
        
        # Denormalize points
        known_coords = denormalize_coords(seq["known_path"])
        gt_coords = denormalize_coords(seq["ground_truth"])
        pred_coords = denormalize_coords(seq["prediction"])
        
        # Add known path polyline (blue)
        folium.PolyLine(
            known_coords,
            color="blue",
            weight=3,
            opacity=0.7,
            tooltip=f"Sequence {seq_id} Known Path"
        ).add_to(m)
        
        # Add ground truth polyline (green)
        folium.PolyLine(
            gt_coords,
            color="green",
            weight=3,
            opacity=0.7,
            tooltip=f"Sequence {seq_id} Ground Truth"
        ).add_to(m)
        
        # Add prediction polyline (red)
        folium.PolyLine(
            pred_coords,
            color="red",
            weight=3,
            opacity=0.7,
            tooltip=f"Sequence {seq_id} Prediction"
        ).add_to(m)
    
    # Save the map
    m.save(output_html)
    print(f"Map saved to {output_html}")

# Example usage:
visualize_sequences(os.path.join(os.getcwd(), "test_predictions.json"), "sequences_map.html")
