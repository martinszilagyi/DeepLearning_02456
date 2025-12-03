import duckdb
import pandas as pd
import folium
import os

folder_path = os.path.join(os.path.dirname(__file__), '../data/parquets/')
filename = 'MMSI=205243000/Segment=0/d7660b18593746e1b773307daba2758b-0.parquet'
file_path = '/zhome/fe/e/213609/DeepLearning/data/csvs/parquets/MMSI=210231000/Segment=3/b5d3cf8375d54e3fbef6e102d5c6213a-0.parquet'

# Load parquet robustly using DuckDB
df = duckdb.query(f"SELECT * FROM read_parquet('{file_path}')").to_df()

print("Columns:", df.columns)
print(df.head())

# Ensure latitude and longitude columns exist and are numeric
df['Latitude'] = pd.to_numeric(df['Latitude'], errors='coerce')
df['Longitude'] = pd.to_numeric(df['Longitude'], errors='coerce')
df = df.dropna(subset=['Latitude', 'Longitude'])

# Denormalization constants
lat_min, lat_max = 51.271528, 59.307808
lon_min, lon_max = 0.0002, 18.4068

# Denormalize Latitude and Longitude
df['Latitude_original'] = df['Latitude'] * (lat_max - lat_min) + lat_min
df['Longitude_original'] = df['Longitude'] * (lon_max - lon_min) + lon_min

# Create map centered on mean of denormalized lat/lon
center = [df['Latitude_original'].mean(), df['Longitude_original'].mean()]
m = folium.Map(location=center, zoom_start=6)

# Plot all points with original lat/lon
for _, row in df.iterrows():
    folium.CircleMarker(
        location=[row['Latitude_original'], row['Longitude_original']],
        radius=2,
        color='blue',
        fill=True,
        fill_opacity=0.6
    ).add_to(m)

# Save map
m.save("map.html")