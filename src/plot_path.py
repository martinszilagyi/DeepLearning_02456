import duckdb
import pandas as pd
import folium
import os

folder_path = os.path.join(os.path.dirname(__file__), '../data/parquets/')
filename = 'MMSI=205243000/Segment=0/d7660b18593746e1b773307daba2758b-0.parquet'
file_path = '/zhome/fe/e/213609/DeepLearning/data/parquets/MMSI=538009531/Segment=1/e361fb1ecba742c880f73f3868db7da2-0.parquet'

# Load parquet robustly using DuckDB
df = duckdb.query(f"SELECT * FROM read_parquet('{file_path}')").to_df()

print("Columns:", df.columns)
print(df.head())

# Ensure latitude and longitude columns exist and are numeric
df['Latitude'] = pd.to_numeric(df['Latitude'], errors='coerce')
df['Longitude'] = pd.to_numeric(df['Longitude'], errors='coerce')
df = df.dropna(subset=['Latitude', 'Longitude'])

# Create map centered on mean lat/lon
center = [df['Latitude'].mean(), df['Longitude'].mean()]
m = folium.Map(location=center, zoom_start=6)

# Plot all points
for _, row in df.iterrows():
    folium.CircleMarker(
        location=[row['Latitude'], row['Longitude']],
        radius=2,
        color='blue',
        fill=True,
        fill_opacity=0.6
    ).add_to(m)

# Save map
m.save("map.html")