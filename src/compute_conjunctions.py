# src/compute_conjunctions.py
import pandas as pd
import numpy as np
from itertools import combinations
import os

os.makedirs("data", exist_ok=True)

df = pd.read_csv("data/positions.csv")

# Convert to ECEF coordinates
EARTH_R = 6371.0  # km
def geodetic_to_ecef(lat, lon, alt):
    lat = np.deg2rad(lat)
    lon = np.deg2rad(lon)
    r = EARTH_R + alt
    x = r * np.cos(lat) * np.cos(lon)
    y = r * np.cos(lat) * np.sin(lon)
    z = r * np.sin(lat)
    return x, y, z

df['x'], df['y'], df['z'] = geodetic_to_ecef(df['latitude_deg'], df['longitude_deg'], df['altitude_km'])

# Flag close approaches (<50 km)
threshold_km = 50
flags = []

for t, group in df.groupby('datetime_utc'):
    names = group['name'].values
    coords = group[['x','y','z']].values
    for (i,j) in combinations(range(len(names)), 2):
        dist = np.linalg.norm(coords[i] - coords[j])
        if dist < threshold_km:
            flags.append([t, names[i], names[j], dist])

df_flags = pd.DataFrame(flags, columns=['datetime_utc','sat1','sat2','distance_km'])
df_flags.to_csv("data/conjunctions.csv", index=False)
print("Conjunctions saved to data/conjunctions.csv")