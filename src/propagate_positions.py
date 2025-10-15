# src/propagate_positions.py
import pandas as pd
from skyfield.api import Loader, EarthSatellite, utc
from datetime import datetime, timedelta
import os

os.makedirs("data", exist_ok=True)

load = Loader('~/skyfield_data')
ts = load.timescale()

# Read TLEs
tle_file = "data/tle.txt"
with open(tle_file) as f:
    lines = f.read().splitlines()

satellites = []
for i in range(0, len(lines), 3):
    name = lines[i].strip()
    if i + 2 < len(lines):
        line1 = lines[i+1].strip()
        line2 = lines[i+2].strip()
        satellites.append(EarthSatellite(line1, line2, name, ts))

# Time range: now to 24h, step 10 min
start_time = datetime.utcnow()
end_time = start_time + timedelta(hours=24)
step_minutes = 10
times = pd.date_range(start=start_time, end=end_time, freq=f'{step_minutes}min')

data = []
for sat in satellites[:10]:  # first 10 satellites for simplicity
    for t in times:
        ts_t = ts.utc(t.year, t.month, t.day, t.hour, t.minute, t.second)
        geocentric = sat.at(ts_t)
        lat, lon = geocentric.subpoint().latitude.degrees, geocentric.subpoint().longitude.degrees
        alt = geocentric.subpoint().elevation.km
        data.append([sat.name, t, lat, lon, alt])

df = pd.DataFrame(data, columns=['name', 'datetime_utc', 'latitude_deg', 'longitude_deg', 'altitude_km'])
df.to_csv("data/positions.csv", index=False)
print("Positions saved to data/positions.csv")