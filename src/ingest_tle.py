# src/ingest_tle.py
import requests
import os

# Create data folder if it doesn't exist
os.makedirs("data", exist_ok=True)

# Galileo TLE URL
TLE_URL = "https://celestrak.org/NORAD/elements/gp.php?GROUP=galileo&FORMAT=tle"
OUT_FILE = "data/tle.txt"

print("Downloading Galileo TLEs...")

resp = requests.get(TLE_URL)
if resp.status_code != 200:
    raise Exception(f"Failed to download TLEs: {resp.status_code}")

# Remove empty lines and strip spaces
lines = [line.strip() for line in resp.text.splitlines() if line.strip()]

# Make sure we have complete 3-line blocks
clean_lines = []
i = 0
while i + 2 < len(lines):
    name = lines[i]
    line1 = lines[i+1]
    line2 = lines[i+2]

    # Check line starts: LINE1 should start with '1', LINE2 with '2'
    if line1.startswith('1') and line2.startswith('2'):
        clean_lines.extend([name, line1, line2])
    else:
        print(f"Skipping malformed TLE block starting with: {name}")
    i += 3

# Save cleaned TLEs
with open(OUT_FILE, "w") as f:
    for l in clean_lines:
        f.write(l + "\n")

print(f"Cleaned Galileo TLEs saved to {OUT_FILE} ({len(clean_lines)//3} satellites)")
