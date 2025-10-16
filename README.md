# Mini Satellite Monitoring Dashboard
## Overview

A lightweight, interactive dashboard to explore satellite behavior in orbit. It visualizes satellite positions, altitude trends, nearest-neighbor distances, and regional coverage over Europe. Developed as an exploratory project for curiosity and learning, not formal scientific analysis.

## Key Features

- Constellation metrics overview

- Nearest-neighbor distances and close approaches

- Altitude distribution and outlier detection

- Regional coverage visualization

- 3D view of satellites around Earth

## Data Source

- Public orbital data from Celestrak

- TLE files processed with Skyfield

## Technical Stack

- Python

- Streamlit for the interface

- Plotly for interactive charts

- Pandas and Numpy for data processing

## Setup & Run Instructions

### 1. Clone the repository

`git clone https://github.com/your-username/new-repo-name.git`
`cd new-repo-name`

### 2. Create and activate a virtual environment

`python -m venv venv`

Windows --> `venv\Scripts\activate`

Mac/Linux --> `source venv/bin/activate`

### 3. Install dependencies

`pip install -r requirements.txt`


### 4. Run the scripts in order

Step 1: Ingest TLE data. This downloads and saves satellite TLE data.

`python ingest_tle.py`

Step 2: Propagate positions. This generates satellite positions over the desired time range.

`python propagate_positions.py`

Step 3: Launch the dashboard. The dashboard will open in your browser.

`streamlit run dashboard.py`

## Usage

Use the sidebar to select satellites and set the distance threshold

Explore the tabs for positions, distances, altitude, coverage, and 3D visualization

## Notes

Calculations and thresholds are illustrative, not scientifically validated.

Developed for learning and exploration of satellite data and interactive visualization.

## License & Acknowledgements

Open-source and public data only.

Inspired by Vyoma and space situational awareness concepts.