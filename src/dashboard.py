# src/dashboard.py
import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import datetime
from scipy.spatial import distance_matrix
import os
import plotly.graph_objects as go


st.set_page_config(page_title="Mini Satellite Monitoring Dashboard", layout="wide")
st.title("🛰️ Mini Satellite Monitoring Dashboard (Galileo)")

# ----------------------
# Create dark-mode view
# ----------------------
st.markdown(
    """
    <style>
    /* Page background */
    .stApp {
        background-color: #0B0D17;  /* Dark blue-black */
        color: #FFFFFF;              /* Default text color */
    }

    /* Sidebar background */
    .css-1d391kg { 
        background-color: #1A1A2E !important; 
        color: #FFFFFF !important;
    }

    /* Tables */
    .dataframe, .stDataFrame {
        background-color: #111827 !important;
        color: #FFFFFF !important;
    }

    /* Expanders */
    .streamlit-expanderHeader {
        color: #FFFFFF;
    }

    /* Headings */
    h1, h2, h3, h4 {
        color: #FFFFFF;
    }
    </style>
    """,
    unsafe_allow_html=True
)


# ----------------------
# Load data
# ----------------------

# Absolute path relative to this script
csv_path = os.path.join(os.path.dirname(__file__), '../data/positions.csv')

# Read CSV safely
df_positions = pd.read_csv(
    csv_path,
    dtype={'altitude_km': float},  # Force float type for altitude
    parse_dates=['datetime_utc'],  # Ensure datetime is recognized
    thousands=','                 # Handle any thousands separators
)

# Optional: check the first few rows to debug in cloud
#st.write(df_positions.head())
#st.write("CSV preview in Cloud:")
#st.dataframe(df_positions.head())
#st.write(df_positions.dtypes)

# ----------------------
# Compute derived metrics
# ----------------------

# 1. Nearest-neighbor distances
def geodetic_to_ecef(lat, lon, alt):
    EARTH_R = 6371.0
    lat = np.deg2rad(lat)
    lon = np.deg2rad(lon)
    r = EARTH_R + alt
    x = r * np.cos(lat) * np.cos(lon)
    y = r * np.cos(lat) * np.sin(lon)
    z = r * np.sin(lat)
    return np.array([x, y, z]).T

positions_ecef = geodetic_to_ecef(df_positions['latitude_deg'], df_positions['longitude_deg'], df_positions['altitude_km'])
df_positions[['x','y','z']] = positions_ecef

st.write("✅ Sample of df_positions:")
st.dataframe(df_positions.head(5))

st.write("🛰 Number of satellites:", df_positions['name'].nunique())
st.write("📅 Number of timestamps:", df_positions['datetime_utc'].nunique())
st.write("🔝 Altitude stats (km):")
st.write(df_positions['altitude_km'].describe())

# Nearest neighbor distances per satellite per timestamp
nearest_distances = []
for t, group in df_positions.groupby('datetime_utc'):
    coords = group[['x','y','z']].values
    names = group['name'].values
    for i, name in enumerate(names):
        others = np.delete(coords, i, axis=0)
        dist = np.linalg.norm(others - coords[i], axis=1).min()
        nearest_distances.append([t, name, dist])

df_nn = pd.DataFrame(nearest_distances, columns=['datetime_utc','name','nearest_neighbor_km'])

st.write("🔍 Nearest-neighbor distances sample:")
st.dataframe(df_nn.head(10))

st.write("📏 Distance stats (km):")
st.write(df_nn['nearest_neighbor_km'].describe())

# 2. Altitude stats
alt_stats = df_positions.groupby('name')['altitude_km'].agg(['min','max','mean']).reset_index()

# 3. Optional: simulate coverage (count of passes over Europe)
# Europe box: lat 35-70, lon -10 to 40
def in_europe(lat, lon):
    return (35 <= lat <= 70) & (-10 <= lon <= 40)

df_positions['over_europe'] = df_positions.apply(lambda r: in_europe(r.latitude_deg, r.longitude_deg), axis=1)
coverage = df_positions.groupby('name')['over_europe'].sum().reset_index().rename(columns={'over_europe':'passes_over_europe'})


# ----------------------
# Sidebar controls
# ----------------------
st.sidebar.header("Controls & Filters")

# Select satellites to display
all_sat_names = df_positions['name'].unique()
selected_sats = st.sidebar.multiselect(
    "Select satellites to display", 
    options=all_sat_names,
    default=all_sat_names[:10]  # default top 10
)
df_positions = df_positions[df_positions['name'].isin(selected_sats)]

# Close approach threshold (optional for highlighting distances)
st.sidebar.subheader("Threshold (Distances Tab Only)")
threshold_km = st.sidebar.number_input(
    "Highlight distances below (km)", 
    value=5000
)
st.sidebar.caption("This only affects the Nearest-Neighbor Distances tab.")

# Quick stats summary
st.sidebar.markdown("---")
st.sidebar.subheader("Quick Stats")
st.sidebar.markdown(f"- Total satellites: **{len(selected_sats)}**")
if len(df_positions) > 0:
    st.sidebar.markdown(f"- Avg altitude: **{df_positions['altitude_km'].mean():.0f} km**")
    st.sidebar.markdown(f"- Min nearest neighbor: **{df_nn['nearest_neighbor_km'].min():.0f} km**")
    st.sidebar.markdown(f"- Max nearest neighbor: **{df_nn['nearest_neighbor_km'].max():.0f} km**")

# Filter satellites based on sidebar multiselect
df_positions = df_positions[df_positions['name'].isin(selected_sats)]


# ----------------------
# Streamlit tabs
# ----------------------
tabs = st.tabs(["Overview", "Distances", "Altitude", "Coverage", "Positions 2D"])

# ----- Overview Tab -----
with tabs[0]:
    st.subheader("🛰️ Constellation Overview")

    # Merge the computed metrics
    summary = alt_stats.merge(df_nn.groupby('name')['nearest_neighbor_km'].mean().reset_index(), on='name')
    summary = summary.merge(coverage, on='name')

    # Compute extra indicators
    summary["stability_index"] = (summary["max"] - summary["min"]) / summary["mean"]
    summary["risk_score"] = np.exp(-summary["nearest_neighbor_km"] / 100)

    def classify_orbit(alt_km):
        if 22800 <= alt_km <= 23600:
            return "Nominal"
        elif 22000 <= alt_km <= 24000:
            return "Slight Drift"
        else:
            return "Out of Range"

    summary["orbit_health"] = summary["mean"].apply(classify_orbit)

    # Header summary
    avg_alt = summary["mean"].mean()
    avg_sep = summary["nearest_neighbor_km"].mean()

    st.markdown(f"""
    **Average Orbit Altitude:** {avg_alt:.0f} km | **Average Separation:** {avg_sep:.0f} km  
    **Satellites Monitored:** {len(summary)} | **Nominal Orbit Band (Galileo):** 22,800–23,600 km
    """)

    with st.expander("ℹ️ Metric definitions"):
        st.markdown("""
        - **Min / Max / Mean Altitude (km):** Statistical range of orbital altitude across the propagation window.  
        - **Stability Index:** Normalized altitude variation — computed as `(max_altitude - min_altitude) / mean_altitude`.  
        Lower values indicate a more stable orbit.  
        - **Avg Nearest (km):** Mean distance to the closest satellite at each timestamp, reflecting constellation spacing.  
        - **Risk Score:** Exponential proximity indicator `exp(-nearest_neighbor_km / 100)` — higher values signal potential close-approach events.  
        - **Passes over Europe:** Number of times the satellite’s ground track crossed the region (lat 35–70°, lon –10–40°).
        """)

    # Conditional formatting
    def color_orbit(val):
        if val == "Nominal":
            color = "#2ecc71"  # green
        elif val == "Slight Drift":
            color = "#f1c40f"  # yellow
        else:
            color = "#e74c3c"  # red
        return f"color: {color}; font-weight: 600;"

    def color_risk(val):
        if val < 0.1:
            return "color: #2ecc71;"
        elif val < 0.5:
            return "color: #f39c12;"
        else:
            return "color: #e74c3c;"

    styled_df = (
        summary.rename(columns={
            "name": "Satellite",
            "min": "Min Alt (km)",
            "max": "Max Alt (km)",
            "mean": "Mean Alt (km)",
            "nearest_neighbor_km": "Avg Nearest (km)",
            "stability_index": "Stability Index",
            "risk_score": "Risk Score",
            "passes_over_europe": "Passes over Europe",
            "orbit_health": "Orbit Health"
        })
        .style
        .map(color_orbit, subset=["Orbit Health"])
        .map(color_risk, subset=["Risk Score"])
        .format({
            "Min Alt (km)": "{:.0f}",
            "Max Alt (km)": "{:.0f}",
            "Mean Alt (km)": "{:.0f}",
            "Avg Nearest (km)": "{:.0f}",
            "Stability Index": "{:.3f}",
            "Risk Score": "{:.3f}"
        })
    )

    st.dataframe(styled_df, use_container_width=True, hide_index=True)
    

# ----- Distances Tab -----
with tabs[1]:
    st.subheader("Nearest-Neighbor Distances")

    # ---------- Summary metrics ----------
    min_dist = df_nn['nearest_neighbor_km'].min()
    mean_dist = df_nn['nearest_neighbor_km'].mean()
    below_threshold = (df_nn['nearest_neighbor_km'] < threshold_km).mean() * 100

    st.markdown(f"""
    **Summary Metrics**
    - **Minimum distance observed:** {min_dist:.2f} km  
    - **Average nearest distance:** {mean_dist:.2f} km  
    - **% of close approaches (< {threshold_km} km):** {below_threshold:.1f}%  
    *(Threshold only applies to this tab)*
    """)

    # ---------- Explanation ----------
    with st.expander("ℹ️ Understanding this view"):
        st.markdown("""
        Each point represents the **distance to the nearest satellite** at a given timestamp.

        - **Sharp drops** → temporary proximity events or potential conjunctions.  
        - **Flat lines** → stable orbital spacing.  
        - **Wide histograms** → satellites with irregular distances.  

        Values below the threshold (default: 50 km) indicate *potentially risky close approaches*.
        """)

    # ---------- Full-range line chart ----------
    fig = px.line(df_nn, x='datetime_utc', y='nearest_neighbor_km', color='name', markers=False, template="plotly_dark")
    fig.update_layout(
    yaxis=dict(
        title="Nearest Neighbor Distance (km)",
        tickformat=".0f"  # Force plain numeric format (no commas)
    ),
    xaxis=dict(
        title="Time (UTC)"
    ),
    title="Distance to Nearest Satellite Over Time"
    )
    fig.add_hline(y=threshold_km, line_dash="dot", line_color="red",
                  annotation_text="Threshold", annotation_position="bottom right")
    st.plotly_chart(fig, use_container_width=True)

    # ---------- Zoomed-in scatter plot ----------
    df_zoom = df_nn[df_nn['nearest_neighbor_km'] < threshold_km*2].copy()
    if not df_zoom.empty:
        df_zoom['color'] = np.where(df_zoom['nearest_neighbor_km'] < threshold_km, 'Below Threshold', 'Normal')
        fig_zoom = px.scatter(
            df_zoom,
            x='datetime_utc',
            y='nearest_neighbor_km',
            color='color',
            hover_name='name',
            template="plotly_dark",
            color_discrete_map={'Below Threshold':'red', 'Normal':'blue'}
        )
        fig_zoom.update_layout(
            yaxis_title="Nearest Neighbor Distance (km)",
            xaxis_title="Time (UTC)",
            title=f"Zoom: Distances Near Threshold (< {threshold_km*2} km)"
        )
        fig_zoom.add_hline(y=threshold_km, line_dash="dot", line_color="red",
                           annotation_text="Threshold", annotation_position="bottom right")
        st.plotly_chart(fig_zoom, use_container_width=True)
    else:
        st.info("No distances below the zoom threshold to display.")

    # ---------- Table of close approaches ----------
    st.subheader(f"Close Approaches (< {threshold_km} km)")
    df_close = df_nn[df_nn['nearest_neighbor_km'] < threshold_km].copy()

    if not df_close.empty:
        nearest_list = []
        for idx, row in df_close.iterrows():
            t = row['datetime_utc']
            sat_name = row['name']
            # Get positions at that timestamp
            group = df_positions[df_positions['datetime_utc'] == t]
            coords = group[['x','y','z']].values
            names = group['name'].values
            i = np.where(names == sat_name)[0][0]
            others = np.delete(coords, i, axis=0)
            other_names = np.delete(names, i)
            if len(others) > 0:
                dist = np.linalg.norm(others - coords[i], axis=1)
                nearest_name = other_names[np.argmin(dist)]
                nearest_dist = dist.min()
            else:
                nearest_name = None
                nearest_dist = None
            nearest_list.append([t, sat_name, nearest_name, nearest_dist])

        df_table = pd.DataFrame(nearest_list, columns=['datetime_utc','satellite','nearest_satellite','distance_km'])
        st.dataframe(df_table)
    else:
        st.info("No close approaches below threshold currently.")

    # ---------- Histogram ----------
    st.subheader("Distance Distribution")
    fig_hist = px.histogram(df_nn, x='nearest_neighbor_km', nbins=30, color='name',
                            barmode='overlay', template="plotly_dark")
    fig_hist.update_layout(
        xaxis_title="Distance (km)",
        yaxis_title="Frequency",
        title="Distribution of Nearest-Neighbor Distances"
    )
    st.plotly_chart(fig_hist, use_container_width=True)


# ----- Altitude Tab -----

with tabs[2]:
    st.subheader("Altitude Over Time")

    # Detect outliers: satellites with unusually high or low mean altitude
    alt_summary = df_positions.groupby('name')['altitude_km'].mean()
    q1, q3 = alt_summary.quantile([0.25, 0.75])
    iqr = q3 - q1
    upper_bound = q3 + 1.5 * iqr
    lower_bound = q1 - 1.5 * iqr

    outliers = alt_summary[(alt_summary > upper_bound) | (alt_summary < lower_bound)].index.tolist()
    df_normal = df_positions[~df_positions['name'].isin(outliers)]
    df_outliers = df_positions[df_positions['name'].isin(outliers)]

    st.markdown(f"Detected **{len(outliers)} outlier satellites** based on altitude range: {', '.join(outliers)}")

    # --- Main altitude plot (normal satellites) ---
    st.subheader("Nominal Altitude Profiles")
    fig_alt = px.line(df_normal, x='datetime_utc', y='altitude_km', color='name', markers=False, template="plotly_dark")
    fig_alt.update_layout(
        yaxis_title="Altitude (km)",
        xaxis_title="Time (UTC)",
        title="Altitude Evolution (excluding outliers)"
    )
    st.plotly_chart(fig_alt, use_container_width=True)

    # --- Outlier altitude plot ---
    if len(outliers) > 0:
        st.subheader("Outlier Altitude Profiles")
        fig_out = px.line(df_outliers, x='datetime_utc', y='altitude_km', color='name', markers=True, template="plotly_dark")
        fig_out.update_layout(
            yaxis_title="Altitude (km)",
            xaxis_title="Time (UTC)",
            title="Satellites with Extreme Altitude Variations"
        )
        st.plotly_chart(fig_out, use_container_width=True)

    # --- Altitude summary table ---
    st.subheader("Altitude Statistics Table")
    alt_table = df_positions.groupby('name').agg(
        min_altitude_km=('altitude_km', 'min'),
        max_altitude_km=('altitude_km', 'max'),
        mean_altitude_km=('altitude_km', 'mean'),
        std_altitude_km=('altitude_km', 'std')
    ).reset_index()
    st.dataframe(alt_table.style.format({
        "min_altitude_km": "{:.2f}",
        "max_altitude_km": "{:.2f}",
        "mean_altitude_km": "{:.2f}",
        "std_altitude_km": "{:.2f}"
    }))

# ----- Coverage Tab -----
with tabs[3]:
    st.subheader("Regional Coverage: Passes Over Europe")

    # Detect passes (contiguous over_europe sequences)
    df_positions = df_positions.sort_values(['name', 'datetime_utc'])
    df_positions['pass_change'] = df_positions.groupby('name')['over_europe'].diff().fillna(0).ne(0)
    df_positions['pass_id'] = df_positions.groupby('name')['pass_change'].cumsum()
    passes = df_positions[df_positions['over_europe']].groupby('name')['pass_id'].nunique().reset_index()
    passes.columns = ['name', 'distinct_passes']

    # Coverage percentage
    coverage_percent = df_positions.groupby('name')['over_europe'].mean().reset_index()
    coverage_percent['coverage_percent'] = coverage_percent['over_europe'] * 100
    coverage_percent = coverage_percent.drop(columns='over_europe')

    coverage_stats = passes.merge(coverage_percent, on='name').sort_values(by='distinct_passes', ascending=False)

    st.markdown("Each **pass** is a continuous time segment when a satellite is above European lat/lon bounds.")

     # Info box
    with st.expander("ℹ️ What does coverage percentage mean?"):
        st.markdown("""
        **Coverage Percentage:** Fraction of total time a satellite spends over Europe.

        - Calculated as:  
          `(Number of timestamps over Europe) / (Total timestamps for the satellite) × 100`  
        - 100% → satellite always over Europe  
        - 0% → never over Europe  
        - Intermediate values → satellite’s observational availability over the region  

        Useful to quantify **regional accessibility** for observations, communication, or monitoring tasks.
        """)

    fig_cov = px.bar(
        coverage_stats,
        x='name', y='distinct_passes',
        color='coverage_percent',
        text=coverage_stats['coverage_percent'].apply(lambda x: f"{x:.1f}%"),
        color_continuous_scale='Blues',
        template="plotly_dark",
        title="Distinct Passes and Average Coverage Time Over Europe"
    )
    fig_cov.update_layout(yaxis_title="Distinct Passes", coloraxis_colorbar_title="Coverage (%)")
    st.plotly_chart(fig_cov, use_container_width=True)

    # Table
    st.subheader("Coverage Summary Table")
    st.dataframe(coverage_stats.style.format({
        "distinct_passes": "{:.0f}",
        "coverage_percent": "{:.2f}"
    }))


# ----- Positions 2D Tab -----
with tabs[4]:
    st.subheader("Satellite 3D Orbit Positions with Earth")

    latest_time = df_positions['datetime_utc'].max()
    df_latest = df_positions[df_positions['datetime_utc'] == latest_time].copy()

    # Compute nearest-neighbor distance for hover
    coords = df_latest[['x','y','z']].values
    dist_matrix = distance_matrix(coords, coords)
    np.fill_diagonal(dist_matrix, np.inf)  # ignore self-distance
    df_latest['nearest_neighbor_km'] = dist_matrix.min(axis=1)

    # ----- Create Earth sphere -----
    radius_earth = 6371  # km
    phi, theta = np.mgrid[0.0:np.pi:50j, 0.0:2.0*np.pi:50j]
    x_sphere = radius_earth * np.sin(phi) * np.cos(theta)
    y_sphere = radius_earth * np.sin(phi) * np.sin(theta)
    z_sphere = radius_earth * np.cos(phi)

    earth_sphere = go.Surface(
        x=x_sphere, y=y_sphere, z=z_sphere,
        colorscale=[[0, 'rgb(10,10,150)'], [1, 'rgb(10,50,200)']],
        opacity=0.7,
        showscale=False
    )

    # ----- Satellite positions -----
    sat_scatter = go.Scatter3d(
        x=df_latest['x'],
        y=df_latest['y'],
        z=df_latest['z'],
        mode='markers',
        marker=dict(
            size=5,  # fixed size
            color=df_latest['altitude_km'],  # color represents altitude
            colorscale='Viridis',
            colorbar=dict(title='Altitude (km)'),
        ),
        text=df_latest['name'] + '<br>Nearest neighbor: ' + df_latest['nearest_neighbor_km'].round(2).astype(str) + ' km',
        hoverinfo='text'
    )

    fig_3d = go.Figure(data=[earth_sphere, sat_scatter])
    fig_3d.update_layout(
        scene=dict(
            xaxis=dict(title='X (km)'),
            yaxis=dict(title='Y (km)'),
            zaxis=dict(title='Z (km)'),
            aspectmode='data',
            bgcolor="rgb(11,13,23)"
        ),
        margin=dict(l=0, r=0, b=0, t=30),
        title=f"Satellite 3D Positions at {latest_time}"
    )

    # Info box
    with st.expander("ℹ️ How to read this 3D orbit view"):
        st.markdown("""
        - Blue sphere = Earth.
        - Each point = satellite at latest timestamp.
        - Marker color = altitude (higher = brighter).
        - Hover over a point to see nearest neighbor distance.
        - Drag the plot to rotate and inspect orbits interactively.
        """)

    st.plotly_chart(fig_3d, use_container_width=True)