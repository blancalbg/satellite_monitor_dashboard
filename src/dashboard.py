# src/dashboard.py
import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import datetime
from scipy.spatial import distance_matrix
import os
import plotly.graph_objects as go
import locale


st.set_page_config(page_title="Satellite Monitor Dashboard - Vyoma", layout="wide")
st.title("🛰️ Satellite Monitor Dashboard – Mini Galileo Constellation Explorer")

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

csv_path = os.path.join(os.path.dirname(__file__), '../data/positions.csv')

df_positions = pd.read_csv(
    csv_path,
    dtype={'altitude_km': float}, 
    parse_dates=['datetime_utc'], 
    thousands=',')

#check the first rows debug in cloud
#st.write(df_positions.head())
#st.write("CSV preview in Cloud:")

# ----------------------
# Compute metrics
# ----------------------

# convert from lat/lon/alt to x/y/z
def geodetic_to_ecef(lat, lon, alt):
    earth_radius = 6371.0
    
    # convert to radians
    lat_r = np.radians(lat)
    lon_r = np.radians(lon)
    
    r = earth_radius + alt
    
    # compute coords
    x = r * np.cos(lat_r) * np.cos(lon_r)
    y = r * np.cos(lat_r) * np.sin(lon_r)
    z = r * np.sin(lat_r)
    
    return np.array([x, y, z]).T

# apply conversion for all positions
positions_xyz = geodetic_to_ecef(
    df_positions['latitude_deg'],
    df_positions['longitude_deg'],
    df_positions['altitude_km']
)
df_positions[['x', 'y', 'z']] = positions_xyz


# nn distances
nearest_rows = []

for time_val, group in df_positions.groupby('datetime_utc'):
    coords = group[['x', 'y', 'z']].values
    names = group['name'].values
    
    # compute distance to nearest other satellite
    for i in range(len(coords)):
        current = coords[i]
        others = np.delete(coords, i, axis=0)
        dist = np.min(np.linalg.norm(others - current, axis=1))
        nearest_rows.append([time_val, names[i], dist])

df_nn = pd.DataFrame(nearest_rows, columns=['datetime_utc', 'name', 'nearest_neighbor_km'])


# altitude stats

alt_stats = (
    df_positions.groupby('name')['altitude_km']
    .agg(['min', 'max', 'mean'])
    .reset_index()
)


# Approx count of passes over Europe

# europe bounding box
def is_in_europe(lat, lon):
    return (35 <= lat <= 70) and (-10 <= lon <= 40)

df_positions['over_europe'] = df_positions.apply(
    lambda row: is_in_europe(row.latitude_deg, row.longitude_deg), axis=1
)

# count how many times each sat is over Europe
coverage = (
    df_positions.groupby('name')['over_europe']
    .sum()
    .reset_index()
    .rename(columns={'over_europe': 'passes_over_europe'})
)


# ----------------------
# Sidebar controls
# ----------------------
st.sidebar.header("Controls & Filters")

# pick which satellites to show
sat_names = df_positions['name'].unique()
selected_sats = st.sidebar.multiselect(
    "Select satellites to display",
    options=sat_names,
    default=sat_names[:10]
)

df_positions = df_positions[df_positions['name'].isin(selected_sats)]


# distance threshold for distances tab
st.sidebar.subheader("Threshold (Distances Tab Only)")
threshold_km = st.sidebar.number_input(
    "Highlight distances below (km)",
    value=5000
)
st.sidebar.caption("Used only in the Nearest-Neighbor Distances tab.")


# summary info
st.sidebar.markdown("---")
st.sidebar.subheader("Quick Stats")

st.sidebar.markdown(f"- Total satellites: **{len(selected_sats)}**")

if not df_positions.empty:
    avg_alt = df_positions['altitude_km'].mean()
    min_nn = df_nn['nearest_neighbor_km'].min()
    max_nn = df_nn['nearest_neighbor_km'].max()

    st.sidebar.markdown(f"- Avg altitude: **{avg_alt:.0f} km**")
    st.sidebar.markdown(f"- Nearest neighbor min: **{min_nn:.0f} km**")
    st.sidebar.markdown(f"- Nearest neighbor max: **{max_nn:.0f} km**")

# small safety filter again
df_positions = df_positions[df_positions['name'].isin(selected_sats)]


# ----------------------
# Streamlit tabs
# ----------------------
tabs = st.tabs(["Overview", "Distances", "Altitude", "Coverage", "Positions 2D"])

# Overview tab
with tabs[0]:
    st.subheader("🛰️ Constellation Overview")

    summary = alt_stats.merge(
        df_nn.groupby('name')['nearest_neighbor_km'].mean().reset_index(),
        on='name'
    )
    summary = summary.merge(coverage, on='name')

    summary["stability_index"] = (summary["max"] - summary["min"]) / summary["mean"]
    summary["risk_score"] = np.exp(-summary["nearest_neighbor_km"] / 100)

    # helper to roughly tag orbit quality
    def classify_orbit(alt_km):
        if 22800 <= alt_km <= 23600:
            return "Nominal"
        elif 22000 <= alt_km <= 24000:
            return "Slight Drift"
        else:
            return "Out of Range"

    summary["orbit_health"] = summary["mean"].apply(classify_orbit)

    avg_alt = summary["mean"].mean()
    avg_sep = summary["nearest_neighbor_km"].mean()

    st.markdown(
        f"**Avg Orbit Altitude:** {avg_alt:.0f} km | "
        f"**Avg Separation:** {avg_sep:.0f} km  \n"
        f"**Satellites:** {len(summary)} | "
        f"**Nominal Band (Galileo):** 22,800–23,600 km"
    )

    # info box
    with st.expander("ℹ️ What these metrics mean"):
        st.markdown("""
        - **Min / Max / Mean Altitude:** Altitude range for each satellite.  
        - **Stability Index:** How much the altitude changes (lower = steadier).  
        - **Avg Nearest:** Typical distance to the closest other satellite.  
        - **Risk Score:** Closeness factor `exp(-nearest_neighbor_km / 100)` — bigger means more risk.  
        - **Passes over Europe:** How often the sat crosses 35–70°N, –10–40°E.
        """)

    # table formatting
    def color_orbit(val):
        if val == "Nominal":
            return "color: #2ecc71; font-weight: 600;"
        elif val == "Slight Drift":
            return "color: #f1c40f; font-weight: 600;"
        else:
            return "color: #e74c3c; font-weight: 600;"

    def color_risk(val):
        if val < 0.1:
            return "color: #2ecc71;"
        elif val < 0.5:
            return "color: #f39c12;"
        else:
            return "color: #e74c3c;"

    # rename columns
    summary = summary.rename(columns={
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

    styled_df = (
        summary.style
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

# Distances tab
with tabs[1]:
    st.subheader("Nearest-Neighbor Distances")

    min_dist = df_nn['nearest_neighbor_km'].min()
    mean_dist = df_nn['nearest_neighbor_km'].mean()
    below_threshold = (df_nn['nearest_neighbor_km'] < threshold_km).mean() * 100

    st.markdown(f"""
    **Summary Metrics**
    - Min distance seen: {min_dist:.2f} km  
    - Avg nearest distance: {mean_dist:.2f} km  
    - Close approaches (< {threshold_km} km): {below_threshold:.1f}%  
    *(Threshold only matters on this tab)*
    """)

    # info box
    with st.expander("ℹ️ About this view"):
        st.markdown("""
        Each point shows how close a satellite got to its nearest neighbor over time.

        - **Sharp dips** → short proximity events (possible conjunctions).  
        - **Flat stretches** → stable spacing.  
        - **Wide histograms** → variable separation patterns.  

        Anything below the red line is below the threshold — i.e. close approaches.
        """)

    # line chart distance vs time
    fig = px.line(
        df_nn,
        x='datetime_utc',
        y='nearest_neighbor_km',
        color='name',
        template="plotly_dark"
    )
    fig.update_layout(
        yaxis_title="Nearest Neighbor Distance (km)",
        xaxis_title="Time (UTC)",
        title="Distance to Nearest Satellite Over Time"
    )
    fig.add_hline(
        y=threshold_km,
        line_dash="dot",
        line_color="red",
        annotation_text="Threshold",
        annotation_position="bottom right"
    )
    st.plotly_chart(fig, use_container_width=True)

    # zoomed chart. threshold 
    df_zoom = df_nn[df_nn['nearest_neighbor_km'] < threshold_km * 2].copy()
    if not df_zoom.empty:
        df_zoom['color'] = np.where(
            df_zoom['nearest_neighbor_km'] < threshold_km,
            'Below Threshold',
            'Normal'
        )
        fig_zoom = px.scatter(
            df_zoom,
            x='datetime_utc',
            y='nearest_neighbor_km',
            color='color',
            hover_name='name',
            template="plotly_dark",
            color_discrete_map={'Below Threshold': 'red', 'Normal': 'blue'}
        )
        fig_zoom.update_layout(
            yaxis_title="Nearest Neighbor Distance (km)",
            xaxis_title="Time (UTC)",
            title=f"Zoomed: Distances Below {threshold_km * 2:.0f} km"
        )
        fig_zoom.add_hline(
            y=threshold_km,
            line_dash="dot",
            line_color="red",
            annotation_text="Threshold",
            annotation_position="bottom right"
        )
        st.plotly_chart(fig_zoom, use_container_width=True)
    else:
        st.info("No points close enough to show in the zoomed view.")

    # table close approach details
    st.subheader(f"Close Approaches (< {threshold_km} km)")
    df_close = df_nn[df_nn['nearest_neighbor_km'] < threshold_km].copy()

    if not df_close.empty:
        nearest_list = []
        for _, row in df_close.iterrows():
            t = row['datetime_utc']
            sat_name = row['name']

            same_time = df_positions[df_positions['datetime_utc'] == t]
            coords = same_time[['x', 'y', 'z']].values
            names = same_time['name'].values

            i = np.where(names == sat_name)[0][0]
            others = np.delete(coords, i, axis=0)
            other_names = np.delete(names, i)

            if len(others) > 0:
                dists = np.linalg.norm(others - coords[i], axis=1)
                nearest_name = other_names[np.argmin(dists)]
                nearest_dist = dists.min()
            else:
                nearest_name = None
                nearest_dist = None

            nearest_list.append([t, sat_name, nearest_name, nearest_dist])

        df_table = pd.DataFrame(
            nearest_list,
            columns=['datetime_utc', 'satellite', 'nearest_satellite', 'distance_km']
        )
        st.dataframe(df_table)
    else:
        st.info("No close approaches found under the current threshold.")

    # histogram
    st.subheader("Distance Distribution")
    fig_hist = px.histogram(
        df_nn,
        x='nearest_neighbor_km',
        nbins=30,
        color='name',
        barmode='overlay',
        template="plotly_dark"
    )
    fig_hist.update_layout(
        xaxis_title="Distance (km)",
        yaxis_title="Frequency",
        title="Distribution of Nearest-Neighbor Distances"
    )
    st.plotly_chart(fig_hist, use_container_width=True)


# Altitude tab
with tabs[2]:
    st.subheader("Altitude Over Time")

    # outlier check IQR
    alt_means = df_positions.groupby('name')['altitude_km'].mean()
    q1, q3 = alt_means.quantile([0.25, 0.75])
    iqr = q3 - q1
    lower = q1 - 1.5 * iqr
    upper = q3 + 1.5 * iqr

    outliers = alt_means[(alt_means < lower) | (alt_means > upper)].index.tolist()

    df_normal = df_positions[~df_positions['name'].isin(outliers)]
    df_outliers = df_positions[df_positions['name'].isin(outliers)]

    if outliers:
        st.markdown(f"Found **{len(outliers)} outlier satellites**: {', '.join(outliers)}")
    else:
        st.markdown("No major altitude outliers detected.")

    # altitude plot
    st.subheader("Nominal Altitude Profiles")
    fig_alt = px.line(
        df_normal,
        x='datetime_utc',
        y='altitude_km',
        color='name',
        template="plotly_dark"
    )
    fig_alt.update_layout(
        yaxis_title="Altitude (km)",
        xaxis_title="Time (UTC)",
        title="Altitude Evolution (excluding outliers)"
    )
    st.plotly_chart(fig_alt, use_container_width=True)

    # outlier altitude plot
    if len(outliers) > 0:
        st.subheader("Outlier Altitude Profiles")
        fig_out = px.line(
            df_outliers,
            x='datetime_utc',
            y='altitude_km',
            color='name',
            markers=True,
            template="plotly_dark"
        )
        fig_out.update_layout(
            yaxis_title="Altitude (km)",
            xaxis_title="Time (UTC)",
            title="Satellites with Unusual Altitude Changes"
        )
        st.plotly_chart(fig_out, use_container_width=True)

    # summary table
    st.subheader("Altitude Statistics")
    alt_table = (
        df_positions.groupby('name')
        .agg(
            min_alt=('altitude_km', 'min'),
            max_alt=('altitude_km', 'max'),
            mean_alt=('altitude_km', 'mean'),
            std_alt=('altitude_km', 'std')
        )
        .reset_index()
    )

    st.dataframe(
        alt_table.style.format({
            "min_alt": "{:.2f}",
            "max_alt": "{:.2f}",
            "mean_alt": "{:.2f}",
            "std_alt": "{:.2f}"
        }),
        use_container_width=True
    )

# Coverage tab
with tabs[3]:
    st.subheader("Regional Coverage: Passes Over Europe")

    # figure out when satellites are over Europe
    df_positions = df_positions.sort_values(['name', 'datetime_utc']).copy()
    df_positions['pass_change'] = (
        df_positions.groupby('name')['over_europe'].diff().fillna(0).ne(0)
    )
    df_positions['pass_id'] = (
        df_positions.groupby('name')['pass_change'].cumsum()
    )

    # count unique passes. continuous over-Europe segments
    passes = (
        df_positions[df_positions['over_europe']]
        .groupby('name')['pass_id']
        .nunique()
        .reset_index()
        .rename(columns={'pass_id': 'distinct_passes'})
    )

    # how much time each sat spends over Europe %
    coverage_pct = (
        df_positions.groupby('name')['over_europe']
        .mean()
        .reset_index()
        .rename(columns={'over_europe': 'coverage_percent'})
    )
    coverage_pct['coverage_percent'] *= 100

    # combine both stats
    coverage_stats = (
        passes.merge(coverage_pct, on='name')
        .sort_values(by='distinct_passes', ascending=False)
    )

    st.markdown(
        "Each **pass** means a continuous stretch where the satellite stays inside the European region bounds."
    )

    # info box
    with st.expander("ℹ️ What does 'coverage %' mean?"):
        st.markdown("""
        **Coverage Percentage** = part of the total time a satellite spends above Europe.

        - Formula: `(timestamps over Europe) / (total timestamps) × 100`  
        - 100% → always above Europe  
        - 0% → never crosses it  
        - Anything between → partial regional visibility  

        Handy for checking how available each satellite is for regional observations or comms.
        """)

    # bar chart
    fig_cov = px.bar(
        coverage_stats,
        x='name',
        y='distinct_passes',
        color='coverage_percent',
        text=coverage_stats['coverage_percent'].apply(lambda x: f"{x:.1f}%"),
        color_continuous_scale='Blues',
        template="plotly_dark",
        title="Pass Counts and Average Coverage Over Europe"
    )
    fig_cov.update_layout(
        yaxis_title="Distinct Passes",
        coloraxis_colorbar_title="Coverage (%)"
    )
    st.plotly_chart(fig_cov, use_container_width=True)

    # summary table
    st.subheader("Coverage Summary")
    st.dataframe(
        coverage_stats.style.format({
            "distinct_passes": "{:.0f}",
            "coverage_percent": "{:.2f}"
        }),
        use_container_width=True
    )


# 3D Positions tab
with tabs[4]:
    st.subheader("3D Orbit View (Satellites + Earth)")

    # most recent time step for plotting
    latest_time = df_positions['datetime_utc'].max()
    df_latest = df_positions[df_positions['datetime_utc'] == latest_time].copy()

    # nn distance for hover
    coords = df_latest[['x', 'y', 'z']].values
    dist_matrix = distance_matrix(coords, coords)
    np.fill_diagonal(dist_matrix, np.inf)
    df_latest['nearest_neighbor_km'] = dist_matrix.min(axis=1)

    # earth sphere
    R_earth = 6371  # km
    phi, theta = np.mgrid[0:np.pi:50j, 0:2*np.pi:50j]
    x_sphere = R_earth * np.sin(phi) * np.cos(theta)
    y_sphere = R_earth * np.sin(phi) * np.sin(theta)
    z_sphere = R_earth * np.cos(phi)

    earth = go.Surface(
        x=x_sphere, y=y_sphere, z=z_sphere,
        colorscale=[[0, 'rgb(10,10,150)'], [1, 'rgb(10,50,200)']],
        opacity=0.7,
        showscale=False
    )

    # scatter plot satellite positions
    sats = go.Scatter3d(
        x=df_latest['x'],
        y=df_latest['y'],
        z=df_latest['z'],
        mode='markers',
        marker=dict(
            size=5,
            color=df_latest['altitude_km'],
            colorscale='Viridis',
            colorbar=dict(title='Altitude (km)')
        ),
        text=df_latest['name'] + '<br>Nearest: ' + df_latest['nearest_neighbor_km'].round(2).astype(str) + ' km',
        hoverinfo='text'
    )

    fig_3d = go.Figure(data=[earth, sats])
    fig_3d.update_layout(
        scene=dict(
            xaxis_title='X (km)',
            yaxis_title='Y (km)',
            zaxis_title='Z (km)',
            aspectmode='data',
            bgcolor="rgb(11,13,23)"
        ),
        margin=dict(l=0, r=0, b=0, t=30),
        title=f"Satellite Positions (latest at {latest_time})"
    )

    # info box
    with st.expander("ℹ️ How to read this view"):
        st.markdown("""
        - Blue sphere = Earth  
        - Dots = satellites (colored by altitude)  
        - Hover to see nearest-neighbor distance  
        - Drag to rotate / zoom in the 3D plot  
        """)

    st.plotly_chart(fig_3d, use_container_width=True)