# gui_app.py
import streamlit as st
import pandas as pd
import geopandas as gpd
import folium
from shapely.geometry import Point
from shapely import wkb
import binascii
from folium.plugins import HeatMap
from streamlit_folium import st_folium
from matplotlib import colormaps
from matplotlib.colors import to_hex
import time
import os

# Local .pkl source
DATA_DIR = "data_exports"
ENDPOINTS = {
    "astar_routes": os.path.join(DATA_DIR, "astar_routes.pkl"),
    "mapf_routes": os.path.join(DATA_DIR, "mapf_routes.pkl"),
    "user_patterns": os.path.join(DATA_DIR, "user_patterns.pkl"),
    "hotspots": os.path.join(DATA_DIR, "hotspots.pkl"),
    "view_latest_client_trajectories": os.path.join(DATA_DIR, "view_latest_client_trajectories.pkl"),
    "predicted_pois_sequence": os.path.join(DATA_DIR, "predicted_pois_sequence.pkl")
}

def fetch_full(endpoint_name):
    try:
        return pd.read_pickle(ENDPOINTS[endpoint_name])
    except Exception as e:
        st.error(f"Failed to load {endpoint_name}.pkl: {e}")
        return pd.DataFrame()

def parse_path(path_wkb_hex):
    try:
        return wkb.loads(binascii.unhexlify(path_wkb_hex), hex=True)
    except Exception:
        return None

def build_point_gdf(df):
    if "lat" in df.columns and "lon" in df.columns:
        df = df.dropna(subset=["lat", "lon"])
        geometry = [Point(lon, lat) for lon, lat in zip(df["lon"], df["lat"])]
    elif "predicted_lat" in df.columns and "predicted_lon" in df.columns:
        df = df.dropna(subset=["predicted_lat", "predicted_lon"])
        geometry = [Point(lon, lat) for lon, lat in zip(df["predicted_lon"], df["predicted_lat"])]
    elif "geometry" in df.columns:
        geometry = gpd.GeoSeries.from_wkt(df["geometry"])
    else:
        raise ValueError("Missing coordinates or geometry column.")
    return gpd.GeoDataFrame(df, geometry=geometry, crs="EPSG:4326")

# UI Init
st.set_page_config(layout="wide")
st.title("UrbanOS Result Viewer")

option = st.sidebar.selectbox("Choose dataset", list(ENDPOINTS.keys()) + ["compare_routes"])

# Pattern Filter Logic
client_filter_applies = option in {"astar_routes", "mapf_routes", "view_latest_client_trajectories", "predicted_pois_sequence"}
client_ids = None

if client_filter_applies:
    df_patterns = fetch_full("user_patterns").dropna(subset=["client_id", "pattern_type"])
    grouped = df_patterns.groupby("pattern_type")["client_id"].apply(list).to_dict()
    selected_pattern = st.sidebar.selectbox("Pattern type:", sorted(grouped))
    selected_client = st.sidebar.selectbox("Client ID:", ["Show All"] + sorted(grouped[selected_pattern]))
    client_ids = grouped[selected_pattern] if selected_client == "Show All" else [selected_client]

df = fetch_full(option) if option != "compare_routes" else None
if df is not None and not df.empty:
    if client_filter_applies and "client_id" in df.columns and client_ids:
        df = df[df["client_id"].isin(client_ids)]
    st.dataframe(df)

# -------------------------------
# MAPS AND VISUALIZATIONS
# -------------------------------
if option == "astar_routes":
    df["path_geom"] = df["path"].apply(parse_path)
    gdf = gpd.GeoDataFrame(df.dropna(subset=["path_geom"]), geometry="path_geom", crs="EPSG:4326")
    if not gdf.empty:
        m = folium.Map(location=[gdf.geometry.centroid.y.mean(), gdf.geometry.centroid.x.mean()], zoom_start=14)
        for _, row in gdf.iterrows():
            folium.PolyLine([(lat, lon) for lon, lat in row["path_geom"].coords],
                            color="blue", weight=3).add_to(m)
        st_folium(m, width=1000, height=600)

elif option == "mapf_routes":
    df["path_geom"] = df["path"].apply(parse_path)
    gdf = gpd.GeoDataFrame(df.dropna(subset=["path_geom"]), geometry="path_geom", crs="EPSG:4326")
    if not gdf.empty:
        m = folium.Map(location=[gdf.geometry.centroid.y.mean(), gdf.geometry.centroid.x.mean()], zoom_start=14)
        for _, row in gdf.iterrows():
            folium.PolyLine([(lat, lon) for lon, lat in row["path_geom"].coords],
                            color="orange", weight=3).add_to(m)
        st_folium(m, width=1000, height=600)

elif option == "user_patterns":
    gdf = build_point_gdf(df)
    if not gdf.empty:
        cmap = colormaps["Set1"]
        color_lookup = {pt: to_hex(cmap(i % cmap.N)) for i, pt in enumerate(gdf["pattern_type"].unique())}
        m = folium.Map(location=[gdf.geometry.y.mean(), gdf.geometry.x.mean()], zoom_start=13)
        for _, row in gdf.iterrows():
            folium.CircleMarker(
                location=(row.geometry.y, row.geometry.x),
                radius=5,
                color=color_lookup.get(row["pattern_type"], "#000000"),
                fill=True,
                fill_opacity=0.8,
                popup=f"{row['client_id']} | {row['pattern_type']}"
            ).add_to(m)
        st_folium(m, width=1000, height=600)

elif option == "hotspots":
    df["updated_at"] = pd.to_datetime(df["updated_at"])
    df_traj = fetch_full("view_latest_client_trajectories")
    df_traj["created_at"] = pd.to_datetime(df_traj["created_at"])
    available_dates = df_traj["created_at"].dt.date.unique()
    selected_date = st.date_input("Select animation date:", value=sorted(available_dates)[-1])
    speed = st.slider("Animation speed (sec per frame):", 0.05, 2.0, 0.3)

    start_time = pd.Timestamp(f"{selected_date} 00:00:00")
    end_time = start_time + pd.Timedelta(hours=24)
    df_window = df[(df["updated_at"] >= start_time) & (df["updated_at"] < end_time)].copy()
    df_window["time_bucket"] = df_window["updated_at"].dt.floor("5min")
    buckets = sorted(df_window["time_bucket"].dropna().unique())

    if not buckets:
        st.warning("No data available for selected 24h window.")
    else:
        for ts in buckets:
            st.write(f"Time bucket: {ts.strftime('%H:%M')}")
            gdf = build_point_gdf(df_window[df_window["time_bucket"] == ts])
            if not gdf.empty:
                m = folium.Map(location=[gdf.geometry.y.mean(), gdf.geometry.x.mean()], zoom_start=12)
                HeatMap(gdf[["lat", "lon"]].values.tolist(), radius=12).add_to(m)
                st_folium(m, width=1000, height=600)

elif option == "view_latest_client_trajectories":
    df["trajectory"] = df["trajectory"].apply(lambda x: x if isinstance(x, list) else [])
    m = folium.Map(location=[59.3, 18.0], zoom_start=11)
    for _, row in df.iterrows():
        points = [(float(p["lat"]), float(p["lon"])) for p in row["trajectory"]]
        if points:
            folium.PolyLine(points, color="green", weight=2.5, opacity=0.6).add_to(m)
    st_folium(m, width=1000, height=600)

elif option == "predicted_pois_sequence":
    gdf = build_point_gdf(df)
    if not gdf.empty:
        m = folium.Map(location=[gdf.geometry.y.mean(), gdf.geometry.x.mean()], zoom_start=13)
        for _, row in gdf.iterrows():
            folium.Marker(
                location=(row.geometry.y, row.geometry.x),
                popup=row.get("client_id", ""),
                icon=folium.Icon(color="purple")
            ).add_to(m)
        st_folium(m, width=1000, height=600)

elif option == "compare_routes":
    st.subheader("Compare routes and Trajectory for a Client")
    df_patterns = fetch_full("user_patterns").dropna(subset=["client_id", "pattern_type"])
    grouped = df_patterns.groupby("pattern_type")["client_id"].apply(list).to_dict()
    selected_pattern = st.selectbox("Pattern type:", sorted(grouped))
    selected_client = st.selectbox("Client ID:", ["Show All"] + sorted(grouped[selected_pattern]))
    client_ids = grouped[selected_pattern] if selected_client == "Show All" else [selected_client]
    color_by = st.radio("Color paths by:", ["path_type", "client_id"])

    df_astar = fetch_full("astar_routes")
    df_mapf = fetch_full("mapf_routes")
    df_traj = fetch_full("view_latest_client_trajectories")

    df_astar = df_astar[df_astar["client_id"].isin(client_ids)]
    df_mapf = df_mapf[df_mapf["client_id"].isin(client_ids)]
    df_traj = df_traj[df_traj["client_id"].isin(client_ids)]

    cmap = colormaps["tab20"]
    color_lookup = {cid: to_hex(cmap(i % cmap.N)) for i, cid in enumerate(client_ids)}
    m = folium.Map(location=[59.3, 18.0], zoom_start=12)

    for _, row in df_astar.iterrows():
        path = parse_path(row["path"])
        if path:
            color = color_lookup[row["client_id"]] if color_by == "client_id" else "blue"
            folium.PolyLine([(lat, lon) for lon, lat in path.coords], color=color, weight=2.5,
                            popup=f"A*: {row['client_id']}").add_to(m)

    for _, row in df_mapf.iterrows():
        path = parse_path(row["path"])
        if path:
            color = color_lookup[row["client_id"]] if color_by == "client_id" else "brown"
            folium.PolyLine([(lat, lon) for lon, lat in path.coords], color=color, weight=2.5,
                            popup=f"MAPF: {row['client_id']}").add_to(m)

    for _, row in df_traj.iterrows():
        if isinstance(row["trajectory"], list):
            points = [(float(p["lat"]), float(p["lon"])) for p in row["trajectory"]]
            if points:
                color = color_lookup[row["client_id"]] if color_by == "client_id" else "purple"
                folium.PolyLine(points, color=color, weight=2, popup=f"Trajectory: {row['client_id']}").add_to(m)

    st_folium(m, width=1000, height=600)
