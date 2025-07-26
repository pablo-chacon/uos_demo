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
    "astar_routes": "data_exports/astar_routes.pkl",
    "mapf_routes": "data_exports/mapf_routes.pkl",
    "user_patterns": "data_exports/user_patterns.pkl",
    "hotspots": "data_exports/hotspots.pkl",
    "view_latest_client_trajectories": "data_exports/view_latest_client_trajectories.pkl",
    "predicted_pois_sequence": "data_exports/predicted_pois_sequence.pkl",
    "lines": "data_exports/lines.pkl",
    "stop_points": "data_exports/stop_points.pkl",
    "view_sites_with_stop_areas": "data_exports/view_sites_with_stop_areas.pkl",
    "view_top_daily_poi": "data_exports/view_top_daily_poi.pkl"
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


# 🧠 Preload all dataframes into memory at app start
PRELOADED_DATA = {}
for key, path in ENDPOINTS.items():
    try:
        PRELOADED_DATA[key] = pd.read_pickle(path)
    except Exception as e:
        PRELOADED_DATA[key] = pd.DataFrame()
        st.warning(f"Warning loading {key}: {e}")


def fetch_full(name):
    return PRELOADED_DATA.get(name, pd.DataFrame())

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


# New Tabs: Lines, Stop Points, Clusters vs Lines
with st.sidebar.expander("🆕 Infrastructure Layers"):
    infra_tab = st.sidebar.radio("🧱 Infrastructure Layers", ["None", "Stop Points", "Lines", "Patterns and Stops"])


if infra_tab == "Stop Points":
    st.subheader("🧭 Public Infrastructure: Stop Points")
    df = fetch_full("stop_points")
    if not df.empty:
        gdf = build_point_gdf(df)
        m = folium.Map(location=[gdf.geometry.y.mean(), gdf.geometry.x.mean()], zoom_start=12)
        for _, row in gdf.iterrows():
            folium.CircleMarker(
                location=(row.geometry.y, row.geometry.x),
                radius=3,
                color="red",
                fill=True,
                fill_opacity=0.8,
                popup=row.get("stop_name", "Stop")
            ).add_to(m)
        st_folium(m, width=1000, height=600)


elif infra_tab == "Lines":
    st.subheader("🛰️ Transit Network: Lines")

    df = fetch_full("lines")
    if df.empty:
        st.warning("No line data available.")
    else:
        # Flatten nested 'content' dictionary
        content_df = pd.json_normalize(df["content"])

        # Optional filter
        mode_options = sorted(content_df["transport_mode"].dropna().unique())
        selected_mode = st.selectbox("Filter by Transport Mode:", ["All"] + mode_options)

        if selected_mode != "All":
            content_df = content_df[content_df["transport_mode"] == selected_mode]

        st.dataframe(content_df[[
            "id", "name", "designation", "group_of_lines", "transport_mode", "contractor.name"
        ]].rename(columns={"contractor.name": "contractor"}))


elif infra_tab == "Patterns and Stops":
    st.subheader("🌍 Combined View: User Patterns + Stop Points")
    pat_df = fetch_full("user_patterns")
    stop_df = fetch_full("stop_points")

    if not pat_df.empty and not stop_df.empty:
        gdf_pat = build_point_gdf(pat_df)
        gdf_stop = build_point_gdf(stop_df)

        # Optional filter
        transport_types = gdf_stop["type"].dropna().unique()
        selected_type = st.selectbox("Filter by stop point type:", ["All"] + sorted(transport_types))
        if selected_type != "All":
            gdf_stop = gdf_stop[gdf_stop["type"] == selected_type]

        # Map center
        lat_center = pd.concat([gdf_pat.geometry.y, gdf_stop.geometry.y]).mean()
        lon_center = pd.concat([gdf_pat.geometry.x, gdf_stop.geometry.x]).mean()
        m = folium.Map(location=[lat_center, lon_center], zoom_start=12)

        # Cluster stop points (optional for large sets)
        from folium.plugins import MarkerCluster
        stop_cluster = MarkerCluster(name="Stop Points").add_to(m)
        for _, row in gdf_stop.iterrows():
            folium.CircleMarker(
                location=(row.geometry.y, row.geometry.x),
                radius=4,
                color="red",
                fill=True,
                fill_opacity=0.7,
                popup=f"Stop: {row.get('name', '')}"
            ).add_to(stop_cluster)

        # Add user patterns
        pattern_types = gdf_pat["pattern_type"].fillna("Unknown").unique()
        cmap = colormaps["Set1"]
        color_lookup = {pt: to_hex(cmap(i % cmap.N)) for i, pt in enumerate(pattern_types)}
        for _, row in gdf_pat.iterrows():
            folium.CircleMarker(
                location=(row.geometry.y, row.geometry.x),
                radius=6,
                color=color_lookup.get(row["pattern_type"], "#000000"),
                fill=True,
                fill_opacity=0.9,
                popup=f"{row.get('client_id', '')} | {row.get('pattern_type', '')}"
            ).add_to(m)

        # Optional: draw line to nearest stop for illustration
        from shapely.ops import nearest_points
        from shapely.geometry import Point

        for _, pat_row in gdf_pat.iterrows():
            nearest_geom = gdf_stop.geometry.iloc[gdf_stop.geometry.distance(pat_row.geometry).argmin()]
            line = folium.PolyLine(
                locations=[
                    (pat_row.geometry.y, pat_row.geometry.x),
                    (nearest_geom.y, nearest_geom.x)
                ],
                color="gray",
                weight=1.5,
                dash_array="5,5",
                opacity=0.6
            )
            m.add_child(line)

        st_folium(m, width=1000, height=600)
    else:
        st.warning("One or both datasets are empty.")


