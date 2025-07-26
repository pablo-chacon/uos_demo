import pandas as pd
import geopandas as gpd
from shapely.geometry import Point
from shapely import wkb
import binascii
import bz2
import shutil
import os
import streamlit as st

def decompress_pickle(path_bz2):
    path_pkl = path_bz2.replace(".bz2", "")
    if not os.path.exists(path_pkl):
        with bz2.open(path_bz2, 'rb') as f_in, open(path_pkl, 'wb') as f_out:
            shutil.copyfileobj(f_in, f_out)
    return path_pkl

def parse_path(path_wkb_hex):
    try:
        return wkb.loads(binascii.unhexlify(path_wkb_hex), hex=True)
    except Exception:
        return None

@st.cache_data
def parse_path_cached(path_wkb_hex):
    return parse_path(path_wkb_hex)

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

def preload_data(endpoints):
    preloaded = {}
    for key, path in endpoints.items():
        try:
            preloaded[key] = pd.read_pickle(path)
        except Exception as e:
            preloaded[key] = pd.DataFrame()
            st.warning(f"⚠️ Warning loading {key}: {e}")
    return preloaded

def preload_geodf(preloaded_df, keys_to_build):
    geodfs = {}
    for key in keys_to_build:
        try:
            geodfs[key] = build_point_gdf(preloaded_df.get(key, pd.DataFrame()))
        except Exception as e:
            geodfs[key] = gpd.GeoDataFrame()
            st.warning(f"⚠️ Warning building GeoDataFrame for {key}: {e}")
    return geodfs
