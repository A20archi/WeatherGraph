#!/usr/bin/env python3
"""
app.py
======
India Weather Forecast Explorer

Features
--------
- Loads saved GraphCast-like forecast sequence from data/fc_seq.npy
- Select a city / district / custom location in India
- Clickable India map with auto-zoom and pulsing selected pin
- Nearby districts / cities forecast panel
- All 14 parameters at the selected point and lead time
- Keyless public API comparison for overlapping surface variables
- Normalized pattern error only

Run
---
    pip install streamlit plotly folium streamlit-folium requests pandas numpy
    streamlit run app.py
"""

from __future__ import annotations

import math
import os
import warnings
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import pandas as pd
import requests
import streamlit as st

warnings.filterwarnings("ignore")

try:
    import plotly.graph_objects as go
    from plotly.subplots import make_subplots
    PLOTLY_AVAILABLE = True
except Exception:
    PLOTLY_AVAILABLE = False

try:
    import folium
    from folium import FeatureGroup, Map, Marker, CircleMarker, PolyLine, Tooltip
    from folium.plugins import MarkerCluster, Fullscreen, MousePosition, MiniMap
    from streamlit_folium import st_folium
    FOLIUM_AVAILABLE = True
except Exception:
    FOLIUM_AVAILABLE = False


# ─────────────────────────────────────────────────────────────────────────────
# Page setup
# ─────────────────────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="India Weather Forecast Explorer",
    page_icon="🌦️",
    layout="wide",
)


# ─────────────────────────────────────────────────────────────────────────────
# Constants
# ─────────────────────────────────────────────────────────────────────────────
FEATURES = [
    "t2m", "msl", "u10", "v10", "tp", "sst",
    "z500", "t850", "q700", "u850", "v850",
    "omega500", "rh850", "z200",
]

INDIA_LAT_MIN = 5.0
INDIA_LAT_MAX = 38.0
INDIA_LON_MIN = 65.0
INDIA_LON_MAX = 100.0

RAW_UNITS = {
    "t2m": "K", "msl": "Pa", "u10": "m/s", "v10": "m/s", "tp": "mm/6h",
    "sst": "K", "z500": "m²/s²", "t850": "K", "q700": "kg/kg", "u850": "m/s",
    "v850": "m/s", "omega500": "Pa/s", "rh850": "%", "z200": "m²/s²",
}

DISPLAY_UNITS = {
    "t2m": "°C", "msl": "hPa", "u10": "m/s", "v10": "m/s", "tp": "mm/6h",
    "sst": "°C", "z500": "m²/s²", "t850": "°C", "q700": "kg/kg", "u850": "m/s",
    "v850": "m/s", "omega500": "Pa/s", "rh850": "%", "z200": "m²/s²",
}

FEATURE_EXPLANATION = {
    "t2m": "2 m air temperature. Best proxy for near-surface heat, warm spells, and heatwave potential.",
    "msl": "Mean sea-level pressure. Useful for low/high-pressure systems and cyclogenesis signals.",
    "u10": "10 m zonal wind (west-east component). Positive values indicate eastward flow.",
    "v10": "10 m meridional wind (south-north component). Positive values indicate northward flow.",
    "tp": "Total precipitation accumulated over 6 hours. Useful for rainfall, monsoon bursts, and flooding risk.",
    "sst": "Sea surface temperature. Important for ocean-atmosphere coupling and cyclone fuel.",
    "z500": "500 hPa geopotential. A large-scale circulation marker linked to troughs and ridges.",
    "t850": "850 hPa temperature. Often used to diagnose lower-tropospheric thermal structure.",
    "q700": "700 hPa specific humidity. Represents mid-level moisture available for convection.",
    "u850": "850 hPa zonal wind. Often used for monsoon flow and low-level jet structure.",
    "v850": "850 hPa meridional wind. Captures north-south moisture transport and convergence.",
    "omega500": "500 hPa vertical velocity. Negative values typically indicate rising motion and convection.",
    "rh850": "850 hPa relative humidity. High values support cloud formation and rainfall processes.",
    "z200": "200 hPa geopotential. Upper-level circulation feature linked to jets and outflow.",
}

CITY_PRESETS = {
    # North
    "New Delhi": (28.6139, 77.2090),
    "Chandigarh": (30.7333, 76.7794),
    "Lucknow": (26.8467, 80.9462),
    "Kanpur": (26.4499, 80.3319),
    "Prayagraj": (25.4358, 81.8463),
    "Varanasi": (25.3176, 82.9739),
    "Ayodhya": (26.7957, 82.1353),
    "Gorakhpur": (26.7606, 83.3732),
    "Jaipur": (26.9124, 75.7873),
    "Amritsar": (31.6340, 74.8723),
    # West
    "Mumbai": (19.0760, 72.8777),
    "Ahmedabad": (23.0225, 72.5714),
    "Surat": (21.1702, 72.8311),
    "Pune": (18.5204, 73.8567),
    # Central
    "Bhopal": (23.2599, 77.4126),
    "Indore": (22.7196, 75.8577),
    "Jabalpur": (23.1815, 79.9864),
    "Nagpur": (21.1458, 79.0882),
    # East
    "Kolkata": (22.5726, 88.3639),
    "Patna": (25.5941, 85.1376),
    "Bhubaneswar": (20.2961, 85.8245),
    "Ranchi": (23.3441, 85.3096),
    "Jamshedpur": (22.8046, 86.2029),
    # South
    "Chennai": (13.0827, 80.2707),
    "Bengaluru": (12.9716, 77.5946),
    "Hyderabad": (17.3850, 78.4867),
    "Kochi": (9.9312, 76.2673),
    "Thiruvananthapuram": (8.5241, 76.9366),
    # North-East
    "Guwahati": (26.1445, 91.7362),
    "Shillong": (25.5788, 91.8933),
    "Imphal": (24.8170, 93.9368),
    # Special
    "Leh": (34.1526, 77.5770),
    "Port Blair": (11.6234, 92.7265),
}

# Region zooms for a nicer feel
REGION_ZOOMS = [
    (("new delhi", "delhi", "gurugram", "noida", "faridabad", "ghaziabad", "sonipat", "meerut", "chandigarh", "amritsar"), (75.0, 81.5, 27.0, 32.5)),
    (("kolkata", "howrah", "barasat", "krishnanagar", "diamond harbour", "tamluk", "bardhaman", "durgapur", "west bengal"), (84.8, 90.8, 21.0, 26.8)),
    (("mumbai", "thane", "navi mumbai", "kalyan", "vasai", "alibag", "palghar", "pune", "surat"), (70.8, 76.0, 17.0, 22.0)),
    (("chennai", "kanchipuram", "chengalpattu", "tiruvallur", "vellore", "pondicherry", "bengaluru"), (76.0, 81.5, 10.0, 15.0)),
    (("bengaluru", "mysuru", "tumakuru", "mandya", "hosur", "ramanagara", "kochi", "thiruvananthapuram"), (73.8, 79.8, 8.5, 16.8)),
    (("hyderabad", "secunderabad", "rangareddy", "warangal", "nizamabad", "medak", "nagpur"), (76.2, 83.0, 15.0, 20.8)),
    (("ahmedabad", "gandhinagar", "vadodara", "surat", "rajkot", "bhopal", "indore", "ujjain"), (70.0, 77.0, 20.0, 25.8)),
    (("lucknow", "kanpur", "agra", "ayodhya", "varanasi", "prayagraj", "allahabad", "gorakhpur"), (77.0, 84.5, 24.0, 28.8)),
    (("jaipur", "ajmer", "alwar", "sikar", "tonk"), (73.0, 77.8, 24.5, 29.8)),
    (("patna", "gaya", "muzaffarpur", "bhagalpur", "darbhanga", "ranchi"), (83.6, 89.8, 23.0, 27.8)),
    (("bhubaneswar", "cuttack", "puri", "rourkela", "berhampur"), (83.8, 87.8, 18.5, 22.8)),
    (("guwahati", "shillong", "imphal", "agartala", "kohima"), (88.0, 96.5, 22.0, 28.8)),
    (("leh", "ladakh"), (76.0, 80.5, 32.0, 36.5)),
    (("port blair", "andaman"), (90.5, 94.5, 8.0, 14.0)),
    (("koch", "kochi", "ernakulam", "thrissur", "alappuzha", "kottayam", "trivandrum", "thiruvananthapuram"), (74.0, 78.8, 7.0, 13.0)),
    (("surat", "navsari", "bharuch", "vapi", "vadodara"), (70.8, 74.8, 20.0, 24.8)),
    (("nagpur", "amravati", "jabalpur", "chhindwara"), (77.0, 81.8, 20.0, 24.8)),
    (("indore", "bhopal", "ujjain", "gwalior"), (74.0, 79.5, 21.0, 26.8)),
    (("varanasi", "prayagraj", "kanpur", "ayodhya", "gorakhpur", "jaunpur", "mirzapur", "bhadohi", "ghazipur", "ballia"), (80.8, 84.8, 24.0, 28.8)),
]

NEARBY_BY_CLUSTER = {
    "new delhi": [("Gurugram", 28.46, 77.03), ("Noida", 28.57, 77.33), ("Faridabad", 28.41, 77.31), ("Ghaziabad", 28.67, 77.45), ("Sonipat", 28.99, 77.02), ("Meerut", 28.98, 77.70)],
    "kolkata": [("Howrah", 22.58, 88.31), ("Barasat", 22.72, 88.48), ("Krishnanagar", 23.40, 88.50), ("Diamond Harbour", 22.19, 88.19), ("Tamluk", 22.30, 87.92), ("Bardhaman", 23.24, 87.86)],
    "mumbai": [("Thane", 19.20, 72.97), ("Navi Mumbai", 19.03, 73.02), ("Kalyan", 19.24, 73.13), ("Vasai", 19.39, 72.83), ("Alibag", 18.64, 72.87), ("Palghar", 19.69, 72.76)],
    "chennai": [("Kanchipuram", 12.83, 79.70), ("Chengalpattu", 12.68, 79.98), ("Tiruvallur", 13.14, 79.91), ("Vellore", 12.92, 79.13), ("Puducherry", 11.94, 79.81)],
    "bengaluru": [("Mysuru", 12.30, 76.64), ("Tumakuru", 13.34, 77.10), ("Mandya", 12.52, 76.90), ("Hosur", 12.74, 77.83), ("Ramanagara", 12.72, 77.28)],
    "hyderabad": [("Secunderabad", 17.44, 78.50), ("Rangareddy", 17.35, 78.53), ("Warangal", 17.97, 79.60), ("Nizamabad", 18.68, 78.10), ("Medak", 18.03, 78.26)],
    "ahmedabad": [("Gandhinagar", 23.22, 72.64), ("Vadodara", 22.30, 73.19), ("Surat", 21.17, 72.83), ("Rajkot", 22.30, 70.80), ("Bharuch", 21.70, 72.98)],
    "lucknow": [("Kanpur", 26.45, 80.33), ("Agra", 27.18, 78.01), ("Ayodhya", 26.80, 82.20), ("Prayagraj", 25.44, 81.84), ("Varanasi", 25.32, 82.97)],
    "jaipur": [("Ajmer", 26.45, 74.64), ("Alwar", 27.56, 76.61), ("Sikar", 27.61, 75.15), ("Tonk", 26.17, 75.79)],
    "patna": [("Gaya", 24.79, 85.00), ("Muzaffarpur", 26.12, 85.39), ("Bhagalpur", 25.25, 87.00), ("Darbhanga", 26.17, 85.89), ("Purnia", 25.78, 87.47)],
    "bhubaneswar": [("Cuttack", 20.46, 85.88), ("Puri", 19.81, 85.83), ("Rourkela", 22.26, 84.85), ("Berhampur", 19.31, 84.80)],
    "guwahati": [("Shillong", 25.58, 91.89), ("Imphal", 24.82, 93.94), ("Agartala", 23.83, 91.28), ("Kohima", 25.67, 94.11)],
    "koch": [("Kochi", 9.93, 76.27), ("Ernakulam", 9.98, 76.28), ("Thrissur", 10.52, 76.21), ("Alappuzha", 9.50, 76.34), ("Kottayam", 9.59, 76.52), ("Thiruvananthapuram", 8.52, 76.94)],
    "kochi": [("Ernakulam", 9.98, 76.28), ("Thrissur", 10.52, 76.21), ("Alappuzha", 9.50, 76.34), ("Kottayam", 9.59, 76.52), ("Thiruvananthapuram", 8.52, 76.94)],
    "thiruvananthapuram": [("Kollam", 8.88, 76.59), ("Kanyakumari", 8.08, 77.54), ("Alappuzha", 9.50, 76.34), ("Kottayam", 9.59, 76.52), ("Kochi", 9.93, 76.27)],
    "leh": [("Leh", 34.15, 77.58), ("Kargil", 34.56, 76.13)],
    "port blair": [("Port Blair", 11.62, 92.73), ("Havelock", 11.97, 92.99), ("Diglipur", 13.27, 93.03)],
    "surat": [("Navsari", 20.95, 72.93), ("Bharuch", 21.70, 72.98), ("Vapi", 20.39, 72.91), ("Vadodara", 22.30, 73.19)],
    "nagpur": [("Amravati", 20.93, 77.75), ("Jabalpur", 23.18, 79.95), ("Chhindwara", 22.06, 78.93), ("Wardha", 20.74, 78.60)],
    "indore": [("Bhopal", 23.26, 77.41), ("Ujjain", 23.18, 75.77), ("Dewas", 22.96, 76.06), ("Gwalior", 26.22, 78.18)],
    "varanasi": [("Prayagraj", 25.44, 81.84), ("Kanpur", 26.45, 80.33), ("Ayodhya", 26.80, 82.20), ("Gorakhpur", 26.76, 83.37), ("Jaunpur", 25.75, 82.69), ("Mirzapur", 25.15, 82.58)],
}

FC_SEQ_PATH = os.path.join("data", "fc_seq.npy")


@st.cache_data(show_spinner=False)
def load_fc_seq(path: str = FC_SEQ_PATH) -> np.ndarray:
    if not os.path.exists(path):
        raise FileNotFoundError(
            f"{path} not found. Run setup_and_run.py first so it saves data/fc_seq.npy."
        )
    arr = np.load(path)
    if arr.ndim != 4 or arr.shape[1] != len(FEATURES):
        raise ValueError(f"Unexpected fc_seq shape: {arr.shape}")
    return arr.astype(np.float32)


# ─────────────────────────────────────────────────────────────────────────────
# Utility functions
# ─────────────────────────────────────────────────────────────────────────────
def to_display_value(feature: str, raw_value: float) -> float:
    if feature in {"t2m", "sst", "t850"}:
        return float(raw_value - 273.15)
    if feature == "msl":
        return float(raw_value / 100.0)
    return float(raw_value)


def display_unit(feature: str) -> str:
    return DISPLAY_UNITS.get(feature, RAW_UNITS.get(feature, ""))


def fmt_value(feature: str, raw_value: Optional[float]) -> str:
    if raw_value is None or (isinstance(raw_value, float) and np.isnan(raw_value)):
        return "N/A"
    return f"{to_display_value(feature, raw_value):.2f} {display_unit(feature)}"


def region_context(lat: float, lon: float) -> str:
    if lat >= 30:
        return "Northern India / Himalaya-adjacent influence."
    if 15 <= lat < 30 and 80 <= lon <= 92:
        return "East/central India corridor, sensitive to Bay of Bengal moisture."
    if 8 <= lat < 20 and 72 <= lon <= 80:
        return "West coast / Arabian Sea moisture corridor."
    if 20 <= lat < 30 and 70 <= lon < 80:
        return "North-western / continental heat and dry-air influence."
    return "Interior India / mixed land-surface and monsoon influence."


def feature_focus_message(feature: str, pred_val: float) -> str:
    if feature == "tp":
        if pred_val > 10:
            return "Heavy rainfall signal. Watch for localized flooding and convective bursts."
        if pred_val > 3:
            return "Moderate precipitation signal. Monsoon activity looks active."
        return "Low rainfall signal. The column looks comparatively dry."
    if feature == "t2m":
        if pred_val > 35:
            return "Very warm near-surface conditions. Heat stress may be elevated."
        if pred_val > 30:
            return "Warm surface conditions. Noticeable heating but not extreme."
        return "Mild near-surface temperature signal."
    if feature == "rh850":
        if pred_val > 80:
            return "Strong mid-level moisture. Convective development becomes more likely."
        if pred_val > 60:
            return "Moderately moist mid-level atmosphere."
        return "Relatively dry mid-level atmosphere."
    if feature == "omega500":
        if pred_val < -0.05:
            return "Rising motion is present aloft, which can support convection."
        return "Weak vertical motion signal."
    return FEATURE_EXPLANATION[feature]


def zscore(x: np.ndarray) -> np.ndarray:
    return (x - np.mean(x)) / (np.std(x) + 1e-6)


def compute_errors(pred: np.ndarray, real: np.ndarray) -> Tuple[Optional[float], Optional[float]]:
    if real is None or pred is None or len(pred) == 0 or len(real) == 0:
        return None, None
    pred_n = zscore(np.asarray(pred, dtype=np.float32))
    real_n = zscore(np.asarray(real, dtype=np.float32))
    pattern_err = float(np.mean(np.abs(pred_n - real_n)))
    corr = None
    if len(pred) > 1 and np.std(pred) > 1e-8 and np.std(real) > 1e-8:
        corr = float(np.corrcoef(pred, real)[0, 1])
    return pattern_err, corr


@st.cache_data(ttl=1800, show_spinner=False)
def geocode_place(query: str) -> Optional[Tuple[float, float, str]]:
    query = (query or "").strip()
    if not query:
        return None
    url = "https://geocoding-api.open-meteo.com/v1/search"
    params = {"name": query, "count": 10, "language": "en", "format": "json"}
    r = requests.get(url, params=params, timeout=20)
    r.raise_for_status()
    data = r.json()
    results = data.get("results", [])
    if not results:
        return None
    for item in results:
        country = str(item.get("country", "")).lower()
        admin1 = str(item.get("admin1", "")).lower()
        name = str(item.get("name", query))
        if "india" in country or "india" in admin1 or name.lower() in query.lower():
            return float(item["latitude"]), float(item["longitude"]), name
    item = results[0]
    return float(item["latitude"]), float(item["longitude"]), item.get("name", query)


@st.cache_data(ttl=1800, show_spinner=False)
def fetch_open_meteo_forecast(lat: float, lon: float) -> Dict[str, Any]:
    url = "https://api.open-meteo.com/v1/forecast"
    params = {
        "latitude": lat,
        "longitude": lon,
        "hourly": ",".join([
            "temperature_2m",
            "relative_humidity_2m",
            "pressure_msl",
            "wind_speed_10m",
            "wind_direction_10m",
            "precipitation",
        ]),
        "forecast_days": 5,
        "timezone": "auto",
        "temperature_unit": "celsius",
        "wind_speed_unit": "kmh",
        "precipitation_unit": "mm",
    }
    r = requests.get(url, params=params, timeout=25)
    r.raise_for_status()
    data = r.json()
    if "hourly" not in data:
        raise RuntimeError("Open-Meteo response missing hourly data.")
    return data


def parse_open_meteo_hourly(api: Dict[str, Any]) -> Dict[str, np.ndarray]:
    h = api["hourly"]
    return {
        "time": np.array(pd.to_datetime(h.get("time", []))),
        "temperature_2m": np.array(h.get("temperature_2m", []), dtype=np.float32),
        "relative_humidity_2m": np.array(h.get("relative_humidity_2m", []), dtype=np.float32),
        "pressure_msl": np.array(h.get("pressure_msl", []), dtype=np.float32),
        "wind_speed_10m": np.array(h.get("wind_speed_10m", []), dtype=np.float32),
        "wind_direction_10m": np.array(h.get("wind_direction_10m", []), dtype=np.float32),
        "precipitation": np.array(h.get("precipitation", []), dtype=np.float32),
    }


def haversine_km(lat1, lon1, lat2, lon2):
    r = 6371.0
    p1 = np.radians(lat1)
    p2 = np.radians(lat2)
    dp = np.radians(lat2 - lat1)
    dl = np.radians(lon2 - lon1)
    a = np.sin(dp / 2) ** 2 + np.cos(p1) * np.cos(p2) * np.sin(dl / 2) ** 2
    return 2 * r * np.arcsin(np.sqrt(a))


def _match_region_bounds(place_name: str, lat: float, lon: float):
    name = (place_name or "").lower()
    for keywords, bounds in REGION_ZOOMS:
        if any(k in name for k in keywords):
            return bounds
    lon_min = max(65.0, lon - 4.0)
    lon_max = min(100.0, lon + 4.0)
    lat_min = max(5.0, lat - 3.0)
    lat_max = min(38.0, lat + 3.0)
    return lon_min, lon_max, lat_min, lat_max


def nearby_places(city_name: str, lat: float, lon: float, n: int = 6):
    key = (city_name or "").lower()
    for cluster, places in NEARBY_BY_CLUSTER.items():
        if cluster in key:
            return places[:n]
    candidates = []
    for city, (clat, clon) in CITY_PRESETS.items():
        if city.lower() == key:
            continue
        candidates.append((city, clat, clon, haversine_km(lat, lon, clat, clon)))
    candidates.sort(key=lambda x: x[3])
    return [(name, clat, clon) for name, clat, clon, _ in candidates[:n]]


def make_india_map(lat: float, lon: float, label: str, place_name: Optional[str] = None):
    if FOLIUM_AVAILABLE:
        lon_min, lon_max, lat_min, lat_max = _match_region_bounds(place_name or label, lat, lon)
        center_lat = (lat_min + lat_max) / 2.0
        center_lon = (lon_min + lon_max) / 2.0
        zoom = 5 if (lon_max - lon_min > 5.0 or lat_max - lat_min > 4.0) else 7
        m = folium.Map(location=[center_lat, center_lon], zoom_start=zoom, tiles="CartoDB positron", control_scale=True, prefer_canvas=True)
        Fullscreen(position="topright").add_to(m)
        MiniMap(toggle_display=True).add_to(m)
        MousePosition(position="bottomleft", separator=" | ", prefix="Lat/Lon:").add_to(m)

        # India boundary rectangle (light, just to show the domain)
        folium.Rectangle(
            bounds=[[INDIA_LAT_MIN, INDIA_LON_MIN], [INDIA_LAT_MAX, INDIA_LON_MAX]],
            color="#4c78a8",
            weight=2,
            fill=False,
            opacity=0.65,
        ).add_to(m)

        # Pin glow layers
        glow_cluster = FeatureGroup(name="Pin")
        glow_cluster.add_to(m)

        for radius, opacity in [(28, 0.08), (20, 0.12), (12, 0.20)]:
            CircleMarker(
                location=[lat, lon],
                radius=radius,
                color="#d62728",
                weight=1,
                fill=True,
                fill_color="#d62728",
                fill_opacity=opacity,
            ).add_to(glow_cluster)

        Marker(
            location=[lat, lon],
            tooltip=Tooltip(f"{label}  ({lat:.2f}, {lon:.2f})", sticky=True),
            icon=folium.Icon(color="red", icon="info-sign"),
        ).add_to(m)

        return m

    # Plotly fallback
    if not PLOTLY_AVAILABLE:
        return None

    lon_min, lon_max, lat_min, lat_max = _match_region_bounds(place_name or label, lat, lon)
    core_sizes = [10, 14, 18, 22, 18, 14]
    ring_sizes = [18, 24, 30, 36, 30, 24]

    fig = go.Figure()
    fig.add_trace(go.Scattergeo(
        lon=[lon], lat=[lat], mode="markers",
        marker=dict(size=ring_sizes[0], color="red", opacity=0.22, line=dict(width=0)),
        hoverinfo="skip", showlegend=False,
    ))
    fig.add_trace(go.Scattergeo(
        lon=[lon], lat=[lat], mode="markers+text",
        text=[label], textposition="top center",
        marker=dict(size=core_sizes[0], color="red", line=dict(width=2, color="white")),
        hovertemplate=f"{label}<extra></extra>", showlegend=False,
    ))
    frames = []
    for ring_s, core_s in zip(ring_sizes, core_sizes):
        frames.append(go.Frame(data=[
            go.Scattergeo(
                lon=[lon], lat=[lat], mode="markers",
                marker=dict(size=ring_s, color="red", opacity=0.18, line=dict(width=0)),
                hoverinfo="skip", showlegend=False,
            ),
            go.Scattergeo(
                lon=[lon], lat=[lat], mode="markers+text",
                text=[label], textposition="top center",
                marker=dict(size=core_s, color="red", line=dict(width=2, color="white")),
                hovertemplate=f"{label}<extra></extra>", showlegend=False,
            ),
        ]))
    fig.frames = frames
    fig.update_geos(
        scope="asia",
        projection_type="mercator",
        showland=True,
        landcolor="rgb(245,245,245)",
        showocean=True,
        oceancolor="rgb(230,245,255)",
        showcountries=True,
        countrycolor="gray",
        showcoastlines=True,
        coastlinecolor="gray",
        lonaxis_range=[lon_min, lon_max],
        lataxis_range=[lat_min, lat_max],
    )
    fig.update_layout(
        height=520,
        margin=dict(l=0, r=0, t=10, b=0),
        showlegend=False,
        paper_bgcolor="white",
        updatemenus=[dict(
            type="buttons",
            showactive=False,
            x=0.02,
            y=0.02,
            bgcolor="white",
            bordercolor="black",
            borderwidth=1.2,
            font=dict(size=13, color="black"),
            buttons=[
                dict(label="▶ Play pulse", method="animate", args=[None, {"frame": {"duration": 180, "redraw": True}, "fromcurrent": True, "transition": {"duration": 100}}]),
                dict(label="❚❚ Pause", method="animate", args=[[None], {"frame": {"duration": 0, "redraw": False}, "mode": "immediate", "transition": {"duration": 0}}]),
            ],
        )],
    )
    return fig


def lead_hours(fc_seq: np.ndarray) -> np.ndarray:
    return np.arange(6, 6 * len(fc_seq) + 1, 6, dtype=np.int32)


def selected_point_series(fc_seq: np.ndarray, lat_idx: int, lon_idx: int, feature_idx: int) -> np.ndarray:
    return fc_seq[:, feature_idx, lat_idx, lon_idx]


def comparable_series_model(fc_seq: np.ndarray, lat_idx: int, lon_idx: int, variable: str) -> np.ndarray:
    if variable == "temperature_2m":
        return np.array([to_display_value("t2m", float(s[FEATURES.index("t2m"), lat_idx, lon_idx])) for s in fc_seq], dtype=np.float32)
    if variable == "pressure_msl":
        return np.array([to_display_value("msl", float(s[FEATURES.index("msl"), lat_idx, lon_idx])) for s in fc_seq], dtype=np.float32)
    if variable == "wind_speed_10m":
        out = []
        for s in fc_seq:
            u10 = float(s[FEATURES.index("u10"), lat_idx, lon_idx])
            v10 = float(s[FEATURES.index("v10"), lat_idx, lon_idx])
            out.append(math.sqrt(u10 * u10 + v10 * v10) * 3.6)
        return np.array(out, dtype=np.float32)
    if variable == "precipitation":
        return np.array([float(s[FEATURES.index("tp"), lat_idx, lon_idx]) for s in fc_seq], dtype=np.float32)
    raise ValueError(variable)


def comparable_series_api(api_hourly: Dict[str, np.ndarray], variable: str, n_steps: int) -> np.ndarray:
    if variable == "temperature_2m":
        return api_hourly["temperature_2m"][:n_steps].astype(np.float32)
    if variable == "pressure_msl":
        return api_hourly["pressure_msl"][:n_steps].astype(np.float32)
    if variable == "wind_speed_10m":
        return api_hourly["wind_speed_10m"][:n_steps].astype(np.float32)
    if variable == "precipitation":
        arr = api_hourly["precipitation"]
        vals = []
        for idx in range(n_steps):
            start = max(idx - 5, 0)
            end = min(idx + 1, len(arr))
            vals.append(float(np.nansum(arr[start:end])) if end > start else np.nan)
        return np.array(vals, dtype=np.float32)
    raise ValueError(variable)


def compute_comparison_table(fc_seq: np.ndarray, lat_idx: int, lon_idx: int, api_hourly: Dict[str, np.ndarray]) -> Tuple[pd.DataFrame, Optional[float]]:
    variables = [
        ("temperature_2m", "2 m temperature", "°C"),
        ("pressure_msl", "Mean sea-level pressure", "hPa"),
        ("wind_speed_10m", "10 m wind speed", "km/h"),
        ("precipitation", "6 h precipitation", "mm/6h"),
    ]
    rows = []
    pattern_errors = []
    for var, label, unit in variables:
        pred = comparable_series_model(fc_seq, lat_idx, lon_idx, var)
        real = comparable_series_api(api_hourly, var, len(pred))
        pat_err, corr = compute_errors(pred, real)
        if pat_err is not None:
            pattern_errors.append(pat_err)
        rows.append({
            "Comparable field": label,
            "Normalized pattern error": "N/A" if pat_err is None else f"{pat_err:.3f}",
            "Correlation": "N/A" if corr is None else f"{corr:.3f}",
            "Model (current lead)": f"{pred[-1]:.2f} {unit}",
            "API (same lead)": f"{real[-1]:.2f} {unit}",
        })
    net_pattern = float(np.mean(pattern_errors)) if pattern_errors else None
    return pd.DataFrame(rows), net_pattern


def build_all_features_table(fc_seq: np.ndarray, lead_idx: int, lat_idx: int, lon_idx: int) -> pd.DataFrame:
    state = fc_seq[lead_idx]
    rows = []
    for i, feat in enumerate(FEATURES):
        raw_val = float(state[i, lat_idx, lon_idx])
        rows.append({
            "Parameter": feat.upper(),
            "Meaning": FEATURE_EXPLANATION[feat],
            "Model value": fmt_value(feat, raw_val),
            "Raw unit": RAW_UNITS[feat],
            "API comparable?": "Yes" if feat in {"t2m", "msl", "tp", "u10", "v10"} else "No",
        })
    return pd.DataFrame(rows)


def dynamic_insight(feature: str, pred_val: float, pat_err: Optional[float], lat: float, lon: float) -> str:
    ctx = region_context(lat, lon)
    feat_msg = feature_focus_message(feature, pred_val)
    if pat_err is None:
        pat_msg = " Pattern comparison is unavailable for this point."
    elif pat_err < 0.5:
        pat_msg = f" Pattern similarity is strong; normalized pattern error is {pat_err:.3f}."
    elif pat_err < 1.0:
        pat_msg = f" Pattern similarity is acceptable; normalized pattern error is {pat_err:.3f}."
    else:
        pat_msg = f" Pattern mismatch is visible; normalized pattern error is {pat_err:.3f}."
    return f"{ctx} {feat_msg}{pat_msg}"


def city_library_table() -> pd.DataFrame:
    rows = []
    for city, (lat, lon) in CITY_PRESETS.items():
        rows.append({"City": city, "Latitude": f"{lat:.4f}", "Longitude": f"{lon:.4f}"})
    return pd.DataFrame(rows)


# ─────────────────────────────────────────────────────────────────────────────
# UI
# ─────────────────────────────────────────────────────────────────────────────
st.title("🌦️ India Weather Forecast Explorer")
st.caption(
    "Select a pin on the India map, inspect the 14 forecast parameters, and compare overlapping surface variables against a keyless public weather API."
)

if not os.path.exists(FC_SEQ_PATH):
    st.error("data/fc_seq.npy not found. Run `python setup_and_run.py --mode real --quick` first.")
    st.stop()

try:
    fc_seq = load_fc_seq()
except Exception as e:
    st.error(f"Could not load forecast sequence: {e}")
    st.stop()

n_steps = fc_seq.shape[0]
lead_hours_arr = lead_hours(fc_seq)

preset_names = list(CITY_PRESETS.keys())
default_index = preset_names.index("Varanasi") if "Varanasi" in preset_names else 0

st.sidebar.header("Controls")
preset_city = st.sidebar.selectbox("Preset city", preset_names, index=default_index)
city_input = st.sidebar.text_input("City / place in India", preset_city)
use_geocode = st.sidebar.checkbox("Use keyless geocoding", value=True)

# Better one-click city selection
quick_pick = st.sidebar.selectbox(
    "Quick city jump",
    ["—"] + preset_names,
    index=0,
)
if quick_pick != "—":
    city_input = quick_pick

lat_guess, lon_guess, resolved_name = CITY_PRESETS[preset_city][0], CITY_PRESETS[preset_city][1], preset_city
geo = None
if use_geocode:
    try:
        geo = geocode_place(city_input)
    except Exception:
        geo = None

if geo is not None:
    lat_guess, lon_guess, resolved_name = geo
else:
    if city_input in CITY_PRESETS:
        lat_guess, lon_guess = CITY_PRESETS[city_input]
        resolved_name = city_input
    else:
        resolved_name = city_input

lat = st.sidebar.slider("Latitude", INDIA_LAT_MIN, INDIA_LAT_MAX, float(lat_guess), 0.01)
lon = st.sidebar.slider("Longitude", INDIA_LON_MIN, INDIA_LON_MAX, float(lon_guess), 0.01)
feature_focus = st.sidebar.selectbox("Focus parameter", FEATURES, index=0)
lead_hour = st.sidebar.slider(
    "Forecast lead (hours)",
    int(lead_hours_arr[0]),
    int(lead_hours_arr[-1]),
    int(lead_hours_arr[min(3, len(lead_hours_arr) - 1)]),
    step=6,
)

lead_idx = max(0, min(lead_hour // 6 - 1, n_steps - 1))
lat_idx = int(np.argmin(np.abs(np.linspace(5.0, 38.0, fc_seq.shape[2]) - lat)))
lon_idx = int(np.argmin(np.abs(np.linspace(65.0, 100.0, fc_seq.shape[3]) - lon)))

api_data = None
api_hourly = None
api_error_msg = None
with st.spinner("Loading public weather API for the selected pin..."):
    try:
        api_data = fetch_open_meteo_forecast(lat, lon)
        api_hourly = parse_open_meteo_hourly(api_data)
    except Exception as e:
        api_error_msg = str(e)

selected_state = fc_seq[lead_idx]
feature_idx = FEATURES.index(feature_focus)
selected_feature_value_raw = float(selected_state[feature_idx, lat_idx, lon_idx])
selected_feature_value_disp = to_display_value(feature_focus, selected_feature_value_raw)

comp_df = None
net_pat_err = None
if api_hourly is not None:
    comp_df, net_pat_err = compute_comparison_table(fc_seq, lat_idx, lon_idx, api_hourly)

col_a, col_b, col_c, col_d = st.columns(4)
col_a.metric("Selected pin", resolved_name, f"{lat:.2f}, {lon:.2f}")
col_b.metric("Selected lead", f"t+{lead_hour}h", region_context(lat, lon))
col_c.metric(
    f"{feature_focus.upper()} @ pin",
    f"{selected_feature_value_disp:.2f} {display_unit(feature_focus)}",
    FEATURE_EXPLANATION[feature_focus].split(".")[0],
)
col_d.metric(
    "API status",
    "Loaded" if api_hourly is not None else "Unavailable",
    "Geoencoding + forecast",
)

left_col, right_col = st.columns([1.1, 1.0])

with left_col:
    st.subheader("Forecast at the selected pin")
    series_raw = selected_point_series(fc_seq, lat_idx, lon_idx, feature_idx)
    series_disp = np.array([to_display_value(feature_focus, float(v)) for v in series_raw], dtype=np.float32)

    if PLOTLY_AVAILABLE:
        fig = go.Figure()
        fig.add_trace(go.Scatter(
            x=lead_hours_arr,
            y=series_disp,
            mode="lines+markers",
            name="Model",
            line=dict(width=3),
        ))
        fig.add_vline(x=lead_hour, line_width=2, line_dash="dash", line_color="crimson")
        fig.update_layout(
            height=380,
            xaxis_title="Lead time (hours)",
            yaxis_title=f"{feature_focus.upper()} ({display_unit(feature_focus)})",
            margin=dict(l=20, r=20, t=20, b=20),
        )
        st.plotly_chart(fig, use_container_width=True)
    else:
        chart_df = pd.DataFrame({"lead_h": lead_hours_arr, feature_focus.upper(): series_disp}).set_index("lead_h")
        st.line_chart(chart_df, height=320)

    st.info(dynamic_insight(feature_focus, selected_feature_value_disp, None, lat, lon))

with right_col:
    st.subheader("Interactive India map")
    map_obj = make_india_map(lat, lon, resolved_name, place_name=resolved_name)
    clicked_coords = None
    if FOLIUM_AVAILABLE and isinstance(map_obj, folium.Map):
        map_state = st_folium(
            map_obj,
            width=None,
            height=520,
            returned_objects=["last_clicked", "bounds", "center", "zoom"],
        )
        clicked_coords = map_state.get("last_clicked") if isinstance(map_state, dict) else None
        if clicked_coords:
            st.caption(f"Map click captured: {clicked_coords['lat']:.4f}, {clicked_coords['lng']:.4f}")
    elif PLOTLY_AVAILABLE and map_obj is not None:
        st.plotly_chart(map_obj, use_container_width=True)
    else:
        st.warning("Install folium and streamlit-folium for the richer clickable India map.")

    if clicked_coords:
        st.info("Use the clicked coordinates as a new point by copying them into the sliders.")

with st.expander("Nearby districts / cities forecast", expanded=True):
    nearby = nearby_places(resolved_name, lat, lon, n=6)
    rows = []
    state_for_lead = fc_seq[lead_idx]
    for place, plat, plon in nearby:
        plat_idx = int(np.argmin(np.abs(np.linspace(5.0, 38.0, fc_seq.shape[2]) - plat)))
        plon_idx = int(np.argmin(np.abs(np.linspace(65.0, 100.0, fc_seq.shape[3]) - plon)))
        t2m = to_display_value("t2m", float(state_for_lead[FEATURES.index("t2m"), plat_idx, plon_idx]))
        msl = to_display_value("msl", float(state_for_lead[FEATURES.index("msl"), plat_idx, plon_idx]))
        tp = float(state_for_lead[FEATURES.index("tp"), plat_idx, plon_idx])
        rh = float(state_for_lead[FEATURES.index("rh850"), plat_idx, plon_idx])
        rows.append({
            "Place": place,
            "Lat": f"{plat:.2f}",
            "Lon": f"{plon:.2f}",
            "T2M (°C)": f"{t2m:.2f}",
            "MSL (hPa)": f"{msl:.2f}",
            "TP (mm/6h)": f"{tp:.2f}",
            "RH850 (%)": f"{rh:.2f}",
        })
    st.dataframe(pd.DataFrame(rows), use_container_width=True, hide_index=True)


tab1, tab2, tab3, tab4 = st.tabs(["All 14 parameters", "API comparison", "City library", "Glossary"])

with tab1:
    st.subheader("All 14 model parameters at the selected pin and lead time")
    summary_df = build_all_features_table(fc_seq, lead_idx, lat_idx, lon_idx)
    st.dataframe(summary_df, use_container_width=True, hide_index=True)
    st.caption(
        "The table shows the model outputs for all 14 ERA5-derived variables. Only the overlapping surface variables can be checked against the public API; the rest are model-only fields."
    )

with tab2:
    st.subheader("Model vs public weather API comparison")
    if api_hourly is None:
        st.error(f"API request failed: {api_error_msg}")
    else:
        st.metric(
            "Net normalized pattern error",
            f"{net_pat_err:.3f}" if net_pat_err is not None else "N/A",
            "Lower is better",
        )
        if comp_df is not None:
            st.dataframe(comp_df, use_container_width=True, hide_index=True)

        if PLOTLY_AVAILABLE:
            model_t2m = comparable_series_model(fc_seq, lat_idx, lon_idx, "temperature_2m")
            model_msl = comparable_series_model(fc_seq, lat_idx, lon_idx, "pressure_msl")
            model_wind = comparable_series_model(fc_seq, lat_idx, lon_idx, "wind_speed_10m")
            model_tp = comparable_series_model(fc_seq, lat_idx, lon_idx, "precipitation")

            api_t2m = comparable_series_api(api_hourly, "temperature_2m", len(model_t2m))
            api_msl = comparable_series_api(api_hourly, "pressure_msl", len(model_msl))
            api_wind = comparable_series_api(api_hourly, "wind_speed_10m", len(model_wind))
            api_tp = comparable_series_api(api_hourly, "precipitation", len(model_tp))

            fig = make_subplots(
                rows=2,
                cols=2,
                subplot_titles=(
                    "2 m Temperature (°C)",
                    "Mean Sea-Level Pressure (hPa)",
                    "6 h Precipitation (mm/6h)",
                    "10 m Wind Speed (km/h)",
                ),
                vertical_spacing=0.12,
                horizontal_spacing=0.10,
            )

            fig.add_trace(go.Scatter(x=lead_hours_arr, y=model_t2m, mode="lines+markers", name="Model T2M"), row=1, col=1)
            fig.add_trace(go.Scatter(x=lead_hours_arr, y=api_t2m, mode="lines+markers", name="API T2M"), row=1, col=1)
            fig.add_trace(go.Scatter(x=lead_hours_arr, y=model_msl, mode="lines+markers", name="Model MSL"), row=1, col=2)
            fig.add_trace(go.Scatter(x=lead_hours_arr, y=api_msl, mode="lines+markers", name="API MSL"), row=1, col=2)
            fig.add_trace(go.Scatter(x=lead_hours_arr, y=model_tp, mode="lines+markers", name="Model TP"), row=2, col=1)
            fig.add_trace(go.Scatter(x=lead_hours_arr, y=api_tp, mode="lines+markers", name="API TP"), row=2, col=1)
            fig.add_trace(go.Scatter(x=lead_hours_arr, y=model_wind, mode="lines+markers", name="Model Wind"), row=2, col=2)
            fig.add_trace(go.Scatter(x=lead_hours_arr, y=api_wind, mode="lines+markers", name="API Wind"), row=2, col=2)

            fig.update_layout(height=720, showlegend=True, margin=dict(l=20, r=20, t=40, b=20))
            fig.update_xaxes(title_text="Lead hours")
            st.plotly_chart(fig, use_container_width=True)

        if net_pat_err is not None:
            if net_pat_err < 0.5:
                st.success(f"Pattern similarity is strong for this pin: {net_pat_err:.3f}")
            elif net_pat_err < 1.0:
                st.warning(f"Pattern similarity is acceptable for this pin: {net_pat_err:.3f}")
            else:
                st.error(f"Pattern mismatch is visible for this pin: {net_pat_err:.3f}")

        st.info(dynamic_insight(feature_focus, selected_feature_value_disp, net_pat_err, lat, lon))

with tab3:
    st.subheader("Indian city library")
    st.dataframe(city_library_table(), use_container_width=True, hide_index=True)
    st.caption("Use the dropdown in the sidebar to jump quickly to any of these locations.")

with tab4:
    st.subheader("What each parameter means")
    glossary_df = pd.DataFrame([
        {"Parameter": k.upper(), "Units": RAW_UNITS[k], "Meaning": v}
        for k, v in FEATURE_EXPLANATION.items()
    ])
    st.dataframe(glossary_df, use_container_width=True, hide_index=True)
    st.markdown(
        """
        **How to read the app**
        - The pin chooses the location in India.
        - The line chart shows how the selected parameter changes across the 6-hour forecast steps.
        - The full table lists all 14 ERA5-derived parameters at the chosen pin.
        - The API tab compares only the overlapping surface variables, because the public forecast API does not expose all pressure-level fields used in this project.
        - The quality metric is shown as normalized pattern error, so the focus stays on structure rather than raw units.
        """
    )

st.caption("If the API request fails, the model forecast still loads from `data/fc_seq.npy`.")
