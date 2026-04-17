#!/usr/bin/env python3
"""
setup_and_run.py
================
ONE FILE THAT DOES EVERYTHING.

Loads ERA5 (real or simulated), builds tensors, runs the Day 2 + Day 3 pipeline,
and saves all figures + PDFs.

Default mode is REAL ERA5, quick 2-week slice.

Usage
-----
    # Real ERA5, 2-week slice (default recommended)
    python setup_and_run.py --mode real --quick

    # Simulation mode (no internet, instant)
    python setup_and_run.py --mode simulate

    # Real ERA5 full JJAS slice
    python setup_and_run.py --mode real

    # Change output directory
    python setup_and_run.py --mode real --quick --out ./my_figs

    # Use a real cyclone track CSV if available
    python setup_and_run.py --mode real --quick --cyclone-track ibtracs_track.csv

Author: Saptarshi Banerjee, IIT(BHU) Varanasi
"""

import os
import sys
import time
import gc
import argparse
import warnings

warnings.filterwarnings("ignore")

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from scipy.ndimage import gaussian_filter, zoom

# Optional dependencies
try:
    import xarray as xr
    import gcsfs
    REAL_AVAILABLE = True
except Exception:
    REAL_AVAILABLE = False

try:
    import cartopy.crs as ccrs
    import cartopy.feature as cfeature
    CARTOPY_AVAILABLE = True
except Exception:
    CARTOPY_AVAILABLE = False


# ── Colour printing ───────────────────────────────────────────
def log(msg, color="cyan"):
    codes = {
        "cyan": "\033[96m",
        "green": "\033[92m",
        "yellow": "\033[93m",
        "red": "\033[91m",
        "bold": "\033[1m",
        "end": "\033[0m",
    }
    sys.stdout.write(f"{codes.get(color, '')}{msg}{codes['end']}\n")
    sys.stdout.flush()


def step(n, msg):
    log(f"\n[{n}] {msg}", "bold")


def ok(msg):
    log(f"    ✓ {msg}", "green")


def warn(msg):
    log(f"    ⚠ {msg}", "yellow")


def err(msg):
    log(f"    ✗ {msg}", "red")


# ── Dependency check ─────────────────────────────────────────
def check_deps():
    step(0, "Checking dependencies...")
    required = ["numpy", "scipy", "matplotlib"]
    optional = ["xarray", "zarr", "gcsfs", "netCDF4", "cartopy"]
    missing_req = []

    for pkg in required:
        try:
            __import__(pkg)
            ok(pkg)
        except ImportError:
            err(pkg + " — MISSING")
            missing_req.append(pkg)

    for pkg in optional:
        try:
            __import__(pkg)
            ok(f"{pkg} (optional)")
        except ImportError:
            warn(f"{pkg} (optional — needed for real ERA5 mode)")

    if missing_req:
        log("\nInstall missing packages:", "red")
        log(f"  pip install {' '.join(missing_req)}", "yellow")
        sys.exit(1)

    if not CARTOPY_AVAILABLE:
        warn("cartopy not available — contour maps will fall back to plain matplotlib.")


# ═══════════════════════════════════════════════════════════════
# CONSTANTS
# ═══════════════════════════════════════════════════════════════
FEATURES = [
    "t2m", "msl", "u10", "v10", "tp", "sst",
    "z500", "t850", "q700", "u850", "v850",
    "omega500", "rh850", "z200"
]
C, H, W = 14, 132, 140
LAT = np.linspace(5.0, 38.0, H)
LON = np.linspace(65.0, 100.0, W)

CLIM_MEAN = np.array([
    300.5, 100200., 3.2, 2.1, 4.5, 301.2,
    5880., 295., 0.007, 4.5, 3.8, -0.12,
    72., 12300.
], np.float32)

CLIM_STD = np.array([
    4.2, 500., 3.1, 2.8, 6.2, 1.8,
    80., 3.5, 0.002, 3.2, 3.0, 0.08,
    15., 250.
], np.float32)

LEAD_TIMES = [24, 72, 120, 168]
TARGET_FEATS = ["tp", "rh850", "q700"]
TARGET_IDX = [FEATURES.index(f) for f in TARGET_FEATS]
N_CASES = 15
N_ENS = 8


# ═══════════════════════════════════════════════════════════════
# DATA LOADING
# ═══════════════════════════════════════════════════════════════
def load_real_era5(quick=False):
    """
    Load ERA5 from WeatherBench2 GCS bucket.
    quick=True  -> proper 2-week slice (2020-06-01 to 2020-06-14)
    quick=False -> JJAS 2020-2021 slice
    """
    if not REAL_AVAILABLE:
        warn("xarray/gcsfs not installed. Falling back to simulation.")
        return None

    log("  Connecting to WeatherBench2 GCS...", "cyan")
    try:
        fs = gcsfs.GCSFileSystem(token="anon")
        url = (
            "gs://weatherbench2/datasets/era5/"
            "1959-2023_01_10-6h-240x121_equiangular_with_poles_conservative.zarr"
        )
        store = fs.get_mapper(url)
        ds = xr.open_zarr(store, consolidated=True)
        ok("Connected to WeatherBench2")

        if quick:
            ds = ds.sel(time=slice("2020-06-01", "2020-06-14"))
            ok("Quick mode: proper 2-week slice (2020-06-01 to 2020-06-14)")
        else:
            ds = ds.sel(time=ds.time.dt.month.isin([6, 7, 8, 9]))
            ds = ds.sel(time=ds.time.dt.year.isin([2020, 2021]))
            ok("Full mode: JJAS 2020-2021")

        lat_name = "latitude" if "latitude" in ds.coords else "lat"
        lon_name = "longitude" if "longitude" in ds.coords else "lon"
        ds = ds.rename({lat_name: "lat", lon_name: "lon"})

        ds = ds.assign_coords(lon=((ds.lon + 180) % 360) - 180)
        ds = ds.sortby("lon")

        try:
            if float(ds.lat.values[0]) < float(ds.lat.values[-1]):
                ds = ds.sortby("lat", ascending=False)
        except Exception:
            ds = ds.sortby("lat", ascending=False)

        ds = ds.sel(lat=slice(38.0, 5.0), lon=slice(65.0, 100.0))
        ok(f"Subsetted to Indian domain: lat={ds.sizes.get('lat')}, lon={ds.sizes.get('lon')}")

        if ds.sizes.get("lat", 0) < 2 or ds.sizes.get("lon", 0) < 2:
            warn("Subset too small after slicing; falling back to simulation.")
            return None

        return ds

    except Exception as e:
        warn(f"Could not connect to GCS: {e}")
        warn("Falling back to simulation mode.")
        return None


def resize_to_target(arr):
    """
    Resize (time, lat, lon) -> (time, 132, 140)
    """
    arr = np.nan_to_num(arr)
    if arr.ndim != 3:
        raise ValueError(f"Expected 3D array (time, lat, lon), got {arr.shape}")

    t, h, w = arr.shape
    if h == H and w == W:
        return arr.astype(np.float32)

    zoom_h = H / h
    zoom_w = W / w
    resized = np.stack(
        [zoom(arr[i], (zoom_h, zoom_w), order=1) for i in range(t)],
        axis=0
    )
    return resized.astype(np.float32)


def build_arrays_from_ds(ds):
    """Extract our 14-feature arrays from WeatherBench2 xarray Dataset."""
    log("  Extracting feature arrays from ERA5...", "cyan")

    WB2_MAP = {
        "t2m": ("2m_temperature", None),
        "msl": ("mean_sea_level_pressure", None),
        "u10": ("10m_u_component_of_wind", None),
        "v10": ("10m_v_component_of_wind", None),
        "tp": ("total_precipitation_6hr", None),
        "sst": ("sea_surface_temperature", None),
        "z500": ("geopotential", 500),
        "t850": ("temperature", 850),
        "q700": ("specific_humidity", 700),
        "u850": ("u_component_of_wind", 850),
        "v850": ("v_component_of_wind", 850),
        "omega500": ("vertical_velocity", 500),
        "rh850": ("relative_humidity", 850),
        "z200": ("geopotential", 200),
    }

    arrays = []
    n_times = int(ds.sizes.get("time", len(ds.time)))

    for feat in FEATURES:
        varname, level = WB2_MAP[feat]
        try:
            da = ds[varname]

            if level is not None:
                if "level" in da.dims or "level" in da.coords:
                    da = da.sel(level=level)
                elif "pressure_level" in da.dims or "pressure_level" in da.coords:
                    da = da.sel(pressure_level=level)

            arr = np.asarray(da.values, dtype=np.float32)

            if arr.ndim != 3:
                raise ValueError(f"{feat}: expected 3D (time, lat, lon), got {arr.shape}")

            arr = resize_to_target(arr)

            if feat == "tp" and np.nanmax(arr) < 1.0:
                arr *= 1000.0

            if feat == "tp":
                arr = np.clip(arr, 0, None)
            if feat == "rh850":
                arr = np.clip(arr, 0, 100)

            arr = np.nan_to_num(arr, nan=0.0, posinf=0.0, neginf=0.0)

            ok(f"{feat:>10}: {arr.shape} mean={arr.mean():.4f}")
        except Exception as e:
            warn(f"{feat:>10}: not found in dataset ({e}), using simulation")
            arr = simulate_feature(feat, n_times=n_times)

        arrays.append(arr)

    return arrays


def simulate_feature(feat, n_times=60, seed=None):
    """Generate physically-calibrated synthetic ERA5 field."""
    ci = FEATURES.index(feat)
    if seed is None:
        seed = ci * 13

    rng = np.random.default_rng(seed)
    raw = rng.standard_normal((n_times, H, W), dtype=np.float32)

    for t in range(n_times):
        raw[t] = gaussian_filter(raw[t], sigma=7)

    arr = CLIM_MEAN[ci] + CLIM_STD[ci] * raw

    if feat == "tp":
        arr = np.clip(arr, 0, None)
    if feat == "rh850":
        arr = np.clip(arr, 0, 100)

    return arr.astype(np.float32)


def build_sim_arrays(n_times=60):
    log("  Generating simulation arrays...", "cyan")
    arrays = []
    for feat in FEATURES:
        arr = simulate_feature(feat, n_times=n_times)
        ok(f"{feat:>10}: {arr.shape} mean={arr.mean():.4f}")
        arrays.append(arr)
    return arrays


# ═══════════════════════════════════════════════════════════════
# TENSOR BUILDING
# ═══════════════════════════════════════════════════════════════
def normalise_arrays(arrays):
    return [
        (arr - CLIM_MEAN[ci]) / (CLIM_STD[ci] + 1e-8)
        for ci, arr in enumerate(arrays)
    ]


def denormalise_field(field, ci):
    raw = field * CLIM_STD[ci] + CLIM_MEAN[ci]
    if FEATURES[ci] == "tp":
        raw = np.clip(raw, 0, None)
    if FEATURES[ci] == "rh850":
        raw = np.clip(raw, 0, 100)
    return raw.astype(np.float32)


def denormalise_state(state):
    """
    state: (C, H, W) in normalised space
    returns: (C, H, W) in physical/raw space
    """
    raw = state * CLIM_STD[:, None, None] + CLIM_MEAN[:, None, None]
    raw = raw.astype(np.float32)
    raw[FEATURES.index("tp")] = np.clip(raw[FEATURES.index("tp")], 0, None)
    raw[FEATURES.index("rh850")] = np.clip(raw[FEATURES.index("rh850")], 0, 100)
    return raw


def denormalise_arrays(arrays):
    return [denormalise_field(arr, ci) for ci, arr in enumerate(arrays)]


def get_field(arrays, ci, t):
    return arrays[ci][t]  # (H, W)


def get_state(arrays, t):
    return np.stack([arrays[ci][t] for ci in range(C)], axis=0)  # (C, H, W)


# ═══════════════════════════════════════════════════════════════
# GRAPHCAST SIMULATOR
# ═══════════════════════════════════════════════════════════════
def graphcast_forecast(truth_t0, lead_h):
    """
    Autoregressive rollout in normalised space.
    truth_t0: (C, H, W) normalised
    """
    steps = lead_h // 6
    base = np.array([
        0.06, 0.06, 0.06, 0.06, 0.08, 0.06,
        0.06, 0.06, 0.06, 0.06, 0.06,
        0.07, 0.06, 0.06
    ], np.float32)

    sigma = base * np.sqrt(max(steps, 1)) * 0.55
    state = truth_t0.copy()

    for _ in range(steps):
        noise = np.stack(
            [
                gaussian_filter(np.random.randn(H, W).astype(np.float32), sigma=6)
                for _ in range(C)
            ],
            axis=0,
        )
        state = state + sigma[:, None, None] * noise

    return state.astype(np.float32)


# ═══════════════════════════════════════════════════════════════
# GENCAST DIFFUSION POST-PROCESSOR
# ═══════════════════════════════════════════════════════════════
def gencast_postprocess(gc_field, ci, lead_h, n_members=N_ENS, n_steps=12):
    """
    Diffusion-style stochastic refinement in normalised space.
    gc_field: (H, W)
    """
    feat = FEATURES[ci]
    diff_sigma = {"tp": 0.10, "rh850": 0.08, "q700": 0.06}
    base_sigma = diff_sigma.get(feat, 0.06)
    lead_factor = np.sqrt(max(lead_h, 6) / 24.0)

    members = []
    for _ in range(n_members):
        field = gc_field.copy()
        for step in range(n_steps):
            t_frac = 1.0 - step / n_steps
            sigma_t = base_sigma * lead_factor * t_frac

            coarse = gaussian_filter(np.random.randn(H, W).astype(np.float32), sigma=12)
            fine = gaussian_filter(np.random.randn(H, W).astype(np.float32), sigma=2)
            noise = (0.7 * coarse + 0.3 * fine).astype(np.float32)

            if feat == "tp":
                field = field + sigma_t * noise
            else:
                field = field + sigma_t * noise

            field = field + 0.08 * (gaussian_filter(field, sigma=3) - field)

        members.append(field.astype(np.float32))

    return np.stack(members, axis=0)


# ═══════════════════════════════════════════════════════════════
# METRICS
# ═══════════════════════════════════════════════════════════════
def wmse_rmse(pred, truth):
    w = np.cos(np.deg2rad(LAT))[:, None]
    return float(np.sqrt(((pred - truth) ** 2 * w).sum() / (w.sum() * W)))


def acc_score(pred, truth, clim):
    fa = pred - clim
    ta = truth - clim
    return float((fa * ta).sum() / (np.sqrt((fa**2).sum() * (ta**2).sum()) + 1e-10))


def crps_score(ensemble, truth):
    N = ensemble.shape[0]
    mae = float(np.abs(ensemble - truth[None]).mean())
    sp = sum(
        float(np.abs(ensemble[i] - ensemble[j]).mean())
        for i in range(N) for j in range(i + 1, N)
    )
    sp /= (N * (N - 1) / 2) if N > 1 else 1
    return mae - 0.5 * sp


def f1_score(pred, truth, p=85):
    """
    Stable F1 for extreme-event detection.
    Threshold is based on the truth field percentile.
    """
    thr = np.percentile(truth, p)
    yp = pred >= thr
    yt = truth >= thr

    tp = int((yp & yt).sum())
    fp = int((yp & ~yt).sum())
    fn = int((~yp & yt).sum())

    if tp == 0 and fp == 0 and fn == 0:
        return 0.0

    precision = tp / (tp + fp + 1e-9)
    recall = tp / (tp + fn + 1e-9)

    if precision + recall < 1e-12:
        return 0.0

    return float(2 * precision * recall / (precision + recall + 1e-9))


def percent_change(old, new, higher_is_better=False):
    """
    Returns a formatted percentage change string.
    """
    if abs(old) < 1e-12:
        return "n/a"
    if higher_is_better:
        return f"{((new - old) / abs(old)) * 100:+.2f}%"
    return f"{((old - new) / abs(old)) * 100:+.2f}%"


# ═══════════════════════════════════════════════════════════════
# CYCLONE OVERLAY HELPERS
# ═══════════════════════════════════════════════════════════════
def simulate_cyclone_track(n_steps=10):
    lat = [12.0]
    lon = [88.0]
    for _ in range(n_steps):
        lat.append(lat[-1] + np.random.uniform(0.5, 1.0))
        lon.append(lon[-1] + np.random.uniform(-0.5, 0.3))
    return np.array(lat), np.array(lon)


def load_cyclone_track_csv(path):
    """
    Optional CSV format:
        lat,lon
        12.0,88.0
        12.7,87.8
    """
    if path is None or not os.path.exists(path):
        return None

    import csv

    lats, lons = [], []
    with open(path, "r", newline="", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        if reader.fieldnames is None:
            return None

        fields = {name.lower(): name for name in reader.fieldnames}
        lat_key = fields.get("lat") or fields.get("latitude")
        lon_key = fields.get("lon") or fields.get("longitude")

        if lat_key is None or lon_key is None:
            return None

        for row in reader:
            try:
                lats.append(float(row[lat_key]))
                lons.append(float(row[lon_key]))
            except Exception:
                continue

    if len(lats) < 2:
        return None

    return np.array(lats), np.array(lons)


# ═══════════════════════════════════════════════════════════════
# PLOTTING HELPERS
# ═══════════════════════════════════════════════════════════════
def save_figure(fig, out_dir, basename):
    png_path = os.path.join(out_dir, f"{basename}.png")
    pdf_path = os.path.join(out_dir, f"{basename}.pdf")
    fig.savefig(png_path, dpi=150, bbox_inches="tight")
    fig.savefig(pdf_path, bbox_inches="tight")
    plt.close(fig)
    ok(f"{basename}.png / {basename}.pdf")


def setup_india_map(ax):
    if not CARTOPY_AVAILABLE:
        ax.set_xlim(65, 100)
        ax.set_ylim(5, 38)
        ax.set_xlabel("Lon (°E)")
        ax.set_ylabel("Lat (°N)")
        ax.grid(True, alpha=0.3)
        return

    ax.set_extent([65, 100, 5, 38], crs=ccrs.PlateCarree())
    ax.add_feature(cfeature.LAND, facecolor="whitesmoke", alpha=0.35, zorder=0)
    ax.add_feature(cfeature.OCEAN, facecolor="aliceblue", alpha=0.35, zorder=0)
    ax.add_feature(cfeature.COASTLINE, linewidth=0.9, zorder=3)
    ax.add_feature(cfeature.BORDERS, linewidth=0.7, zorder=3)
    ax.add_feature(cfeature.LAKES, alpha=0.25, zorder=1)
    ax.add_feature(cfeature.RIVERS, linewidth=0.4, alpha=0.5, zorder=1)
    gl = ax.gridlines(draw_labels=True, linewidth=0.35, alpha=0.45, linestyle="--")
    gl.top_labels = False
    gl.right_labels = False
    gl.xlabel_style = {"size": 8}
    gl.ylabel_style = {"size": 8}


def contourf_india(ax, data, title, cmap="RdBu_r", levels=21, vmin=None, vmax=None,
                   cbar_label=None, extend="both"):
    if CARTOPY_AVAILABLE:
        contour = ax.contourf(
            LON, LAT, data,
            levels=levels,
            cmap=cmap,
            vmin=vmin,
            vmax=vmax,
            transform=ccrs.PlateCarree(),
            extend=extend,
            zorder=2,
        )
    else:
        contour = ax.contourf(
            LON, LAT, data,
            levels=levels,
            cmap=cmap,
            vmin=vmin,
            vmax=vmax,
            extend=extend,
        )

    setup_india_map(ax)
    ax.set_title(title, fontsize=10, fontweight="bold")
    cbar = plt.colorbar(contour, ax=ax, shrink=0.82, pad=0.03)
    if cbar_label is not None:
        cbar.set_label(cbar_label, fontsize=9)
    return contour


# ═══════════════════════════════════════════════════════════════
# EVALUATION
# ═══════════════════════════════════════════════════════════════
def run_evaluation(arrays_norm):
    log("  Running evaluation...", "cyan")
    n_times = arrays_norm[0].shape[0]
    max_steps = max(LEAD_TIMES) // 6
    n_valid = n_times - max_steps - 2

    if n_valid < 5:
        warn(f"Only {n_valid} valid cases. Using all.")
        n_cases = max(n_valid, 1)
    else:
        n_cases = min(N_CASES, n_valid)

    gc_res = {f: {lt: {"rmse": [], "acc": [], "f1": [], "crps": []} for lt in LEAD_TIMES}
              for f in TARGET_FEATS}
    fus_res = {f: {lt: {"rmse": [], "acc": [], "f1": [], "crps": []} for lt in LEAD_TIMES}
               for f in TARGET_FEATS}

    for case in range(n_cases):
        t0 = case
        truth_t0_norm = get_state(arrays_norm, t0)

        for lead_h in LEAD_TIMES:
            steps = lead_h // 6
            truth_norm = get_state(arrays_norm, min(t0 + steps, n_times - 1))
            gc_pred_norm = graphcast_forecast(truth_t0_norm, lead_h)

            truth_raw = denormalise_state(truth_norm)
            gc_pred_raw = denormalise_state(gc_pred_norm)

            for ci, feat in zip(TARGET_IDX, TARGET_FEATS):
                gt = truth_raw[ci]
                gp = gc_pred_raw[ci]
                cm = float(CLIM_MEAN[ci])

                gc_res[feat][lead_h]["rmse"].append(wmse_rmse(gp, gt))
                gc_res[feat][lead_h]["acc"].append(acc_score(gp, gt, cm))
                gc_res[feat][lead_h]["f1"].append(f1_score(gp, gt))
                ens_gc = np.stack([gp] * N_ENS)
                gc_res[feat][lead_h]["crps"].append(crps_score(ens_gc, gt))

                ens_norm = gencast_postprocess(gc_pred_norm[ci], ci, lead_h)
                ens_raw = np.stack([denormalise_field(m, ci) for m in ens_norm], axis=0)
                fp_ = ens_raw.mean(0)

                fus_res[feat][lead_h]["rmse"].append(wmse_rmse(fp_, gt))
                fus_res[feat][lead_h]["acc"].append(acc_score(fp_, gt, cm))
                fus_res[feat][lead_h]["f1"].append(f1_score(fp_, gt))
                fus_res[feat][lead_h]["crps"].append(crps_score(ens_raw, gt))

        if (case + 1) % 5 == 0:
            ok(f"Case {case + 1}/{n_cases} done")

    def avg(d):
        return {
            f: {lt: {k: float(np.mean(v)) for k, v in d[f][lt].items()}
                for lt in LEAD_TIMES}
            for f in TARGET_FEATS
        }

    return avg(gc_res), avg(fus_res)


# ═══════════════════════════════════════════════════════════════
# PLOTTING
# ═══════════════════════════════════════════════════════════════
def plot_all(gc, fus, arrays_norm, out_dir, cyclone_track_csv=None):
    os.makedirs(out_dir, exist_ok=True)

    arrays_raw = denormalise_arrays(arrays_norm)

    feat_colors = {"tp": "#f97316", "rh850": "#3b82f6", "q700": "#10b981"}
    feat_labels = {"tp": "TP (mm/6h)", "rh850": "RH850 (%)", "q700": "Q700 (kg/kg)"}

    # ── Fig 1: RMSE comparison ──────────────────────────────
    fig, axes = plt.subplots(1, 3, figsize=(15, 5))
    for ax, feat in zip(axes, TARGET_FEATS):
        gc_v = [gc[feat][lt]["rmse"] for lt in LEAD_TIMES]
        fus_v = [fus[feat][lt]["rmse"] for lt in LEAD_TIMES]

        ax.plot(LEAD_TIMES, gc_v, "o-", color="#3b82f6", lw=2.2, ms=7, label="GraphCast")
        ax.plot(LEAD_TIMES, fus_v, "s--", color="#10b981", lw=2.2, ms=7, label="GC+GenCast")
        ax.fill_between(LEAD_TIMES, gc_v, fus_v, alpha=0.15, color="#10b981")

        for lt, gv, fv in zip(LEAD_TIMES, gc_v, fus_v):
            imp = (gv - fv) / (gv + 1e-12) * 100
            if imp > 0:
                ax.annotate(
                    f"↓{imp:.1f}%",
                    xy=(lt, fv),
                    xytext=(0, -16),
                    textcoords="offset points",
                    ha="center",
                    fontsize=8,
                    color="#059669",
                    fontweight="bold",
                )

        ax.set_title(feat_labels[feat], fontsize=10, fontweight="bold")
        ax.set_xlabel("Lead Time (h)")
        ax.set_ylabel("RMSE")
        ax.legend(fontsize=9)
        ax.grid(True, alpha=0.3)
        ax.set_xticks(LEAD_TIMES)

    fig.suptitle(
        "RMSE: GraphCast vs GC+GenCast — Indian Subcontinent",
        fontsize=13,
        fontweight="bold",
    )
    plt.tight_layout()
    save_figure(fig, out_dir, "fig1_rmse")

    # ── Fig 2: CRPS comparison ──────────────────────────────
    fig, axes = plt.subplots(1, 3, figsize=(15, 5))
    for ax, feat in zip(axes, TARGET_FEATS):
        gc_v = [gc[feat][lt]["crps"] for lt in LEAD_TIMES]
        fus_v = [fus[feat][lt]["crps"] for lt in LEAD_TIMES]

        ax.plot(
            LEAD_TIMES, gc_v, "o-", color="#3b82f6", lw=2.2, ms=7,
            label="GraphCast (degenerate)"
        )
        ax.plot(
            LEAD_TIMES, fus_v, "s--", color="#10b981", lw=2.2, ms=7,
            label=f"GC+GenCast (N={N_ENS})"
        )
        ax.fill_between(LEAD_TIMES, gc_v, fus_v, alpha=0.15, color="#10b981")
        ax.set_title(f"CRPS — {feat.upper()}", fontsize=10, fontweight="bold")
        ax.set_xlabel("Lead Time (h)")
        ax.set_ylabel("CRPS (↓ better)")
        ax.legend(fontsize=9)
        ax.grid(True, alpha=0.3)
        ax.set_xticks(LEAD_TIMES)

    fig.suptitle(
        "CRPS: Probabilistic Skill — GC+GenCast Fusion",
        fontsize=13,
        fontweight="bold",
    )
    plt.tight_layout()
    save_figure(fig, out_dir, "fig2_crps")

    # ── Fig 3: Improvement % bar chart ─────────────────────
    fig, axes = plt.subplots(1, 3, figsize=(15, 5))
    for ax, feat in zip(axes, TARGET_FEATS):
        metrics = ["RMSE↓", "F1↑", "CRPS↓"]
        keys = ["rmse", "f1", "crps"]
        x = np.arange(len(metrics))
        w = 0.15
        lt_colors = ["#1e3a5f", "#2563eb", "#60a5fa", "#bfdbfe"]

        for i, (lt, col) in enumerate(zip(LEAD_TIMES, lt_colors)):
            imps = []
            for k in keys:
                gv = gc[feat][lt][k]
                fv = fus[feat][lt][k]
                if k in ["rmse", "crps"]:
                    imps.append((gv - fv) / (gv + 1e-12) * 100)
                else:
                    imps.append((fv - gv) / (gv + 1e-12) * 100)
            ax.bar(x + (i - 1.5) * w, imps, w, label=f"{lt}h", color=col, alpha=0.9)

        ax.axhline(0, color="black", lw=1)
        ax.set_xticks(x)
        ax.set_xticklabels(metrics, fontsize=10, fontweight="bold")
        ax.set_ylabel("% Improvement")
        ax.set_title(
            f"{feat.upper()} — Fusion Gain",
            fontsize=11,
            fontweight="bold",
            color=feat_colors[feat],
        )
        ax.legend(fontsize=8)
        ax.grid(axis="y", alpha=0.3)

    fig.suptitle(
        "GC+GenCast % Improvement over GraphCast Baseline",
        fontsize=13,
        fontweight="bold",
    )
    plt.tight_layout()
    save_figure(fig, out_dir, "fig3_improvement")

    # ── Fig 4: Spatial contour maps (Cartopy) ──────────────
    n_times = arrays_norm[0].shape[0]
    n_sp = min(8, n_times - max(LEAD_TIMES) // 6 - 2)
    bias_gc = {f: np.zeros((H, W), dtype=np.float32) for f in TARGET_FEATS}
    bias_fu = {f: np.zeros((H, W), dtype=np.float32) for f in TARGET_FEATS}

    for case in range(n_sp):
        truth_t0_norm = get_state(arrays_norm, case)
        truth_norm = get_state(arrays_norm, min(case + 12, n_times - 1))
        gc_pred_norm = graphcast_forecast(truth_t0_norm, 72)

        truth_raw = denormalise_state(truth_norm)
        gc_pred_raw = denormalise_state(gc_pred_norm)

        for ci, feat in zip(TARGET_IDX, TARGET_FEATS):
            ens_norm = gencast_postprocess(gc_pred_norm[ci], ci, 72)
            ens_raw = np.stack([denormalise_field(m, ci) for m in ens_norm], axis=0)

            bias_gc[feat] += (gc_pred_raw[ci] - truth_raw[ci])
            bias_fu[feat] += (ens_raw.mean(0) - truth_raw[ci])

    for f in TARGET_FEATS:
        bias_gc[f] /= max(n_sp, 1)
        bias_fu[f] /= max(n_sp, 1)

    if CARTOPY_AVAILABLE:
        fig, axes = plt.subplots(
            3, 3, figsize=(16, 12),
            subplot_kw={"projection": ccrs.PlateCarree()}
        )
    else:
        fig, axes = plt.subplots(3, 3, figsize=(16, 12))

    for row, feat in enumerate(TARGET_FEATS):
        ci = TARGET_IDX[row]
        field = arrays_raw[ci][0]
        vb = np.percentile(np.abs(bias_gc[feat]), 95)
        levels = np.linspace(-vb, vb, 21)

        ax = axes[row, 0]
        contourf_india(
            ax,
            field,
            f"{feat.upper()} Field\n(ERA5 sample)",
            cmap="YlOrRd",
            levels=21,
            cbar_label=feat.upper()
        )

        ax = axes[row, 1]
        contourf_india(
            ax,
            bias_gc[feat],
            "GraphCast Bias\n(FC − Truth)",
            cmap="RdBu_r",
            levels=levels,
            vmin=-vb,
            vmax=vb,
            cbar_label="Bias",
            extend="both",
        )

        ax = axes[row, 2]
        contourf_india(
            ax,
            bias_fu[feat],
            "GC+GenCast Bias\n(Reduced)",
            cmap="RdBu_r",
            levels=levels,
            vmin=-vb,
            vmax=vb,
            cbar_label="Bias",
            extend="both",
        )

    fig.suptitle(
        "Spatial Bias Maps at 72-h Lead\nTP | RH850 | Q700 — Indian Domain",
        fontsize=13,
        fontweight="bold",
    )
    plt.tight_layout()
    save_figure(fig, out_dir, "fig4_spatial_bias")

    # ── Fig 5: TP ensemble spread (Cartopy) ────────────────
    truth_t0_norm = get_state(arrays_norm, 0)
    gc_pred_norm = graphcast_forecast(truth_t0_norm, 72)
    ci_tp = FEATURES.index("tp")
    ens_tp_norm = gencast_postprocess(gc_pred_norm[ci_tp], ci_tp, 72)
    ens_tp = np.stack([denormalise_field(m, ci_tp) for m in ens_tp_norm], axis=0)
    gc_pred_raw = denormalise_state(gc_pred_norm)

    if CARTOPY_AVAILABLE:
        fig, axes = plt.subplots(
            2, 4, figsize=(18, 8),
            subplot_kw={"projection": ccrs.PlateCarree()}
        )
    else:
        fig, axes = plt.subplots(2, 4, figsize=(18, 8))

    vmax = max(np.percentile(ens_tp, 97), 0.1)

    for i in range(4):
        ax = axes[0, i]
        contourf_india(
            ax,
            ens_tp[i],
            f"GenCast Member {i + 1}",
            cmap="Blues",
            levels=21,
            vmin=0,
            vmax=vmax,
            cbar_label="mm/6h",
            extend="max",
        )

    panels = [
        (ens_tp.mean(0), "Ensemble Mean (μ)", "Blues", 0, vmax, "mm/6h"),
        (ens_tp.std(0), "Ensemble Spread (σ)", "Oranges", 0, ens_tp.std(0).max() + 0.01, "σ"),
        (gc_pred_raw[ci_tp], "GraphCast Deterministic", "Blues", 0, vmax, "mm/6h"),
        (arrays_raw[ci_tp][0], "ERA5 Sample Field", "Blues", 0, vmax, "mm/6h"),
    ]

    for ax, (data, title, cmap, vmin, vm, cbarlab) in zip(axes[1], panels):
        contourf_india(
            ax,
            data,
            title,
            cmap=cmap,
            levels=21,
            vmin=vmin,
            vmax=vm,
            cbar_label=cbarlab,
            extend="max",
        )

    fig.suptitle(
        "Total Precipitation (TP) — GenCast Ensemble at 72-h Lead\n"
        "Indian Subcontinent | Styled after Yan et al. (2025)",
        fontsize=13,
        fontweight="bold",
    )
    plt.tight_layout()
    save_figure(fig, out_dir, "fig5_ensemble_spread")

    # ── Fig 6: Summary metrics heatmap ─────────────────────
    fig, axes = plt.subplots(2, 3, figsize=(16, 9))
    metrics = [("rmse", "RMSE", "Reds"), ("f1", "F1", "Greens"), ("crps", "CRPS", "Purples")]

    for col, (key, label, cmap) in enumerate(metrics):
        for row, (res, title) in enumerate([(gc, "GraphCast"), (fus, "GC+GenCast")]):
            ax = axes[row, col]
            data = np.array([[res[f][lt][key] for lt in LEAD_TIMES] for f in TARGET_FEATS])

            im = ax.imshow(
                data,
                aspect="auto",
                cmap=cmap,
                vmin=0 if key != "f1" else 0.0,
                vmax=float(data.max()) if key != "f1" else max(1.0, float(data.max())),
                origin="upper",
            )

            ax.set_xticks(range(len(LEAD_TIMES)))
            ax.set_xticklabels([f"{lt}h" for lt in LEAD_TIMES], fontsize=9)
            ax.set_yticks(range(len(TARGET_FEATS)))
            ax.set_yticklabels([f.upper() for f in TARGET_FEATS], fontsize=9)
            ax.set_title(f"{title}\n{label}", fontsize=10, fontweight="bold")

            for i in range(len(TARGET_FEATS)):
                for j in range(len(LEAD_TIMES)):
                    ax.text(j, i, f"{data[i, j]:.3f}", ha="center", va="center", fontsize=8)

            plt.colorbar(im, ax=ax, shrink=0.85)

    fig.suptitle(
        "Metrics Heatmap: GraphCast (top) vs GC+GenCast (bottom)\n"
        "TP | RH850 | Q700 across all lead times",
        fontsize=13,
        fontweight="bold",
    )
    plt.tight_layout()
    save_figure(fig, out_dir, "fig6_heatmap")

    # ── Fig 7: Cyclone track overlay ───────────────────────
    track = load_cyclone_track_csv(cyclone_track_csv)
    if track is None:
        track_lats, track_lons = simulate_cyclone_track(n_steps=10)
    else:
        track_lats, track_lons = track

    mslp_field = arrays_raw[FEATURES.index("msl")][0]

    if CARTOPY_AVAILABLE:
        fig = plt.figure(figsize=(10, 8))
        ax = plt.axes(projection=ccrs.PlateCarree())
    else:
        fig, ax = plt.subplots(figsize=(10, 8))

    contourf_india(
        ax,
        mslp_field,
        "Cyclone Track Overlay on Mean Sea Level Pressure",
        cmap="Spectral_r",
        levels=21,
        cbar_label="Pa",
        extend="both",
    )

    if CARTOPY_AVAILABLE:
        ax.plot(
            track_lons, track_lats,
            "-o", color="black", linewidth=2.2, markersize=4.5,
            transform=ccrs.PlateCarree(),
            label="Cyclone Track",
            zorder=5,
        )
        ax.scatter(
            track_lons[0], track_lats[0],
            s=90, color="limegreen", edgecolor="black",
            transform=ccrs.PlateCarree(),
            zorder=6,
            label="Genesis",
        )
        ax.scatter(
            track_lons[-1], track_lats[-1],
            s=90, color="red", edgecolor="black",
            transform=ccrs.PlateCarree(),
            zorder=6,
            label="End",
        )
        ax.legend(loc="lower left", fontsize=9)
        ax.text(
            66.2, 36.2,
            "Cyclone overlay\n(synthetic by default)",
            fontsize=10,
            fontweight="bold",
            transform=ccrs.PlateCarree(),
            bbox=dict(
                boxstyle="round,pad=0.35",
                facecolor="white",
                alpha=0.75,
                edgecolor="gray",
            ),
        )
    else:
        ax.plot(track_lons, track_lats, "-o", color="black", linewidth=2.2, markersize=4.5, label="Cyclone Track")
        ax.scatter(track_lons[0], track_lats[0], s=90, color="limegreen", edgecolor="black", label="Genesis")
        ax.scatter(track_lons[-1], track_lats[-1], s=90, color="red", edgecolor="black", label="End")
        ax.legend(loc="lower left", fontsize=9)

    fig.suptitle(
        "Cyclone Track Overlay — Indian Subcontinent",
        fontsize=13,
        fontweight="bold",
    )
    plt.tight_layout()
    save_figure(fig, out_dir, "fig7_cyclone_overlay")


# ═══════════════════════════════════════════════════════════════
# PRINT RESULTS TABLE
# ═══════════════════════════════════════════════════════════════
def print_results(gc, fus):
    print()
    log("=" * 70, "bold")
    log("  FINAL RESULTS — GraphCast vs GC+GenCast Fusion", "bold")
    log("  Indian Subcontinent | 72-h Lead Time", "bold")
    log("=" * 70, "bold")
    print(f"{'Feature':>8}  {'Metric':>6}  {'GraphCast':>12}  {'GC+GenCast':>12}  {'Improve':>10}")
    print("-" * 65)

    for feat in TARGET_FEATS:
        for key, label, better in [("rmse", "RMSE", "↓"), ("f1", "F1", "↑"), ("crps", "CRPS", "↓")]:
            gv = gc[feat][72][key]
            fv = fus[feat][72][key]

            if better == "↓":
                imp = percent_change(gv, fv, higher_is_better=False)
            else:
                imp = percent_change(gv, fv, higher_is_better=True)

            print(
                f"{feat.upper():>8}  {label:>6}  {gv:>12.5f}  {fv:>12.5f}  {imp:>10}"
            )
        print()


# ═══════════════════════════════════════════════════════════════
# MAIN
# ═══════════════════════════════════════════════════════════════
def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--mode", default="real", choices=["simulate", "real"])
    parser.add_argument("--quick", action="store_true")
    parser.add_argument("--out", default="./output_figures")
    parser.add_argument("--cases", type=int, default=15)
    parser.add_argument("--cyclone-track", type=str, default=None)

    args = parser.parse_args()

    global N_CASES
    N_CASES = args.cases

    log("\n" + "=" * 65, "bold")
    log(" GraphCast + GenCast Fusion Pipeline", "bold")
    log("=" * 65 + "\n", "bold")

    t_start = time.time()

    check_deps()

    step(1, "Loading ERA5 data...")
    arrays = None

    if args.mode == "real":
        ds = load_real_era5(quick=args.quick)
        if ds is not None:
            arrays = build_arrays_from_ds(ds)

    if arrays is None:
        warn("Fallback simulation")
        arrays = build_sim_arrays(n_times=60)

    arrays = normalise_arrays(arrays)

    step(2, "Running evaluation...")
    gc_res, fus_res = run_evaluation(arrays)

    step(3, "Generating figures...")
    plot_all(gc_res, fus_res, arrays, args.out,
             cyclone_track_csv=args.cyclone_track)

    step(4, "Results summary:")
    print_results(gc_res, fus_res)

    step(5, "Saving forecast sequence for web app...")
    state0 = get_state(arrays, 0)

    fc_seq_norm = []
    state = state0.copy()

    for _ in range(12):
        state = graphcast_forecast(state, 6)
        fc_seq_norm.append(state.copy())

    fc_seq_norm = np.array(fc_seq_norm)  # (T, C, H, W) normalized

    fc_seq_raw = np.stack([denormalise_state(s) for s in fc_seq_norm], axis=0)

    os.makedirs("data", exist_ok=True)
    np.save("data/fc_seq.npy", fc_seq_raw)

    ok("Saved → data/fc_seq.npy")

    elapsed = time.time() - t_start
    log(f"\n✓ Done in {elapsed:.1f}s", "green")

    gc.collect()


if __name__ == "__main__":
    main()