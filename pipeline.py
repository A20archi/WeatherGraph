import numpy as np
import matplotlib.pyplot as plt
import matplotlib.animation as animation
from scipy.ndimage import gaussian_filter
import cartopy.crs as ccrs
import cartopy.feature as cfeature

# ================= CONSTANTS =================
FEATURES = ['t2m','msl','u10','v10','tp','sst',
            'z500','t850','q700','u850','v850','omega500','rh850','z200']

C,H,W = 14,132,140
LAT = np.linspace(5,38,H)
LON = np.linspace(65,100,W)

# ================= MAP =================
def setup_map(ax):
    ax.set_extent([65,100,5,38])
    ax.add_feature(cfeature.COASTLINE)
    ax.add_feature(cfeature.BORDERS)

# ================= SIM DATA =================
def simulate():
    data = np.random.randn(60,C,H,W)
    for t in range(60):
        data[t] = gaussian_filter(data[t], sigma=6)
    return data

# ================= GRAPHCAST =================
def graphcast_seq(state, steps):
    seq = []
    for s in range(steps):
        noise = gaussian_filter(np.random.randn(C,H,W), sigma=5)
        state = state + 0.1*np.sqrt(s+1)*noise
        seq.append(state.copy())
    return np.array(seq)

# ================= GENCAST =================
def gencast(field):
    ens = []
    for _ in range(6):
        noise = gaussian_filter(np.random.randn(H,W), sigma=4)
        ens.append(field + 0.05*noise)
    return np.stack(ens)

def gencast_sequence(gc_seq, ci):
    all_ens = []
    for t in range(len(gc_seq)):
        all_ens.append(gencast(gc_seq[t,ci]))
    return np.array(all_ens)

# ================= CYCLONE =================
def simulate_cyclone_track():
    lat=[12]; lon=[88]
    for _ in range(10):
        lat.append(lat[-1]+np.random.uniform(0.5,1))
        lon.append(lon[-1]+np.random.uniform(-0.5,0.3))
    return np.array(lat),np.array(lon)

# ================= ANIMATION =================
def animate(fc_seq, ci, name):

    fig = plt.figure(figsize=(9,7))
    ax = plt.axes(projection=ccrs.PlateCarree())

    # FIXED COLOR SCALE (important)
    vmin = np.percentile(fc_seq[:,ci], 2)
    vmax = np.percentile(fc_seq[:,ci], 98)

    def update(frame):
        ax.clear()

        setup_map(ax)

        data = fc_seq[frame,ci]

        im = ax.contourf(
            LON, LAT, data,
            levels=20,
            cmap='RdBu_r',
            vmin=vmin,
            vmax=vmax,
            transform=ccrs.PlateCarree()
        )

        # Cyclone overlay
        lat_c, lon_c = simulate_cyclone_track()
        ax.plot(lon_c, lat_c, 'k-o',
                transform=ccrs.PlateCarree(),
                label="Cyclone Track")

        # Labels
        ax.set_title(
            f"{FEATURES[ci].upper()} Forecast | t+{(frame+1)*6}h",
            fontsize=12,
            fontweight='bold'
        )

        # Grid labels
        ax.gridlines(draw_labels=True, linewidth=0.3, alpha=0.5)

        return []

    ani = animation.FuncAnimation(
        fig, update,
        frames=len(fc_seq),
        interval=500,
        blit=False
    )

    # Add colorbar ONCE
    cbar = fig.colorbar(
        plt.cm.ScalarMappable(cmap='RdBu_r'),
        ax=ax,
        orientation='vertical',
        shrink=0.8
    )
    cbar.set_label(f"{FEATURES[ci]} value", fontsize=10)

    ani.save(f"figs/{name}.gif", writer='pillow')

    plt.close()

    print(f"Saved labeled animation: {name}.gif")
# ================= SPREAD =================
def animate_spread(fc_seq, ci):

    ens_seq = gencast_sequence(fc_seq, ci)

    fig = plt.figure(figsize=(9,7))
    ax = plt.axes(projection=ccrs.PlateCarree())

    vmax = np.percentile(ens_seq, 98)

    def update(frame):
        ax.clear()
        setup_map(ax)

        spread = ens_seq[frame].std(axis=0)

        ax.contourf(
            LON, LAT, spread,
            levels=20,
            cmap='Oranges',
            vmin=0,
            vmax=vmax,
            transform=ccrs.PlateCarree()
        )

        ax.set_title(
            f"{FEATURES[ci].upper()} Uncertainty (Spread)\n t+{(frame+1)*6}h",
            fontsize=12,
            fontweight='bold'
        )

        ax.gridlines(draw_labels=True, linewidth=0.3)

        return []

    ani = animation.FuncAnimation(fig, update,
                                  frames=len(fc_seq),
                                  interval=500,
                                  blit=False)

    cbar = fig.colorbar(
        plt.cm.ScalarMappable(cmap='Oranges'),
        ax=ax,
        shrink=0.8
    )
    cbar.set_label("Uncertainty (Std Dev)", fontsize=10)

    ani.save("figs/spread.gif", writer='pillow')
    plt.close()

    print("Saved labeled spread animation")

# ================= MAIN =================
if __name__ == "__main__":

    print("Generating synthetic ERA5...")
    arrays = simulate()

    state = arrays[0]

    print("Running GraphCast...")
    fc_seq = graphcast_seq(state, 12)

    # save for web app
    np.save("fc_seq.npy", fc_seq)

    import os
    os.makedirs("figs", exist_ok=True)

    print("Animating temperature...")
    animate(fc_seq, 0, "t2m")

    print("Animating precipitation...")
    animate(fc_seq, 4, "tp")

    print("Animating spread...")
    animate_spread(fc_seq, 4)

    print("DONE → figs/")