import pandas as pd
import matplotlib.pyplot as plt
from pathlib import Path
import numpy as np
import re
from matplotlib.patches import Circle

# =========================
# CONFIG
# =========================
CSV_DIR = Path("/Users/margotferris/Desktop/repo/Brillouin_project/src/eye_position/drift_data")
CSV_FILES = sorted(CSV_DIR.glob("set_*.csv"))

if not CSV_FILES:
    print("No CSV files found.")
    raise SystemExit

# --- filename parser: set_1, set_2, etc. ---
pat = re.compile(r"^set_(\d+)$")

records = []

# =========================
# LOAD DATA
# =========================
for file in CSV_FILES:
    try:
        m = pat.fullmatch(file.stem)
        if not m:
            raise ValueError(f"Filename does not match expected pattern set_<number>: {file.stem}")

        set_id = int(m.group(1))

        df = pd.read_csv(file, header=None)

        if df.shape[1] < 3:
            raise ValueError("Expected at least 3 columns: index/time, x, y")

        # Parse columns
        sec_vals = pd.to_numeric(df.iloc[:, 0], errors="coerce")
        x_vals = pd.to_numeric(df.iloc[:, 1], errors="coerce")
        y_vals = pd.to_numeric(df.iloc[:, 2], errors="coerce")

        valid = ~(sec_vals.isna() | x_vals.isna() | y_vals.isna())
        sec_vals = sec_vals[valid].to_numpy(dtype=float)
        x_vals = x_vals[valid].to_numpy(dtype=float)
        y_vals = y_vals[valid].to_numpy(dtype=float)

        if len(x_vals) == 0:
            raise ValueError("No valid x,y measurements found in file")

        # Sort by sample/time column so the line connects in time order
        order = np.argsort(sec_vals)
        sec_vals = sec_vals[order]
        x_vals = x_vals[order]
        y_vals = y_vals[order]

        # Total movement = sum of step-to-step Euclidean distances
        dx = np.diff(x_vals)
        dy = np.diff(y_vals)
        step_distances = np.sqrt(dx**2 + dy**2)
        total_movement = np.sum(step_distances)

        # Net displacement = distance from first point to last point
        net_displacement = np.sqrt((x_vals[-1] - x_vals[0])**2 + (y_vals[-1] - y_vals[0])**2)

        # Total elapsed time
        total_time = sec_vals[-1] - sec_vals[0]
        if total_time <= 0:
            raise ValueError("Time column must increase so total time is positive")

        # Path length per second
        radial_movement_per_second = total_movement / total_time

        records.append({
            "file": file.name,
            "set_id": set_id,
            "sec": sec_vals,
            "x": x_vals,
            "y": y_vals,
            "total_movement": total_movement,
            "net_displacement": net_displacement,
            "total_time": total_time,
            "radial_movement_per_second": radial_movement_per_second,
            "n_points": len(x_vals),
        })

    except Exception as e:
        print(f"[WARNING] Skipped {file.name}: {e}")

if not records:
    print("No valid files parsed.")
    raise SystemExit

# Sort by set number
records = sorted(records, key=lambda r: r["set_id"])

# =========================
# PRINT MOVEMENT STATS
# =========================
print("\nMovement summary by file:")
for r in records:
    print(
        f"Set {r['set_id']:>2d} ({r['file']}): "
        f"points = {r['n_points']}, "
        f"time = {r['total_time']:.4f} s, "
        f"total movement = {r['total_movement']:.4f} mm, "
        f"net displacement = {r['net_displacement']:.4f} mm, "
        f"path movement per second = {r['radial_movement_per_second']:.4f} mm/s"
    )

all_totals = np.array([r["total_movement"] for r in records], dtype=float)
all_nets = np.array([r["net_displacement"] for r in records], dtype=float)
all_rates = np.array([r["radial_movement_per_second"] for r in records], dtype=float)

print("\nAcross all files:")
print(f"  Average total movement: {np.mean(all_totals):.4f} mm")
print(f"  Std of total movement:  {np.std(all_totals):.4f} mm")
print(f"  Average net displacement: {np.mean(all_nets):.4f} mm")
print(f"  Average path movement per second: {np.mean(all_rates):.4f} mm/s")
print(f"  Std of path movement per second:  {np.std(all_rates):.4f} mm/s")

# =========================
# PLOT
# =========================
fig, ax = plt.subplots(figsize=(7, 7))
ax.set_aspect("equal")

# Axis limits
max_radius = 1.2
ax.set_xlim(-max_radius, max_radius)
ax.set_ylim(-max_radius, max_radius)

# =========================
# Polar-style overlay: spokes + circles
# =========================
overlay_color = "grey"
overlay_alpha = 0.6
overlay_lw = 0.8
reference_radii = [1]

for r in reference_radii:
    ax.add_patch(
        Circle(
            (0, 0), r,
            fill=False,
            edgecolor=overlay_color,
            linewidth=overlay_lw,
            linestyle="--",
            alpha=overlay_alpha,
            zorder=1
        )
    )

spoke_angles_deg = [0, 45, 90, 135, 180, 225, 270, 315]
R = max_radius
for ang in spoke_angles_deg:
    theta = np.deg2rad(ang)
    ax.plot(
        [0, R * np.cos(theta)],
        [0, R * np.sin(theta)],
        linestyle="--",
        color=overlay_color,
        linewidth=overlay_lw,
        alpha=overlay_alpha,
        zorder=1
    )

# =========================
# Plot each file in a different color
# =========================
cmap = plt.get_cmap("tab10")

for i, r in enumerate(records):
    color = cmap(i % 10)

    # connecting line in time order
    ax.plot(
        r["x"], r["y"],
        "-", linewidth=1.5,
        color=color,
        alpha=0.9,
        label=f"Set {r['set_id']}",
        zorder=3
    )

    # scatter points
    ax.scatter(
        r["x"], r["y"],
        s=25,
        color=color,
        zorder=4
    )

    # mark first point
    ax.scatter(
        r["x"][0], r["y"][0],
        s=60,
        facecolors="none",
        edgecolors=color,
        linewidths=1.5,
        zorder=5
    )

# =========================
# Formatting
# =========================
for spine in ["top", "right"]:
    ax.spines[spine].set_visible(False)

ax.spines["left"].set_position(("outward", 10))
ax.spines["bottom"].set_position(("outward", 10))

ax.tick_params(axis="both", direction="out", length=4)
ax.set_xlabel("X distance (mm)", fontsize=14, labelpad=10)
ax.set_ylabel("Y distance (mm)", fontsize=14, labelpad=10)
ax.set_title("Patient Drift Over Time", fontsize=20)
ax.grid(False)

ax.legend(loc="upper right", frameon=True)

plt.tight_layout()
plt.show()