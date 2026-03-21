import pandas as pd
import matplotlib.pyplot as plt
from pathlib import Path
import numpy as np
import matplotlib.image as mpimg
import re

# === Load and plot background image ===
img_path = Path("/Users/margotferris/Desktop/grayscale_eye_cropped_to_bounding_box.png")
eye_img = mpimg.imread(img_path)

# === Control these parameters ===
image_opacity = .9             # 0 = transparent, 1 = opaque
image_width_mm = 34            # Desired image width in mm
x_shift_mm = 2.8               # shift image 1 mm to the left (adjust as needed)

# === Load image and compute its true aspect ratio ===
eye_img = mpimg.imread(img_path)
img_height_px, img_width_px = eye_img.shape[:2]
aspect_ratio = img_height_px / img_width_px
image_height_mm = image_width_mm * aspect_ratio

# === Image extent in mm, centered at (0, 0) ===
img_extent = [
    -image_width_mm / 2 + x_shift_mm, image_width_mm / 2 + x_shift_mm,
    -image_height_mm / 2, image_height_mm / 2
]

# === CONFIG ===
CSV_DIR = Path("/Users/margotferris/Desktop/repo/Brillouin_project/src/eye_position/xy_data/calibration2")
CSV_FILES = list(CSV_DIR.glob("xy_*.csv"))

if not CSV_FILES:
    print("No CSV files found.")
    exit()

# --- Filename parser: supports things like xy_-0.707_.707 (note: .707 w/o leading 0) ---
_num = r"-?(?:\d+(?:\.\d+)?|\.\d+)"
pat = re.compile(rf"^xy_({_num})_({_num})$")

records = []

# === Load each file: col1 x_mean, col2 y_mean, col3 x_std, col4 y_std ===
for file in CSV_FILES:
    try:
        m = pat.fullmatch(file.stem)
        if not m:
            raise ValueError(f"Filename does not match expected pattern xy_<x>_<y>: {file.stem}")

        x_ref = float(m.group(1))
        y_ref = float(m.group(2))

        df = pd.read_csv(file, header=None)

        if df.shape[1] < 2:
            raise ValueError("Expected at least 2 columns: x_mean, y_mean")

        x_meas = float(df.iloc[0, 0])
        y_meas = float(df.iloc[0, 1])

        # NEW: read std devs if present (col3, col4)
        x_std = float(df.iloc[0, 2]) if df.shape[1] >= 3 else np.nan
        y_std = float(df.iloc[0, 3]) if df.shape[1] >= 4 else np.nan

        dx = x_meas - x_ref
        dy = y_meas - y_ref
        err = np.hypot(dx, dy)
        r_ref = np.hypot(x_ref, y_ref)

        records.append({
            "file": file.name,
            "x_ref": x_ref,
            "y_ref": y_ref,
            "x_meas": x_meas,
            "y_meas": y_meas,
            "x_std": x_std,   # NEW
            "y_std": y_std,   # NEW
            "dx": dx,
            "dy": dy,
            "error": err,
            "r_ref": r_ref,
        })

    except Exception as e:
        print(f"[WARNING] Skipped {file.name}: {e}")

if not records:
    print("No valid files parsed.")
    exit()

df_all = pd.DataFrame(records)

# === Print individual errors ===
print("\nErrors between measured and filename reference points:")
for i, row in df_all.reset_index(drop=True).iterrows():
    print(
        f"Point {i+1:2d}: Error = {row['error']:.4f} mm | "
        f"Measured = [{row['x_meas']:.4f}, {row['y_meas']:.4f}], "
        f"Ref = [{row['x_ref']:.4f}, {row['y_ref']:.4f}] | "
        f"dx = {row['dx']:.4f}, dy = {row['dy']:.4f}"
    )

# === Summary statistics ===
errors = df_all["error"].to_numpy()
print("\nSummary statistics:")
print(f"  Mean error:     {np.mean(errors):.4f} mm")
print(f"  Max error:      {np.max(errors):.4f} mm")
print(f"  Std deviation:  {np.std(errors):.4f} mm")

# === Mean error per radial mm (error / radius), excluding r=0 ===
nonzero = df_all["r_ref"].to_numpy() > 0
normalized_errors = df_all.loc[nonzero, "error"].to_numpy() / df_all.loc[nonzero, "r_ref"].to_numpy()
if normalized_errors.size:
    mean_relative_error = np.mean(normalized_errors)
    print(f"\nThe error for each mm is approximately {mean_relative_error:.4f} mm/mm")
else:
    print("\nNo nonzero-radius reference points; cannot compute mm/mm normalization.")

# === Compute per-radius mean error ===
radius_errors = {r: [] for r in [0, 1, 2, 3, 4]}
for _, row in df_all.iterrows():
    r_bucket = round(row["r_ref"])
    if r_bucket in radius_errors:
        radius_errors[r_bucket].append(row["error"])

print("\nError by reference radius:")
for r in sorted(radius_errors):
    errs = radius_errors[r]
    if errs:
        print(f"  Radius {r} mm: Error = {np.mean(errs):.4f} mm")
    else:
        print(f"  Radius {r} mm: No points")

# =========================
# Plot
# =========================
fig, ax = plt.subplots(figsize=(6, 6))
ax.set_aspect("equal")

# Image
ax.imshow(eye_img, extent=img_extent, alpha=image_opacity, zorder=0)

# Reference points
ax.scatter(
    df_all["x_ref"], df_all["y_ref"],
    color="red", label="Reference Points", s=12, zorder=3
)

# Measured mean points (keep scatter if you want the dots)
ax.scatter(
    df_all["x_meas"], df_all["y_meas"],
    color="blue", label="Mean Measured Points", s=12, zorder=4
)

# NEW: error bars for measured points
# (matplotlib will ignore NaN error values)
ax.errorbar(
    df_all["x_meas"], df_all["y_meas"],
    xerr=df_all["x_std"], yerr=df_all["y_std"],
    fmt="none",            # don't redraw markers (we already did scatter)
    ecolor="blue",
    elinewidth=1.0,
    capsize=2,
    alpha=0.9,
    zorder=2
)

# Legend
ax.legend(loc="upper right", frameon=True)

# Axis limits
max_radius = 4.5
ax.set_xlim(-max_radius, max_radius)
ax.set_ylim(-max_radius, max_radius)

from matplotlib.patches import Circle

# =========================
# Polar-style overlay: spokes + circles
# =========================
overlay_color = "grey"
overlay_alpha = 0.6
overlay_lw = 0.8
reference_radii = [1, 2, 3, 4]

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

# Formatting
for spine in ['top', 'right']:
    ax.spines[spine].set_visible(False)

ax.spines['left'].set_position(('outward', 10))
ax.spines['bottom'].set_position(('outward', 10))

ax.tick_params(axis='both', direction='out', length=4)
ax.set_xlabel("X distance (mm)", fontsize=14, labelpad=10)
ax.set_ylabel("Y distance (mm)", fontsize=14, labelpad=10)
ax.set_title(" X-Y Human Eye Accuracy Test", fontsize=20)
ax.grid(False)

plt.tight_layout()
plt.show()
