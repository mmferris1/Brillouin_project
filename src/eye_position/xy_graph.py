import pandas as pd
import matplotlib.pyplot as plt
from pathlib import Path
import numpy as np

# === CONFIG ===
SCRIPT_DIR = Path(__file__).resolve().parent
CSV_FILES = list(SCRIPT_DIR.glob("circlexy_*.csv"))

if not CSV_FILES:
    print("No CSV files found.")
    exit()

# === Load and combine all files ===
all_data = []
for file in CSV_FILES:
    try:
        df = pd.read_csv(file, header=None)
        if df.shape[1] != 5:
            raise ValueError("Expected 5 columns")
        df.columns = ["x_measured", "y_measured", "z", "x_true", "y_true"]

        # Convert from meters to millimeters
        df[["x_measured", "y_measured", "x_true", "y_true"]] *= 1000

        all_data.append(df)
    except Exception as e:
        print(f"[WARNING] Skipped {file.name}: {e}")

if not all_data:
    print("No valid files could be processed.")
    exit()

combined_df = pd.concat(all_data, ignore_index=True)

# === Compute center of measured data and center all points ===
x_center = combined_df["x_measured"].mean()
y_center = combined_df["y_measured"].mean()
combined_df["x_measured_centered"] = combined_df["x_measured"] - x_center
combined_df["y_measured_centered"] = combined_df["y_measured"] - y_center

# === Compute group-wise averages ===
grouped = combined_df.groupby(["x_true", "y_true"])
summary = grouped[["x_measured_centered", "y_measured_centered"]].mean().reset_index()

# === Compute overall standard deviation of all centered measurements ===
x_std = combined_df["x_measured_centered"].std()
y_std = combined_df["y_measured_centered"].std()

print(f"\nStandard deviation of all measured points (centered):")
print(f"  X std = {x_std:.4f} mm")
print(f"  Y std = {y_std:.4f} mm")

# === Generate red reference dots ===
radii = [1, 2, 3, 4]  # in mm
angles_deg = [0, 45, 90, 135, 180, 225, 270, 315]
angles_rad = np.deg2rad(angles_deg)

ref_x = []
ref_y = []

for r in radii:
    for theta in angles_rad:
        ref_x.append(r * np.cos(theta))
        ref_y.append(r * np.sin(theta))

# Add origin
ref_x.append(0)
ref_y.append(0)

# === Plotting ===
fig, ax = plt.subplots(figsize=(6, 6))
ax.set_aspect("equal")
ax.grid(True)

# Plot group-wise averaged measured points (centered)
ax.scatter(summary["x_measured_centered"], summary["y_measured_centered"],
           color="blue", label="Mean Measured per Spot", alpha=0.9, s=25)

# Plot red reference dots
ax.scatter(ref_x, ref_y, color="red", label="Reference Grid", s=12, zorder=3)

# Labels and limits
ax.set_title("Averaged Measured XY Positions with Reference Grid")
ax.set_xlabel("X offset (mm)")
ax.set_ylabel("Y offset (mm)")
ax.legend()

# Autoscale
all_x = np.concatenate([summary["x_measured_centered"], ref_x])
all_y = np.concatenate([summary["y_measured_centered"], ref_y])
x_pad = 0.5
y_pad = 0.5
ax.set_xlim([all_x.min() - x_pad, all_x.max() + x_pad])
ax.set_ylim([all_y.min() - y_pad, all_y.max() + y_pad])

plt.tight_layout()
plt.show()
