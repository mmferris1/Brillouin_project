import pandas as pd
import matplotlib.pyplot as plt
from pathlib import Path
import numpy as np
from scipy.spatial import cKDTree
import matplotlib.image as mpimg

# === Load and plot background image ===
img_path = Path("/Users/margaretferris/Desktop/grayscale_eye_cropped_to_bounding_box.png")
eye_img = mpimg.imread(img_path)

# === Control these parameters ===
image_opacity = 0.4             # 0 = transparent, 1 = opaque
image_width_mm = 19             # Desired image width in mm
x_shift_mm = 1.6  # shift image 1 mm to the left (adjust as needed)


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
SCRIPT_DIR = Path(__file__).resolve().parent
CSV_FILES = list(SCRIPT_DIR.glob("xy_*.csv"))

if not CSV_FILES:
    print("No CSV files found.")
    exit()

mean_points = []

# === Load each file and compute its mean measured point ===
for file in CSV_FILES:
    try:
        df = pd.read_csv(file, header=None)
        if df.shape[1] != 3:
            raise ValueError("Expected 5 columns")
        df.columns = ["x_measured", "y_measured", "z"]

        # Convert from meters to millimeters
        df[["x_measured", "y_measured"]] *= 1000

        # Mean and standard deviation of measured points
        x_mean = df["x_measured"].mean()
        y_mean = df["y_measured"].mean()
        x_std = df["x_measured"].std()
        y_std = df["y_measured"].std()

        mean_points.append((x_mean, y_mean, x_std, y_std))
    except Exception as e:
        print(f"[WARNING] Skipped {file.name}: {e}")


# === Compute global center to shift to (0, 0) ===
all_x, all_y, all_x_std, all_y_std = zip(*mean_points)
x_ref = all_x[0]  # First point’s measured x
y_ref = all_y[0]  # First point’s measured y

x_shifted = [0]  # First point stays at origin
y_shifted = [0]

# === Center measured points around the mean ===
x_shifted = [x - np.mean(all_x) for x in all_x]
y_shifted = [y - np.mean(all_y) for y in all_y]


x_stds = list(all_x_std)
y_stds = list(all_y_std)


# === Create reference dots at r = 1, 2, 3, 4 mm in 8 directions + origin ===
radii = [1, 2, 3, 4]  # in mm
angles_deg = [0, 45, 90, 135, 180, 225, 270, 315]
angles_rad = np.deg2rad(angles_deg)

ref_x = [0]  # Include origin
ref_y = [0]
for r in radii:
    for angle in angles_rad:
        ref_x.append(r * np.cos(angle))
        ref_y.append(r * np.sin(angle))


# Assume x_shifted, y_shifted, ref_x, ref_y are defined as in your current script

fig, ax = plt.subplots(figsize=(6, 6))
ax.set_aspect("equal")

# === Plot measured points and reference dots ===
scatter_measured = ax.scatter(x_shifted, y_shifted, color="blue", label="Mean Measured Points", s=10)
scatter_reference = ax.scatter(ref_x, ref_y, color="red", label="Reference Grid", s=5, zorder=3)



for x, y, x_err, y_err in zip(x_shifted, y_shifted, x_stds, y_stds):
    ax.errorbar(x, y, xerr=2*x_err, yerr=2*y_err, fmt='none', ecolor='black', alpha=0.6, capsize=1, zorder=4)

# Add legend
ax.legend(loc="upper right", frameon=True)

# === Draw axis lines at x=0 and y=0 ===
ax.axhline(0, color='black', linewidth=1)
ax.axvline(0, color='black', linewidth=1)

# Convert to numpy arrays
measured_points = np.column_stack((x_shifted, y_shifted))    # blue
reference_points = np.column_stack((ref_x, ref_y))           # red

# Build KD-tree for fast nearest-neighbor search
ref_tree = cKDTree(reference_points)

# For each measured point, find the closest reference point
distances, indices = ref_tree.query(measured_points)

# Now 'distances' holds the error for each measured point
# 'indices' gives which reference point was closest

# Print individual errors
print("\nNearest-neighbor errors between measured and reference points:")
for i, (d, idx) in enumerate(zip(distances, indices)):
    ref_pt = reference_points[idx]
    meas_pt = measured_points[i]
    print(f"Point {i+1:2d}: Error = {d:.4f} mm | Measured = {meas_pt}, Closest Ref = {ref_pt}")

# Summary
print("\nSummary statistics:")
print(f"  Mean error:     {np.mean(distances):.4f} mm")
print(f"  Max error:      {np.max(distances):.4f} mm")
print(f"  Std deviation:  {np.std(distances):.4f} mm")

# === Mean error per radial mm ===
# Get radius of each matched reference point
ref_radii = np.linalg.norm(reference_points[indices], axis=1)

# Avoid division by zero for r=0 (we'll exclude it from this analysis)
nonzero = ref_radii > 0
normalized_errors = distances[nonzero] / ref_radii[nonzero]

mean_relative_error = np.mean(normalized_errors)

print(f"\nThe error for each mm is approximately {mean_relative_error:.4f} mm/mm")

# === Compute per-radius mean error ===
# First get the radius of each reference point
ref_radii = np.linalg.norm(reference_points, axis=1)

# Dictionary to accumulate errors per rounded radius (to nearest integer)
radius_errors = {r: [] for r in [0, 1, 2, 3, 4]}

for d, idx in zip(distances, indices):
    r = round(ref_radii[idx])  # round to nearest integer mm
    if r in radius_errors:
        radius_errors[r].append(d)

# Print average error per radius
print("\nError by reference radius:")
for r in sorted(radius_errors):
    errs = radius_errors[r]
    if errs:
        mean_err = np.mean(errs)
        print(f"  Radius {r} mm: Error = {mean_err:.4f} mm")
    else:
        print(f"  Radius {r} mm: No matched points")


# === Draw concentric circles at r = 1, 2, 3, 4 mm ===
for r in [1, 2, 3, 4]:
    circle = plt.Circle((0, 0), r, color='gray', fill=False, linestyle='--', linewidth=0.8)
    ax.add_artist(circle)

# === Set axis limits ===
max_radius = 4.5
ax.set_xlim(-max_radius, max_radius)
ax.set_ylim(-max_radius, max_radius)

# === Display the image underneath ===
ax.imshow(eye_img, extent=img_extent, alpha=image_opacity, zorder=0)

# === Axis formatting ===
for spine in ['top', 'right']:
    ax.spines[spine].set_visible(False)

ax.spines['left'].set_position(('outward', 10))
ax.spines['bottom'].set_position(('outward', 10))

ax.tick_params(axis='both', direction='out', length=4)
ax.set_xlabel("X distance (mm)", fontsize=14, labelpad=10)
ax.set_ylabel("Y distance (mm)", fontsize=14, labelpad=10)

ax.set_title(" X-Y Location", fontsize=20)

ax.grid(False)

plt.tight_layout()
plt.show()
