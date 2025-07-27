import pandas as pd
import matplotlib.pyplot as plt
from pathlib import Path
import numpy as np
import re

# === Locate all z_*.csv files ===
SCRIPT_DIR = Path(__file__).resolve().parent
CSV_FILES = sorted(SCRIPT_DIR.glob("z_*.csv"), key=lambda f: int(re.findall(r'\d+', f.stem)[0]))

if not CSV_FILES:
    print("No 'z_*.csv' files found.")
    exit()

relative_means = []
stds_mm = []
expected_positions = []

# === Load z_0 mean for reference ===
df_0 = pd.read_csv(CSV_FILES[0], header=None)
z0_mm = df_0.iloc[:, 2] * 1000  # convert from meters to mm
z0_mean = z0_mm.mean()

# === Process each file ===
for file in CSV_FILES:
    try:
        df = pd.read_csv(file, header=None)
        if df.shape[1] < 3:
            raise ValueError("Expected at least 3 columns")

        z_mm = df.iloc[:, 2] * 1000  # convert to mm
        mean = z_mm.mean()
        std = z_mm.std()

        # Get expected position from filename
        match = re.search(r"z_(\d+)", file.stem)
        if not match:
            raise ValueError(f"Invalid filename format: {file.name}")
        expected = int(match.group(1))

        relative_means.append(abs(mean - z0_mean))
        stds_mm.append(std*2)
        expected_positions.append(expected)

    except Exception as e:
        print(f"[WARNING] Skipped {file.name}: {e}")

# === Compute error ===
errors = np.abs(np.array(relative_means) - np.array(expected_positions))

# Print errors
print("\nError between actual and expected positions:")
for i, (actual, expected, err) in enumerate(zip(relative_means, expected_positions, errors)):
    print(f"Point {i+1:2d} (File z_{expected}): Actual = {actual:.3f} mm, Expected = {expected} mm, Error = {err:.3f} mm")

# Summary stats
print("\nSummary:")
print(f"  Mean error: {np.mean(errors):.3f} mm")
print(f"  Max error:  {np.max(errors):.3f} mm")
print(f"  Std dev:    {np.std(errors):.3f} mm")

# === Compute error per mm ===
errors = np.abs(np.array(relative_means) - np.array(expected_positions))

# Avoid division by zero (exclude z_0)
nonzero_expected = np.array(expected_positions) > 0
errors_nonzero = errors[nonzero_expected]
distances_nonzero = np.array(expected_positions)[nonzero_expected]
error_per_mm = errors_nonzero / distances_nonzero

print("\nError per mm of expected travel:")
for i, (e, d) in enumerate(zip(errors_nonzero, distances_nonzero), start=1):
    print(f"Step to {d} mm: Error = {e:.3f} mm → Error per mm = {e/d:.4f} mm/mm")

print(f"\nAverage error per mm: {np.mean(error_per_mm):.4f} mm/mm")


# === Plot ===
fig, ax = plt.subplots(figsize=(10, 2))
y = [0] * len(relative_means)

ax.errorbar(relative_means, y, xerr=stds_mm, fmt='o', color='blue', ecolor='black',
            elinewidth=1, capsize=4, alpha=1)

# === Dashed vertical lines at expected positions ===
x_min = np.floor(min(relative_means)) - 0.5
x_max = np.ceil(max(relative_means)) - 0.5
for x in range(min(expected_positions), max(expected_positions) + 3):
    ax.axvline(x, linestyle='--', color='lightgray', linewidth=0.8, zorder=0)

# === Formatting ===
ax.set_yticks([])
ax.set_xlabel("Z Position Relative to 0 (mm)")
ax.set_title("Relative Z location ")
ax.set_xlim(x_min, x_max)
ax.set_xticks(np.arange(np.floor(x_min), np.ceil(x_max) + 1, 1))  # whole numbers

plt.tight_layout()
plt.show()
