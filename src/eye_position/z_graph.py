import pandas as pd
import matplotlib.pyplot as plt
from pathlib import Path
import numpy as np
import re

# =========================
# Config
# =========================
DATA_DIR = Path("/Users/margotferris/Desktop/repo/Brillouin_project/src/eye_position/z_data/Calibration2")

# =========================
# Helpers
# =========================
def load_z_mm(file: Path) -> np.ndarray:
    # Read as strings so we can normalize characters first
    s = pd.read_csv(file, header=None, dtype=str).iloc[:, 0]

    # Normalize unicode minus signs/dashes to ASCII '-'
    s = (s.astype(str)
           .str.strip()
           .str.replace("−", "-", regex=False)  # U+2212
           .str.replace("–", "-", regex=False)  # en-dash
           .str.replace("—", "-", regex=False)  # em-dash
    )

    z = pd.to_numeric(s, errors="coerce").dropna().to_numpy(dtype=float)
    return z


def parse_expected_from_stem(stem: str) -> float:
    """
    Parse expected position from filename stem like:
      z_0.csv
      z_1.csv
      z_0.csv.99998.csv
      z_-1.csv.25
    """
    m = re.fullmatch(r"z_(-?\d+(?:\.\d+)?)", stem)
    if not m:
        raise ValueError(f"Invalid filename format (expected z_<number>): {stem}")
    return float(m.group(1))

# =========================
# Locate files
# =========================
candidates = list(DATA_DIR.glob("z_*.csv"))

if not candidates:
    print(f"No 'z_*.csv' files found in: {DATA_DIR}")
    raise SystemExit(0)

# Keep only files that match z_<number> exactly, and sort by that number (float)
parsed = []
for f in candidates:
    try:
        exp = parse_expected_from_stem(f.stem)
        parsed.append((exp, f))
    except ValueError:
        # Skip non-matching names like z_0_copy.csv, etc.
        pass

if not parsed:
    print(f"No valid files named exactly like z_<number>.csv found in: {DATA_DIR}")
    raise SystemExit(1)

parsed.sort(key=lambda t: t[0])
expected_positions = np.array([t[0] for t in parsed], dtype=float)
CSV_FILES = [t[1] for t in parsed]

print("\nUsing files (expected -> filename):")
for e, f in zip(expected_positions, CSV_FILES):
    print(f"  {e:.5f} -> {f.name}")

# =========================
# Choose reference (expected closest to 0)
# =========================
ref_idx = int(np.argmin(np.abs(expected_positions - 0.0)))
ref_file = CSV_FILES[ref_idx]

z0_mm = load_z_mm(ref_file)
if z0_mm.size == 0:
    print(f"Reference file {ref_file.name} contains no valid numeric data.")
    raise SystemExit(1)

z0_mean = z0_mm.mean()
print(f"\nReference file: {ref_file.name} (expected={expected_positions[ref_idx]:.5f}), mean={z0_mean:.6f} mm")

# =========================
# Process each file
# =========================
relative_means = []
stds_mm = []

for expected, file in zip(expected_positions, CSV_FILES):
    z_mm = load_z_mm(file)
    if z_mm.size == 0:
        print(f"[WARNING] Skipped {file.name}: no valid numeric data")
        continue

    mean = z_mm.mean()
    std = z_mm.std()

    relative_means.append(mean - z0_mean)
    stds_mm.append(std * 2)  # your convention

relative_means = np.array(relative_means, dtype=float)
stds_mm = np.array(stds_mm, dtype=float)

# =========================
# Compute error
# =========================
errors = np.abs(relative_means - expected_positions)

print("\nError between actual and expected positions:")
for i, (actual, expected, err) in enumerate(zip(relative_means, expected_positions, errors)):
    print(
        f"Point {i+1:2d} (File {CSV_FILES[i].stem}): "
        f"Actual = {actual:.5f} mm, Expected = {expected:.5f} mm, Error = {err:.5f} mm"
    )

print("\nSummary:")
print(f"  Mean error: {np.mean(errors):.5f} mm")
print(f"  Max error:  {np.max(errors):.5f} mm")
print(f"  Std dev:    {np.std(errors):.5f} mm")

# =========================
# Error per mm (exclude expected ~ 0)
# =========================
nonzero = np.abs(expected_positions) > 1e-12
errors_nonzero = errors[nonzero]
dist_nonzero = np.abs(expected_positions[nonzero])

if errors_nonzero.size > 0:
    error_per_mm = errors_nonzero / dist_nonzero
    print("\nError per mm of expected travel:")
    for e, d in zip(errors_nonzero, dist_nonzero):
        print(f"Step to {d:.5f} mm: Error = {e:.5f} mm → Error per mm = {e/d:.5f} mm/mm")
    print(f"\nAverage error per mm: {np.mean(error_per_mm):.5f} mm/mm")

# =========================
# Plot
# =========================
fig, ax = plt.subplots(figsize=(10, 2))
y = np.zeros_like(relative_means)

ax.errorbar(
    relative_means, y,
    xerr=stds_mm,
    fmt="o",
    ecolor="black",
    elinewidth=1,
    capsize=4,
    alpha=1
)

# === Vertical grid lines at expected positions ===
for x in expected_positions:
    ax.axvline(
        x,
        linestyle="--",
        color="lightgray",
        linewidth=0.8,
        zorder=0
    )

# Formatting
x_min = min(expected_positions) - 0.5
x_max = max(expected_positions) + 0.5
ax.set_xlim(x_min, x_max)

ax.set_yticks([])
ax.set_xlabel("Z distance (mm)")
ax.set_title("Paper Accuracy Test in the Axial Direction")

plt.tight_layout()
plt.show()

