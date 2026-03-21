import pandas as pd
import matplotlib.pyplot as plt
from pathlib import Path
import numpy as np
import re

# =========================
# Config
# =========================
DATA_DIR = Path("/Users/margotferris/Desktop/repo/Brillouin_project/src/eye_position/z_data/Calibration2")

# Width of each box in x-units (mm)
BOX_WIDTH = 0.18

# =========================
# Helpers
# =========================
def load_z_mm(file: Path) -> np.ndarray:
    # Read as strings so we can normalize characters first
    s = pd.read_csv(file, header=None, dtype=str).iloc[:, 0]

    # Normalize unicode minus signs/dashes to ASCII '-'
    s = (
        s.astype(str)
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
      z_0
      z_1
      z_0.99998
      z_-1.25
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

parsed = []
for f in candidates:
    try:
        exp = parse_expected_from_stem(f.stem)
        parsed.append((exp, f))
    except ValueError:
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
all_relative_data = []
means = []
stds = []

for expected, file in zip(expected_positions, CSV_FILES):
    z_mm = load_z_mm(file)

    if z_mm.size == 0:
        print(f"[WARNING] Skipped {file.name}: no valid numeric data")
        all_relative_data.append(np.array([]))
        means.append(np.nan)
        stds.append(np.nan)
        continue

    # Zero each file relative to the mean of the reference file
    relative_z = z_mm - z0_mean

    all_relative_data.append(relative_z)
    means.append(np.mean(relative_z))
    stds.append(np.std(relative_z, ddof=0))

means = np.array(means, dtype=float)
stds = np.array(stds, dtype=float)

# =========================
# Compute error using means
# =========================
valid = ~np.isnan(means)
errors = np.abs(means[valid] - expected_positions[valid])

print("\nError between actual mean and expected positions:")
valid_files = [f for i, f in enumerate(CSV_FILES) if valid[i]]
valid_expected = expected_positions[valid]
valid_means = means[valid]

for i, (actual, expected, err, f) in enumerate(zip(valid_means, valid_expected, errors, valid_files)):
    print(
        f"Point {i+1:2d} (File {f.stem}): "
        f"Actual mean = {actual:.5f} mm, Expected = {expected:.5f} mm, Error = {err:.5f} mm"
    )

if errors.size > 0:
    print("\nSummary:")
    print(f"  Mean error: {np.mean(errors):.5f} mm")
    print(f"  Max error:  {np.max(errors):.5f} mm")
    print(f"  Std dev of error: {np.std(errors):.5f} mm")

# =========================
# Plot
# =========================
fig, ax = plt.subplots(figsize=(10, 6))

# Only plot non-empty datasets
plot_data = []
plot_positions = []

for pos, arr in zip(expected_positions, all_relative_data):
    if arr.size > 0:
        plot_data.append(arr)
        plot_positions.append(pos)

# Boxplots centered at expected positions
bp = ax.boxplot(
    plot_data,
    positions=plot_positions,
    widths=BOX_WIDTH,
    patch_artist=True,
    showfliers=True,     # outliers
    showmeans=False,     # we'll overlay mean ± std separately
    whis=1.5             # standard whisker rule
)

# Optional styling
for box in bp["boxes"]:
    box.set(facecolor="lightgray", edgecolor="black", linewidth=1)

for whisker in bp["whiskers"]:
    whisker.set(color="black", linewidth=1)

for cap in bp["caps"]:
    cap.set(color="black", linewidth=1)

for median in bp["medians"]:
    median.set(color="black", linewidth=1.5)

for flier in bp["fliers"]:
    flier.set(marker="o", markersize=4, markerfacecolor="red", markeredgecolor="red", alpha=0.7)

# Overlay mean ± standard deviation
ax.errorbar(
    expected_positions,
    means,
    yerr=stds,
    fmt="o",
    color="blue",
    ecolor="blue",
    elinewidth=1.2,
    capsize=4,
    markersize=5,
    label="Mean ± 1 SD"
)

# Reference line: perfect agreement
x_min = min(expected_positions) - 0.5
x_max = max(expected_positions) + 0.5
ax.plot([x_min, x_max], [x_min, x_max], linestyle="--", color="gray", linewidth=1, label="Ideal: measured = expected")

# Vertical guide lines at expected positions
for x in expected_positions:
    ax.axvline(x, linestyle="--", color="lightgray", linewidth=0.8, zorder=0)

# Formatting
ax.set_xlim(x_min, x_max)

# Make y limits based on data
all_vals = np.concatenate([arr for arr in all_relative_data if arr.size > 0])
pad = 0.2 if all_vals.size == 0 else max(0.2, 0.1 * (all_vals.max() - all_vals.min()))
ax.set_ylim(all_vals.min() - pad, all_vals.max() + pad)

ax.set_xlabel("Expected Z position (mm)")
ax.set_ylabel("Measured relative Z position (mm)")
ax.set_title("Paper Accuracy Test in the Axial Direction")
ax.legend()
ax.grid(True, axis="y", alpha=0.3)

plt.tight_layout()
plt.show()