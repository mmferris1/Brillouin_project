import pandas as pd
import matplotlib.pyplot as plt
from pathlib import Path
import numpy as np
import re

# =========================
# Config
# =========================
DATA_DIR = Path("/Users/margotferris/Desktop/repo/Brillouin_project/src/eye_position/z_data/RealEye1")

# =========================
# Helpers
# =========================
def _normalize_minus(s: pd.Series) -> pd.Series:
    return (s.astype(str)
              .str.strip()
              .str.replace("−", "-", regex=False)  # U+2212
              .str.replace("–", "-", regex=False)  # en-dash
              .str.replace("—", "-", regex=False)  # em-dash
           )

def load_z_mm(file: Path) -> np.ndarray:
    """If your z files are raw samples in column 1, this still works."""
    s = pd.read_csv(file, header=None, dtype=str).iloc[:, 0]
    s = _normalize_minus(s)
    z = pd.to_numeric(s, errors="coerce").dropna().to_numpy(dtype=float)
    return z

def load_mean_std_mm(file: Path) -> tuple[float, float]:
    """
    NEW: If your z files are summary rows:
      col 1 (index 0) = mean
      col 2 (index 1) = std
    """
    df = pd.read_csv(file, header=None, dtype=str)
    if df.shape[1] < 2:
        raise ValueError("Expected at least 2 columns: mean (col1), std (col2)")

    mean_s = _normalize_minus(df.iloc[:, 0]).iloc[0]
    std_s  = _normalize_minus(df.iloc[:, 1]).iloc[0]

    mean = float(pd.to_numeric(mean_s, errors="coerce"))
    std  = float(pd.to_numeric(std_s,  errors="coerce"))

    if np.isnan(mean) or np.isnan(std):
        raise ValueError(f"Could not parse mean/std from first row in {file.name}")

    return mean, std

def parse_expected_from_stem(stem: str) -> float:
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

# NEW: reference uses mean from col1
z0_mean, z0_std = load_mean_std_mm(ref_file)
print(f"\nReference file: {ref_file.name} (expected={expected_positions[ref_idx]:.5f}), mean={z0_mean:.6f} mm")

# =========================
# Process each file (mean from col1, std from col2)
# =========================
relative_means = []
stds_mm = []
kept_expected = []
kept_files = []

for expected, file in zip(expected_positions, CSV_FILES):
    try:
        mean, std = load_mean_std_mm(file)
    except Exception as e:
        print(f"[WARNING] Skipped {file.name}: {e}")
        continue

    relative_means.append(mean - z0_mean)
    stds_mm.append(std * 2)          # your convention
    kept_expected.append(expected)
    kept_files.append(file)

relative_means = np.array(relative_means, dtype=float)
stds_mm = np.array(stds_mm, dtype=float)
kept_expected = np.array(kept_expected, dtype=float)

# =========================
# Compute error
# =========================
errors = np.abs(relative_means - kept_expected)

print("\nError between actual and expected positions:")
for i, (actual, expected, err, f) in enumerate(zip(relative_means, kept_expected, errors, kept_files)):
    print(
        f"Point {i+1:2d} (File {f.stem}): "
        f"Actual = {actual:.5f} mm, Expected = {expected:.5f} mm, Error = {err:.5f} mm"
    )

print("\nSummary:")
print(f"  Mean error: {np.mean(errors):.5f} mm")
print(f"  Max error:  {np.max(errors):.5f} mm")
print(f"  Std dev:    {np.std(errors):.5f} mm")

# =========================
# Error per mm (exclude expected ~ 0)
# =========================
nonzero = np.abs(kept_expected) > 1e-12
errors_nonzero = errors[nonzero]
dist_nonzero = np.abs(kept_expected[nonzero])

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

# Vertical grid lines at expected positions
for x in kept_expected:
    ax.axvline(
        x,
        linestyle="--",
        color="lightgray",
        linewidth=0.8,
        zorder=0
    )

x_min = float(np.min(kept_expected)) - 0.5
x_max = float(np.max(kept_expected)) + 0.5
ax.set_xlim(x_min, x_max)

ax.set_yticks([])
ax.set_xlabel("Z distance (mm)")
ax.set_title("Human Eye Accuracy Test in the Axial Direction")

plt.tight_layout()
plt.show()
