import pandas as pd
import matplotlib.pyplot as plt
from pathlib import Path
import numpy as np
import re

# =========================
# Config
# =========================
DATA_DIR = Path("/Users/margotferris/Desktop/repo/Brillouin_project/src/eye_position/z_data/RealEye3")

# =========================
# Helpers
# =========================
def _normalize_minus(s: pd.Series) -> pd.Series:
    return (s.astype(str)
              .str.strip()
              .str.replace("−", "-", regex=False)  # Unicode minus
              .str.replace("–", "-", regex=False)  # en-dash
              .str.replace("—", "-", regex=False)  # em-dash
           )

def load_z_samples_mm(file: Path) -> np.ndarray:
    """
    Load all valid Z samples from column 1 of the CSV.
    Assumes each file contains multiple raw Z measurements.
    """
    df = pd.read_csv(file, header=None, dtype=str)

    if df.shape[1] < 1:
        raise ValueError("Expected at least 1 column of Z values")

    s = _normalize_minus(df.iloc[:, 0])
    z = pd.to_numeric(s, errors="coerce").dropna().to_numpy(dtype=float)

    if z.size == 0:
        raise ValueError(f"No valid Z values found in {file.name}")

    return z

def load_mean_std_mm(file: Path) -> tuple[float, float, int]:
    """
    Compute mean and std from all raw Z samples in the file.
    Returns: mean, std, n
    """
    z = load_z_samples_mm(file)

    mean = float(np.mean(z))
    std = float(np.std(z, ddof=1)) if z.size > 1 else 0.0

    return mean, std, z.size

def parse_expected_from_stem(stem: str) -> float:
    stem = stem.replace("−", "-").replace("–", "-").replace("—", "-")
    m = re.fullmatch(r"z_(-?(?:\d+(?:\.\d+)?|\.\d+))", stem)
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

z0_mean, z0_std, z0_n = load_mean_std_mm(ref_file)
print(
    f"\nReference file: {ref_file.name} "
    f"(expected={expected_positions[ref_idx]:.5f}), "
    f"mean={z0_mean:.6f} mm, std={z0_std:.6f} mm, n={z0_n}"
)

# =========================
# Process each file
# =========================
relative_means = []
stds_mm = []
kept_expected = []
kept_files = []
sample_counts = []
all_relative_samples = []

for expected, file in zip(expected_positions, CSV_FILES):
    try:
        z_samples = load_z_samples_mm(file)
        mean = float(np.mean(z_samples))
        std = float(np.std(z_samples, ddof=1)) if z_samples.size > 1 else 0.0
        n = z_samples.size
    except Exception as e:
        print(f"[WARNING] Skipped {file.name}: {e}")
        continue

    relative_samples = z_samples - z0_mean

    relative_means.append(mean - z0_mean)
    stds_mm.append(std * 2)   # keeping your convention
    kept_expected.append(expected)
    kept_files.append(file)
    sample_counts.append(n)
    all_relative_samples.append(np.abs(relative_samples))

relative_means = np.array(relative_means, dtype=float)
stds_mm = np.array(stds_mm, dtype=float)
kept_expected = np.array(kept_expected, dtype=float)
sample_counts = np.array(sample_counts, dtype=int)

# =========================
# Compute error
# =========================
errors = np.abs(relative_means - kept_expected)

print("\nError between actual and expected positions:")
for i, (actual, expected, err, f, std_mm, n) in enumerate(
    zip(relative_means, kept_expected, errors, kept_files, stds_mm, sample_counts)
):
    print(
        f"Point {i+1:2d} (File {f.stem}): "
        f"Actual = {actual:.5f} mm, "
        f"Expected = {expected:.5f} mm, "
        f"Error = {err:.5f} mm, "
        f"Std(2x) = {std_mm:.5f} mm, "
        f"n = {n}"
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

# --- REMOVE THIS BLOCK ---
# ax.errorbar(
#     np.abs(relative_means), y,
#     xerr=stds_mm,
#     fmt="o",
#     ecolor="black",
#     elinewidth=1,
#     capsize=4,
#     alpha=1,
#     zorder=4
# )

# plot each individual data point
rng = np.random.default_rng(0)
for samples in all_relative_samples:
    y_jitter = rng.uniform(-0.035, 0.035, size=len(samples))
    ax.scatter(
        samples,
        y_jitter,
        s=12,
        color="black",
        alpha=0.75,
        zorder=2
    )

# boxplots (NOW BLUE)
ax.boxplot(
    all_relative_samples,
    vert=False,
    positions=np.zeros(len(all_relative_samples)),
    widths=0.10,
    patch_artist=True,
    showfliers=True,
    boxprops=dict(facecolor="blue", edgecolor="black", alpha=0.4),
    medianprops=dict(color="black", linewidth=1.2),
    whiskerprops=dict(color="black", linewidth=1),
    capprops=dict(color="black", linewidth=1),
    flierprops=dict(marker='o', markersize=3, markerfacecolor='black', markeredgecolor='black', alpha=0.6)
)

# Vertical grid lines at expected positions
for x in np.abs(kept_expected):
    ax.axvline(
        x,
        linestyle="--",
        color="lightgray",
        linewidth=0.8,
        zorder=0
    )

x_min = -.5
x_max = float(np.max(np.abs(kept_expected))) + 0.5
ax.set_xlim(x_min, x_max)

ax.set_ylim(-0.15, 0.15)
ax.set_yticks([])
ax.set_xlabel("Z distance (mm)")
ax.set_title("Human Eye Accuracy Test in the Axial Direction")

plt.tight_layout()
plt.show()