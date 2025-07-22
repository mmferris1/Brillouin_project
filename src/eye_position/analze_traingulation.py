import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path

# === Automatically use the same folder the script is in ===
SCRIPT_DIR = Path(__file__).resolve().parent
FOLDER_PATH = SCRIPT_DIR  # CSVs are in the same folder

def load_all_triangulation_data(folder_path):
    csv_files = sorted(folder_path.glob("triangulation_results_z_step*.csv"))

    all_data = []
    for file in csv_files:
        try:
            df = pd.read_csv(file, header=None)
            df.columns = ["x", "y", "z_measured", "z_true"]
            all_data.append(df)
        except Exception as e:
            print(f"[WARNING] Skipped {file.name} due to error: {e}")

    if not all_data:
        raise RuntimeError("No valid data files found.")

    combined_df = pd.concat(all_data, ignore_index=True)
    return combined_df

def analyze_triangulation(dataframe, folder_path):
    # Convert to millimeters
    dataframe["z_measured_mm"] = dataframe["z_measured"] * 1000
    dataframe["z_true_mm"] = dataframe["z_true"] * 1000

    # Group and compute stats
    grouped = dataframe.groupby("z_true_mm")
    means = grouped["z_measured_mm"].mean().sort_index()
    stds = grouped["z_measured_mm"].std().sort_index()

    # Summary table
    summary_df = pd.DataFrame({
        "True Z (mm)": means.index,
        "Mean Measured Z (mm)": means.values,
        "Standard Deviation (mm)": stds.values
    })

    print("\nTriangulation Accuracy and Precision:\n")
    print(summary_df.to_string(index=False))

    # Plot
    plt.errorbar(means.index, means.values, yerr=stds.values, fmt='o', capsize=5)
    plt.xlabel("True Z Position (mm)")
    plt.ylabel("Measured Z (mm)")
    plt.title("Triangulation Accuracy and Precision")
    plt.grid(True)
    plt.tight_layout()
    plt.show()

    # Save summary
    output_file = folder_path / "triangulation_summary_mm.csv"
    summary_df.to_csv(output_file, index=False)
    print(f"\nSaved summary to: {output_file}")

    # ==== ΔZ analysis ====
    delta_z_measured = means.diff().dropna()
    expected_step = 0.1  # mm
    avg_step = delta_z_measured.mean()
    std_step = delta_z_measured.std()

    print("\nStep-wise ΔZ Analysis:")
    print(f"Expected ΔZ per step: {expected_step:.3f} mm")
    print(f"Average Measured ΔZ: {avg_step:.4f} mm")
    print(f"Standard Deviation of ΔZ: {std_step:.4f} mm")

    # Optional: save step deltas
    delta_output = pd.DataFrame({
        "Step Index": delta_z_measured.index,
        "ΔZ_measured (mm)": delta_z_measured.values
    })
    delta_output.to_csv(folder_path / "stepwise_deltas.csv", index=False)

# === Run when executed ===
if __name__ == "__main__":
    try:
        df_all = load_all_triangulation_data(FOLDER_PATH)
        analyze_triangulation(df_all, FOLDER_PATH)
    except Exception as e:
        print(f"[ERROR] {e}")
