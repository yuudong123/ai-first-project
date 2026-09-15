"""Compare Raw NORMAL data and Generator V1 through V5 under one method."""

from pathlib import Path

import matplotlib
import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler


matplotlib.use("Agg")
import matplotlib.pyplot as plt


PROJECT_ROOT = Path(__file__).resolve().parents[2]
DATA_DIR = PROJECT_ROOT / "data" / "processed" / "simulator"
GRAPH_DIR = DATA_DIR / "graphs"
RAW_FILE = DATA_DIR / "uci_1hz_17sensors.npz"
MODEL_FILES = {
    version: DATA_DIR / f"generated_300s_{version.lower()}.csv"
    for version in ["V1", "V2", "V3", "V4", "V5"]
}

SUMMARY_FILE = DATA_DIR / "generator_v1_v2_v3_v4_v5_summary.csv"
SENSOR_SUMMARY_FILE = (
    DATA_DIR / "generator_v1_v2_v3_v4_v5_sensor_comparison.csv"
)
LINE_FILE = GRAPH_DIR / "raw_v1_v2_v3_v4_v5_ts1_300s.png"
MEAN_FILE = GRAPH_DIR / "v1_v2_v3_v4_v5_mean_error.png"
STD_FILE = GRAPH_DIR / "v1_v2_v3_v4_v5_std_error.png"
HIST_FILE = GRAPH_DIR / "raw_v1_v2_v3_v4_v5_ts1_histogram.png"
BOX_FILE = GRAPH_DIR / "raw_v1_v2_v3_v4_v5_ts1_boxplot.png"
REPRESENTATIVE_FILE = GRAPH_DIR / "raw_v5_representative_sensors_300s.png"

TRAIN_RATIO = 0.8
GENERATED_SECONDS = 300
RAW_REFERENCE_RECORDS = 5
WINDOW_BY_MODEL = {"V1": 20, "V2": 30, "V3": 30, "V4": 30, "V5": 30}
REPRESENTATIVE_SENSORS = ["PS1", "FS1", "TS1", "VS1"]


def shift_record_phase(records, window_size):
    shifted = np.concatenate(
        [records[:, window_size:, :], records[:, :window_size, :]], axis=1
    )
    return shifted.reshape(GENERATED_SECONDS, records.shape[2])


def main():
    GRAPH_DIR.mkdir(parents=True, exist_ok=True)
    with np.load(RAW_FILE, allow_pickle=False) as npz_file:
        raw_data = npz_file["data"].astype(np.float64)
        sensor_names = [str(name) for name in npz_file["sensor_names"]]
    validation_start = int(raw_data.shape[0] * TRAIN_RATIO)
    normal_records = raw_data[
        validation_start : validation_start + RAW_REFERENCE_RECORDS
    ]
    raw_normal = normal_records.reshape(GENERATED_SECONDS, len(sensor_names))
    raw_all = raw_data.reshape(-1, len(sensor_names))

    generated = {}
    for version, file_path in MODEL_FILES.items():
        frame = pd.read_csv(file_path)
        values = frame[sensor_names].to_numpy(dtype=np.float64)
        if values.shape != (GENERATED_SECONDS, len(sensor_names)):
            raise ValueError(f"{version} shape error: {values.shape}")
        if not np.isfinite(values).all():
            raise ValueError(f"{version} contains NaN or Inf")
        generated[version] = values

    global_mean = raw_all.mean(axis=0)
    global_std = raw_all.std(axis=0)
    global_min = raw_all.min(axis=0)
    global_max = raw_all.max(axis=0)
    normal_mean = raw_normal.mean(axis=0)
    normal_std = raw_normal.std(axis=0)
    normal_min = raw_normal.min(axis=0)
    normal_max = raw_normal.max(axis=0)

    comparison_rows = []
    sensor_rows = []
    for version, values in generated.items():
        aligned_raw = shift_record_phase(
            normal_records, WINDOW_BY_MODEL[version]
        )
        scaler = StandardScaler()
        aligned_raw_scaled = scaler.fit_transform(aligned_raw)
        generated_scaled = scaler.transform(values)

        model_mean = values.mean(axis=0)
        model_std = values.std(axis=0)
        model_min = values.min(axis=0)
        model_max = values.max(axis=0)
        global_mean_error = np.abs(model_mean - global_mean) / np.maximum(
            np.abs(global_mean), 1e-8
        ) * 100.0
        global_std_error = np.abs(model_std - global_std) / np.maximum(
            global_std, 1e-8
        ) * 100.0
        normal_mean_error = np.abs(model_mean - normal_mean) / np.maximum(
            np.abs(normal_mean), 1e-8
        ) * 100.0
        normal_std_error = np.abs(model_std - normal_std) / np.maximum(
            normal_std, 1e-8
        ) * 100.0
        global_range_pass = (model_min >= global_min) & (
            model_max <= global_max
        )
        normal_range_pass = (model_min >= normal_min) & (
            model_max <= normal_max
        )
        first_cycle_mean = values[:60].mean(axis=0)
        previous_cycle_mean = values[-120:-60].mean(axis=0)
        last_cycle_mean = values[-60:].mean(axis=0)

        comparison_rows.append(
            {
                "model": version,
                "raw_all_mean_error_pct": global_mean_error.mean(),
                "raw_all_std_error_pct": global_std_error.mean(),
                "raw_normal_mean_error_pct": normal_mean_error.mean(),
                "raw_normal_std_error_pct": normal_std_error.mean(),
                "raw_aligned_mse": np.mean((values - aligned_raw) ** 2),
                "raw_aligned_mae": np.mean(np.abs(values - aligned_raw)),
                "standardized_mse": np.mean(
                    (generated_scaled - aligned_raw_scaled) ** 2
                ),
                "standardized_mae": np.mean(
                    np.abs(generated_scaled - aligned_raw_scaled)
                ),
                "raw_all_range_pass_count": int(global_range_pass.sum()),
                "raw_normal_range_pass_count": int(normal_range_pass.sum()),
                "mean_abs_first_to_last_cycle_change": np.mean(
                    np.abs(last_cycle_mean - first_cycle_mean)
                ),
                "mean_abs_last_two_cycle_change": np.mean(
                    np.abs(last_cycle_mean - previous_cycle_mean)
                ),
            }
        )

        for sensor_index, sensor in enumerate(sensor_names):
            sensor_rows.append(
                {
                    "dataset": version,
                    "sensor": sensor,
                    "mean": model_mean[sensor_index],
                    "std": model_std[sensor_index],
                    "min": model_min[sensor_index],
                    "max": model_max[sensor_index],
                    "raw_normal_mean": normal_mean[sensor_index],
                    "raw_normal_std": normal_std[sensor_index],
                    "raw_normal_mean_error_pct": normal_mean_error[
                        sensor_index
                    ],
                    "raw_normal_std_error_pct": normal_std_error[
                        sensor_index
                    ],
                    "raw_all_range_pass": bool(global_range_pass[sensor_index]),
                }
            )

    for label, values in [("RAW_ALL", raw_all), ("RAW_NORMAL", raw_normal)]:
        for sensor_index, sensor in enumerate(sensor_names):
            sensor_rows.append(
                {
                    "dataset": label,
                    "sensor": sensor,
                    "mean": values[:, sensor_index].mean(),
                    "std": values[:, sensor_index].std(),
                    "min": values[:, sensor_index].min(),
                    "max": values[:, sensor_index].max(),
                    "raw_normal_mean": normal_mean[sensor_index],
                    "raw_normal_std": normal_std[sensor_index],
                    "raw_normal_mean_error_pct": (
                        np.abs(values[:, sensor_index].mean() - normal_mean[sensor_index])
                        / max(np.abs(normal_mean[sensor_index]), 1e-8)
                        * 100.0
                    ),
                    "raw_normal_std_error_pct": (
                        np.abs(values[:, sensor_index].std() - normal_std[sensor_index])
                        / max(normal_std[sensor_index], 1e-8)
                        * 100.0
                    ),
                    "raw_all_range_pass": True,
                }
            )

    summary = pd.DataFrame(comparison_rows)
    sensor_summary = pd.DataFrame(sensor_rows)
    summary.to_csv(SUMMARY_FILE, index=False, float_format="%.10f")
    sensor_summary.to_csv(
        SENSOR_SUMMARY_FILE, index=False, float_format="%.10f"
    )

    seconds = np.arange(1, GENERATED_SECONDS + 1)
    ts1_index = sensor_names.index("TS1")
    plt.figure(figsize=(16, 7))
    plt.plot(seconds, raw_normal[:, ts1_index], label="Raw NORMAL", linewidth=2)
    for version, values in generated.items():
        plt.plot(seconds, values[:, ts1_index], label=version, alpha=0.8)
    for boundary in [60, 120, 180, 240]:
        plt.axvline(boundary, linestyle="--", color="gray", alpha=0.3)
    plt.title("TS1 300 Seconds: Raw NORMAL vs V1-V5")
    plt.xlabel("Second")
    plt.ylabel("TS1")
    plt.grid(True, alpha=0.3)
    plt.legend(ncol=3)
    plt.tight_layout()
    plt.savefig(LINE_FILE, dpi=150)
    plt.close()

    x_positions = np.arange(len(summary))
    for column, ylabel, title, path in [
        (
            "raw_normal_mean_error_pct",
            "Mean error (%)",
            "Mean Error against Seed-matched Raw NORMAL",
            MEAN_FILE,
        ),
        (
            "raw_normal_std_error_pct",
            "Standard deviation error (%)",
            "Std Error against Seed-matched Raw NORMAL",
            STD_FILE,
        ),
    ]:
        plt.figure(figsize=(9, 6))
        plt.bar(x_positions, summary[column], color="tab:blue")
        plt.xticks(x_positions, summary["model"])
        plt.ylabel(ylabel)
        plt.title(title)
        plt.grid(True, axis="y", alpha=0.3)
        plt.tight_layout()
        plt.savefig(path, dpi=150)
        plt.close()

    plt.figure(figsize=(13, 7))
    plt.hist(raw_normal[:, ts1_index], bins=20, alpha=0.45, label="Raw NORMAL")
    for version, values in generated.items():
        plt.hist(values[:, ts1_index], bins=20, alpha=0.35, label=version)
    plt.title("TS1 Histogram: Raw NORMAL vs V1-V5")
    plt.xlabel("TS1")
    plt.ylabel("Frequency")
    plt.grid(True, axis="y", alpha=0.3)
    plt.legend(ncol=3)
    plt.tight_layout()
    plt.savefig(HIST_FILE, dpi=150)
    plt.close()

    box_values = [raw_normal[:, ts1_index]] + [
        values[:, ts1_index] for values in generated.values()
    ]
    plt.figure(figsize=(10, 7))
    plt.boxplot(box_values, tick_labels=["Raw"] + list(generated.keys()))
    plt.title("TS1 Box Plot: Raw NORMAL vs V1-V5")
    plt.ylabel("TS1")
    plt.grid(True, axis="y", alpha=0.3)
    plt.tight_layout()
    plt.savefig(BOX_FILE, dpi=150)
    plt.close()

    figure, axes = plt.subplots(2, 2, figsize=(14, 9), sharex=True)
    for axis, sensor in zip(axes.flat, REPRESENTATIVE_SENSORS):
        sensor_index = sensor_names.index(sensor)
        axis.plot(seconds, raw_normal[:, sensor_index], label="Raw NORMAL")
        axis.plot(seconds, generated["V5"][:, sensor_index], label="V5")
        for boundary in [60, 120, 180, 240]:
            axis.axvline(boundary, linestyle="--", color="gray", alpha=0.25)
        axis.set_title(sensor)
        axis.set_xlabel("Second")
        axis.set_ylabel(sensor)
        axis.grid(True, alpha=0.3)
        axis.legend()
    figure.suptitle("Representative Sensors: Raw NORMAL vs V5")
    figure.tight_layout()
    figure.savefig(REPRESENTATIVE_FILE, dpi=150)
    plt.close(figure)

    print("=" * 110)
    print("V1-V5 COMPARISON (Raw NORMAL is the primary reference)")
    print("=" * 110)
    print(summary.to_string(index=False))
    ts1_rows = sensor_summary[
        sensor_summary["sensor"].eq("TS1")
        & sensor_summary["dataset"].isin(["RAW_NORMAL", "V1", "V2", "V3", "V4", "V5"])
    ]
    print("\n[TS1 MEAN / STD / MIN / MAX]")
    print(ts1_rows[["dataset", "mean", "std", "min", "max"]].to_string(index=False))
    for output_path in [
        SUMMARY_FILE,
        SENSOR_SUMMARY_FILE,
        LINE_FILE,
        MEAN_FILE,
        STD_FILE,
        HIST_FILE,
        BOX_FILE,
        REPRESENTATIVE_FILE,
    ]:
        print(f"[SAVED] {output_path}")


if __name__ == "__main__":
    main()
