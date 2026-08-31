"""Analyze 60-second position patterns in the UCI Raw 1 Hz sensor data.

This script only reads ``uci_1hz_17sensors.npz``.  It does not read labels,
generated data, scalers, or models, and it does not train a model.
"""

from pathlib import Path

import matplotlib
import numpy as np
import pandas as pd


# Use a non-interactive backend so the script also runs on a server.
matplotlib.use("Agg")
import matplotlib.pyplot as plt


PROJECT_ROOT = Path(__file__).resolve().parents[2]
RAW_DATA_PATH = (
    PROJECT_ROOT
    / "data"
    / "processed"
    / "simulator"
    / "uci_1hz_17sensors.npz"
)
ANALYSIS_DIR = (
    PROJECT_ROOT / "data" / "processed" / "simulator" / "analysis"
)
GRAPH_DIR = PROJECT_ROOT / "data" / "processed" / "simulator" / "graphs"

TS1_STATS_PATH = ANALYSIS_DIR / "ts1_60s_position_stats.csv"
ALL_POSITION_STATS_PATH = ANALYSIS_DIR / "all_sensor_60s_position_stats.csv"
SUMMARY_PATH = ANALYSIS_DIR / "all_sensor_60s_pattern_summary.csv"

TS1_MEAN_GRAPH_PATH = GRAPH_DIR / "ts1_60s_position_mean.png"
TS1_MEAN_STD_GRAPH_PATH = GRAPH_DIR / "ts1_60s_mean_std.png"
TS1_OVERLAY_GRAPH_PATH = GRAPH_DIR / "ts1_60s_records_overlay.png"
REPRESENTATIVE_GRAPH_PATH = (
    GRAPH_DIR / "representative_sensors_60s_position_mean.png"
)

SENSOR_NAMES = [
    "PS1",
    "PS2",
    "PS3",
    "PS4",
    "PS5",
    "PS6",
    "EPS1",
    "FS1",
    "FS2",
    "TS1",
    "TS2",
    "TS3",
    "TS4",
    "VS1",
    "CE",
    "CP",
    "SE",
]

EXPECTED_SHAPE = (2205, 60, 17)
SECONDS = np.arange(1, 61)
OVERLAY_RECORD_COUNT = 20
REPRESENTATIVE_SENSORS = ["PS1", "FS1", "TS1", "VS1"]


def load_and_validate_raw_data():
    """Load the single permitted Raw NPZ file and validate its structure."""
    if not RAW_DATA_PATH.exists():
        raise FileNotFoundError(f"Raw dataset not found: {RAW_DATA_PATH}")

    with np.load(RAW_DATA_PATH, allow_pickle=False) as npz_file:
        if "data" not in npz_file.files:
            raise ValueError(f"NPZ key 'data' is missing: {npz_file.files}")
        if "sensor_names" not in npz_file.files:
            raise ValueError(f"NPZ key 'sensor_names' is missing: {npz_file.files}")

        data = npz_file["data"].astype(np.float64)
        sensor_names = [str(name) for name in npz_file["sensor_names"]]

    if data.shape != EXPECTED_SHAPE:
        raise ValueError(
            f"Expected dataset shape {EXPECTED_SHAPE}, actual shape {data.shape}"
        )
    if sensor_names != SENSOR_NAMES:
        raise ValueError(
            "Sensor names/order do not match the expected 17 sensors: "
            f"{sensor_names}"
        )
    if not np.isfinite(data).all():
        raise ValueError("Raw dataset contains NaN or Inf")
    if len(SECONDS) != 60 or SECONDS[0] != 1 or SECONDS[-1] != 60:
        raise ValueError("Second positions must be exactly 1 through 60")

    return data, sensor_names


def calculate_position_statistics(data, sensor_names):
    """Calculate allowed basic statistics at every second position."""
    position_mean = data.mean(axis=0)
    position_std = data.std(axis=0)
    position_min = data.min(axis=0)
    position_max = data.max(axis=0)

    # Subtracting each Record's own first value removes its absolute level.
    # This basic difference makes repeated within-Record shapes visible even
    # when operating levels differ greatly between Records.
    difference_from_start = data - data[:, 0:1, :]
    difference_mean = difference_from_start.mean(axis=0)
    difference_std = difference_from_start.std(axis=0)

    rows = []
    for sensor_index, sensor_name in enumerate(sensor_names):
        for second_index, second in enumerate(SECONDS):
            rows.append(
                {
                    "sensor": sensor_name,
                    "second": int(second),
                    "mean": position_mean[second_index, sensor_index],
                    "std": position_std[second_index, sensor_index],
                    "min": position_min[second_index, sensor_index],
                    "max": position_max[second_index, sensor_index],
                    "mean_minus_start": difference_mean[
                        second_index, sensor_index
                    ],
                    "std_minus_start": difference_std[
                        second_index, sensor_index
                    ],
                }
            )

    all_position_stats = pd.DataFrame(rows)
    ts1_stats = (
        all_position_stats[all_position_stats["sensor"] == "TS1"]
        .drop(columns="sensor")
        .reset_index(drop=True)
    )

    return (
        ts1_stats,
        all_position_stats,
        position_mean,
        position_std,
        difference_std,
    )


def calculate_sensor_summary(data, sensor_names, position_mean, difference_std):
    """Create one summary row per sensor using basic values and differences."""
    rows = []

    for sensor_index, sensor_name in enumerate(sensor_names):
        sensor_data = data[:, :, sensor_index]
        start_values = sensor_data[:, 0]
        end_values = sensor_data[:, -1]
        end_minus_start = end_values - start_values
        boundary_jump = sensor_data[1:, 0] - sensor_data[:-1, -1]
        record_ranges = sensor_data.max(axis=1) - sensor_data.min(axis=1)

        mean_pattern = position_mean[:, sensor_index]
        position_mean_range = mean_pattern.max() - mean_pattern.min()
        overall_std = sensor_data.std()
        mean_difference_std = difference_std[:, sensor_index].mean()

        if overall_std == 0:
            range_over_overall_std = 0.0
        else:
            range_over_overall_std = position_mean_range / overall_std

        if mean_difference_std == 0:
            repeatability_ratio = 0.0
        else:
            repeatability_ratio = position_mean_range / mean_difference_std

        rows.append(
            {
                "sensor": sensor_name,
                "start_mean": start_values.mean(),
                "start_std": start_values.std(),
                "end_mean": end_values.mean(),
                "end_std": end_values.std(),
                "end_minus_start_mean": end_minus_start.mean(),
                "end_minus_start_std": end_minus_start.std(),
                "boundary_jump_mean": boundary_jump.mean(),
                "boundary_jump_std": boundary_jump.std(),
                "overall_mean": sensor_data.mean(),
                "overall_std": overall_std,
                "position_mean_min": mean_pattern.min(),
                "position_mean_max": mean_pattern.max(),
                "position_mean_range": position_mean_range,
                "position_range_over_overall_std": range_over_overall_std,
                "mean_difference_from_start_std": mean_difference_std,
                "repeatability_ratio": repeatability_ratio,
                "record_range_mean": record_ranges.mean(),
                "record_range_std": record_ranges.std(),
            }
        )

    return pd.DataFrame(rows)


def save_ts1_graphs(data, sensor_names, position_mean, position_std):
    """Save the three requested TS1 graphs."""
    ts1_index = sensor_names.index("TS1")
    ts1_mean = position_mean[:, ts1_index]
    ts1_std = position_std[:, ts1_index]

    fig, axis = plt.subplots(figsize=(10, 5))
    axis.plot(SECONDS, ts1_mean, color="tab:red", linewidth=2)
    axis.set_title("TS1 Mean by Second Position (2,205 Raw Records)")
    axis.set_xlabel("Second position in Record")
    axis.set_ylabel("TS1 mean")
    axis.set_xticks(np.arange(0, 61, 5))
    axis.grid(True, alpha=0.3)
    fig.tight_layout()
    fig.savefig(TS1_MEAN_GRAPH_PATH, dpi=150)
    plt.close(fig)

    fig, axis = plt.subplots(figsize=(10, 5))
    axis.plot(SECONDS, ts1_mean, color="tab:red", linewidth=2, label="Mean")
    axis.fill_between(
        SECONDS,
        ts1_mean - ts1_std,
        ts1_mean + ts1_std,
        color="tab:red",
        alpha=0.2,
        label="Mean +/- std",
    )
    axis.set_title("TS1 Mean and Standard Deviation by Second Position")
    axis.set_xlabel("Second position in Record")
    axis.set_ylabel("TS1")
    axis.set_xticks(np.arange(0, 61, 5))
    axis.grid(True, alpha=0.3)
    axis.legend()
    fig.tight_layout()
    fig.savefig(TS1_MEAN_STD_GRAPH_PATH, dpi=150)
    plt.close(fig)

    # The last 20 Records are a deterministic sample from the validation-side
    # end of the dataset; no label or generated data is used for selection.
    first_overlay_index = data.shape[0] - OVERLAY_RECORD_COUNT
    fig, axis = plt.subplots(figsize=(11, 6))
    for record_index in range(first_overlay_index, data.shape[0]):
        axis.plot(
            SECONDS,
            data[record_index, :, ts1_index],
            linewidth=1,
            alpha=0.65,
            label=f"Record {record_index + 1}",
        )
    axis.set_title("TS1 Overlay: Last 20 Raw Records")
    axis.set_xlabel("Second position in Record")
    axis.set_ylabel("TS1")
    axis.set_xticks(np.arange(0, 61, 5))
    axis.grid(True, alpha=0.3)
    axis.legend(ncol=2, fontsize=7, loc="best")
    fig.tight_layout()
    fig.savefig(TS1_OVERLAY_GRAPH_PATH, dpi=150)
    plt.close(fig)


def save_representative_sensor_graph(sensor_names, position_mean):
    """Save readable 60-second mean patterns for four sensor types."""
    fig, axes = plt.subplots(2, 2, figsize=(12, 8), sharex=True)

    for axis, sensor_name in zip(axes.flat, REPRESENTATIVE_SENSORS):
        sensor_index = sensor_names.index(sensor_name)
        axis.plot(
            SECONDS,
            position_mean[:, sensor_index],
            linewidth=2,
            color="tab:blue",
        )
        axis.set_title(f"{sensor_name} 60-second Mean Pattern")
        axis.set_xlabel("Second position")
        axis.set_ylabel(f"{sensor_name} mean")
        axis.set_xticks(np.arange(0, 61, 10))
        axis.grid(True, alpha=0.3)

    fig.tight_layout()
    fig.savefig(REPRESENTATIVE_GRAPH_PATH, dpi=150)
    plt.close(fig)


def print_results(data, sensor_names, ts1_stats, summary):
    """Print validations and the numerical evidence used for interpretation."""
    ts1_row = summary[summary["sensor"] == "TS1"].iloc[0]
    ts1_min_row = ts1_stats.loc[ts1_stats["mean"].idxmin()]
    ts1_max_row = ts1_stats.loc[ts1_stats["mean"].idxmax()]
    adjacent_changes = np.abs(np.diff(ts1_stats["mean"].to_numpy()))
    largest_change_index = int(adjacent_changes.argmax())

    print("=" * 72)
    print("HydroTwin UCI Raw 60-second Pattern Analysis")
    print("=" * 72)
    print(f"Raw file              : {RAW_DATA_PATH}")
    print(f"Dataset Shape          : {data.shape}")
    print(f"Sensor Count           : {len(sensor_names)}")
    print(f"Sensor Names           : {', '.join(sensor_names)}")
    print(f"NaN/Inf                : {not np.isfinite(data).all()}")
    print(f"Second positions       : {SECONDS[0]}..{SECONDS[-1]} ({len(SECONDS)})")
    print("Labels/models/generated: NOT USED")

    print("\n[TS1 60-SECOND POSITION PATTERN]")
    print(
        f"Position mean minimum  : {ts1_min_row['mean']:.9f} "
        f"at second {int(ts1_min_row['second'])}"
    )
    print(
        f"Position mean maximum  : {ts1_max_row['mean']:.9f} "
        f"at second {int(ts1_max_row['second'])}"
    )
    print(f"Position mean range    : {ts1_row['position_mean_range']:.9f}")
    print(
        "Largest adjacent change: "
        f"second {largest_change_index + 1}->{largest_change_index + 2}, "
        f"{adjacent_changes[largest_change_index]:.9f}"
    )
    print(
        "Repeatability ratio     : "
        f"{ts1_row['repeatability_ratio']:.6f} "
        "(position range / mean std of within-Record difference)"
    )

    print("\n[TS1 RECORD BOUNDARY CHECK]")
    print(
        f"Start Mean / Std       : {ts1_row['start_mean']:.9f} / "
        f"{ts1_row['start_std']:.9f}"
    )
    print(
        f"End Mean / Std         : {ts1_row['end_mean']:.9f} / "
        f"{ts1_row['end_std']:.9f}"
    )
    print(
        f"End-Start Mean / Std   : {ts1_row['end_minus_start_mean']:.9f} / "
        f"{ts1_row['end_minus_start_std']:.9f}"
    )
    print(
        "60s mean - 1s mean    : "
        f"{ts1_row['end_mean'] - ts1_row['start_mean']:.9f}"
    )
    print(
        "Next start-prev end    : "
        f"{ts1_row['boundary_jump_mean']:.9f} mean / "
        f"{ts1_row['boundary_jump_std']:.9f} std"
    )

    print("\n[ALL SENSOR SUMMARY]")
    print(
        summary[
            [
                "sensor",
                "start_mean",
                "end_mean",
                "end_minus_start_mean",
                "position_mean_range",
                "repeatability_ratio",
            ]
        ].to_string(index=False)
    )

    ranking = summary.sort_values("repeatability_ratio", ascending=False)
    print("\n[REPEATABILITY RANKING - HIGH TO LOW]")
    print(ranking[["sensor", "repeatability_ratio"]].to_string(index=False))


def main():
    data, sensor_names = load_and_validate_raw_data()
    ANALYSIS_DIR.mkdir(parents=True, exist_ok=True)
    GRAPH_DIR.mkdir(parents=True, exist_ok=True)

    (
        ts1_stats,
        all_position_stats,
        position_mean,
        position_std,
        difference_std,
    ) = calculate_position_statistics(data, sensor_names)
    summary = calculate_sensor_summary(
        data, sensor_names, position_mean, difference_std
    )

    ts1_stats.to_csv(TS1_STATS_PATH, index=False, float_format="%.10f")
    all_position_stats.to_csv(
        ALL_POSITION_STATS_PATH, index=False, float_format="%.10f"
    )
    summary.to_csv(SUMMARY_PATH, index=False, float_format="%.10f")

    save_ts1_graphs(data, sensor_names, position_mean, position_std)
    save_representative_sensor_graph(sensor_names, position_mean)
    print_results(data, sensor_names, ts1_stats, summary)

    print("\n[FILES SAVED]")
    for output_path in [
        TS1_STATS_PATH,
        ALL_POSITION_STATS_PATH,
        SUMMARY_PATH,
        TS1_MEAN_GRAPH_PATH,
        TS1_MEAN_STD_GRAPH_PATH,
        TS1_OVERLAY_GRAPH_PATH,
        REPRESENTATIVE_GRAPH_PATH,
    ]:
        print(output_path)


if __name__ == "__main__":
    main()
