"""Validate and graph the actual 300-message V5 Kafka live demo output."""

from pathlib import Path

import matplotlib
import numpy as np
import pandas as pd


matplotlib.use("Agg")
import matplotlib.pyplot as plt


PROJECT_ROOT = Path(__file__).resolve().parents[2]
DATA_DIR = PROJECT_ROOT / "data" / "processed" / "simulator"
GRAPH_DIR = DATA_DIR / "graphs"
LIVE_FILE = DATA_DIR / "v5_drift_live_300s.csv"
NORMAL_FILE = DATA_DIR / "v5_normal_live_reference_300s.csv"
GROUND_TRUTH_FILE = DATA_DIR / "v5_drift_scenario_ground_truth.csv"
STATISTICS_FILE = DATA_DIR / "v5_drift_phase_statistics.csv"

TEMPERATURE_SENSORS = ["TS1", "TS2", "TS3", "TS4"]
SENSOR_NAMES = [
    "PS1", "PS2", "PS3", "PS4", "PS5", "PS6", "EPS1",
    "FS1", "FS2", "TS1", "TS2", "TS3", "TS4", "VS1",
    "CE", "CP", "SE",
]
NON_TARGET_SENSORS = [
    sensor for sensor in SENSOR_NAMES if sensor not in TEMPERATURE_SENSORS
]
PHASES = [
    "NORMAL", "GRADUAL_DRIFT", "HIGH_TEMP_HOLD", "RECOVERY",
    "FINAL_NORMAL",
]
BOUNDARIES = [60, 120, 180, 240]

TEMPERATURE_GRAPH = GRAPH_DIR / "v5_environment_drift_temperature_300s.png"
TS1_GRAPH = GRAPH_DIR / "v5_environment_drift_ts1_300s.png"
NORMAL_VS_DRIFT_GRAPH = GRAPH_DIR / "v5_normal_vs_drift_ts1.png"


def load_and_validate():
    live = pd.read_csv(LIVE_FILE)
    normal = pd.read_csv(NORMAL_FILE)
    truth = pd.read_csv(GROUND_TRUTH_FILE)
    if live.shape != (300, 19):
        raise ValueError(f"Live CSV shape error: {live.shape}")
    if normal.shape != (300, 18):
        raise ValueError(f"NORMAL CSV shape error: {normal.shape}")
    if truth.shape != (300, 3):
        raise ValueError(f"Ground truth shape error: {truth.shape}")
    if live.columns.tolist() != ["elapsed_sec", "timestamp"] + SENSOR_NAMES:
        raise ValueError(f"Live CSV columns changed: {live.columns.tolist()}")
    if normal.columns.tolist() != ["elapsed_sec"] + SENSOR_NAMES:
        raise ValueError("NORMAL reference columns changed")
    if live["elapsed_sec"].tolist() != list(range(300)):
        raise ValueError("Live elapsed_sec must be exactly 0..299")
    if truth["elapsed_sec"].tolist() != list(range(300)):
        raise ValueError("Ground-truth elapsed_sec must be exactly 0..299")
    numeric = np.concatenate(
        [
            live[SENSOR_NAMES].to_numpy(dtype=np.float64).reshape(-1),
            normal[SENSOR_NAMES].to_numpy(dtype=np.float64).reshape(-1),
            truth["temperature_offset"].to_numpy(dtype=np.float64),
        ]
    )
    if not np.isfinite(numeric).all():
        raise ValueError("Live output contains NaN or Inf")
    return live, normal, truth


def validate_scenario(live, normal, truth):
    expected_counts = {phase: 60 for phase in PHASES}
    actual_counts = truth["phase"].value_counts().to_dict()
    if actual_counts != expected_counts:
        raise ValueError(f"Unexpected phase counts: {actual_counts}")

    offsets = truth["temperature_offset"].to_numpy(dtype=np.float64)
    checks = {
        "normal_zero": np.allclose(offsets[:60], 0.0),
        "drift_start": np.isclose(offsets[60], 0.0),
        "drift_end": np.isclose(offsets[119], 4.0),
        "drift_increasing": np.all(np.diff(offsets[60:120]) >= 0.0),
        "hold_four": np.allclose(offsets[120:180], 4.0),
        "recovery_start": np.isclose(offsets[180], 4.0),
        "recovery_end": np.isclose(offsets[239], 0.0),
        "recovery_decreasing": np.all(np.diff(offsets[180:240]) <= 0.0),
        "final_zero": np.allclose(offsets[240:], 0.0),
    }

    live_values = live[SENSOR_NAMES].to_numpy(dtype=np.float64)
    normal_values = normal[SENSOR_NAMES].to_numpy(dtype=np.float64)
    differences = live_values - normal_values
    temperature_indices = [SENSOR_NAMES.index(sensor) for sensor in TEMPERATURE_SENSORS]
    non_target_indices = [SENSOR_NAMES.index(sensor) for sensor in NON_TARGET_SENSORS]
    expected_temperature_difference = offsets[:, np.newaxis]
    max_temperature_error = np.max(
        np.abs(
            differences[:, temperature_indices]
            - expected_temperature_difference
        )
    )
    max_non_target_difference = np.max(
        np.abs(differences[:, non_target_indices])
    )
    checks["temperature_offset_only"] = max_temperature_error <= 0.000002
    checks["non_target_unchanged"] = max_non_target_difference == 0.0

    initial = live.iloc[:60]
    final = live.iloc[240:]
    final_mean_differences = {
        sensor: abs(final[sensor].mean() - initial[sensor].mean())
        for sensor in TEMPERATURE_SENSORS
    }
    final_return_checks = {
        sensor: difference <= max(initial[sensor].std() * 2.0, 0.10)
        for sensor, difference in final_mean_differences.items()
    }
    checks["final_normal_return"] = all(final_return_checks.values())

    if not all(checks.values()):
        failed = [name for name, passed in checks.items() if not passed]
        raise ValueError(f"Scenario validation failed: {failed}")

    return (
        checks,
        max_temperature_error,
        max_non_target_difference,
        final_mean_differences,
    )


def calculate_phase_statistics(live, normal, truth):
    rows = []
    for phase in PHASES:
        mask = truth["phase"].eq(phase).to_numpy()
        for sensor in TEMPERATURE_SENSORS:
            values = live.loc[mask, sensor].to_numpy(dtype=np.float64)
            normal_values = normal.loc[mask, sensor].to_numpy(dtype=np.float64)
            rows.append(
                {
                    "phase": phase,
                    "sensor": sensor,
                    "count": len(values),
                    "mean": values.mean(),
                    "std": values.std(),
                    "min": values.min(),
                    "max": values.max(),
                    "normal_reference_mean": normal_values.mean(),
                    "normal_reference_std": normal_values.std(),
                    "mean_applied_offset": (values - normal_values).mean(),
                }
            )
    statistics = pd.DataFrame(rows)
    statistics.to_csv(STATISTICS_FILE, index=False, float_format="%.9f")
    return statistics


def add_phase_boundaries(axis):
    for boundary in BOUNDARIES:
        axis.axvline(boundary, color="black", linestyle="--", alpha=0.4)
    axis.set_xlim(0, 300)
    axis.grid(True, alpha=0.3)


def create_graphs(live, normal):
    GRAPH_DIR.mkdir(parents=True, exist_ok=True)
    seconds = live["elapsed_sec"].to_numpy()

    figure, axis = plt.subplots(figsize=(15, 7))
    for sensor in TEMPERATURE_SENSORS:
        axis.plot(seconds, live[sensor], label=sensor)
    add_phase_boundaries(axis)
    axis.set_title("V5 Live Environmental Drift: TS1-TS4")
    axis.set_xlabel("Elapsed time (sec)")
    axis.set_ylabel("Temperature")
    axis.legend()
    figure.tight_layout()
    figure.savefig(TEMPERATURE_GRAPH, dpi=150)
    plt.close(figure)

    figure, axis = plt.subplots(figsize=(15, 7))
    axis.plot(seconds, live["TS1"], color="tab:red", linewidth=2, label="TS1")
    add_phase_boundaries(axis)
    phase_centers = [30, 90, 150, 210, 270]
    phase_labels = ["NORMAL", "RISE", "HOLD", "RECOVERY", "NORMAL"]
    top = live["TS1"].max()
    for center, label in zip(phase_centers, phase_labels):
        axis.text(center, top, label, ha="center", va="bottom", fontsize=9)
    axis.set_title("V5 Live TS1: NORMAL to Drift and Recovery")
    axis.set_xlabel("Elapsed time (sec)")
    axis.set_ylabel("TS1")
    axis.legend()
    figure.tight_layout()
    figure.savefig(TS1_GRAPH, dpi=150)
    plt.close(figure)

    figure, axis = plt.subplots(figsize=(15, 7))
    axis.plot(seconds, normal["TS1"], label="V5 NORMAL before offset")
    axis.plot(seconds, live["TS1"], label="Kafka LIVE after offset", alpha=0.85)
    add_phase_boundaries(axis)
    axis.set_title("V5 NORMAL vs Environmental Drift: TS1")
    axis.set_xlabel("Elapsed time (sec)")
    axis.set_ylabel("TS1")
    axis.legend()
    figure.tight_layout()
    figure.savefig(NORMAL_VS_DRIFT_GRAPH, dpi=150)
    plt.close(figure)


def main():
    live, normal, truth = load_and_validate()
    (
        checks,
        max_temperature_error,
        max_non_target_difference,
        final_mean_differences,
    ) = validate_scenario(live, normal, truth)
    statistics = calculate_phase_statistics(live, normal, truth)
    create_graphs(live, normal)

    print("=" * 96)
    print("V5 ENVIRONMENT DRIFT LIVE VALIDATION")
    print("=" * 96)
    print(f"Live Shape                  : {live.shape}")
    print("NaN / Inf                  : False")
    print(f"Unique timestamps           : {live['timestamp'].nunique()}/300")
    print(f"Max temperature offset error: {max_temperature_error:.9f}")
    print(f"Max non-target difference   : {max_non_target_difference:.9f}")
    print(f"Final-vs-initial mean diff   : {final_mean_differences}")
    print(f"Checks                       : {checks}")
    print("\n[PHASE STATISTICS]")
    print(
        statistics[
            ["phase", "sensor", "mean", "std", "min", "max", "mean_applied_offset"]
        ].to_string(index=False)
    )
    for path in [
        STATISTICS_FILE,
        TEMPERATURE_GRAPH,
        TS1_GRAPH,
        NORMAL_VS_DRIFT_GRAPH,
    ]:
        print(f"[SAVED] {path}")
    print("STATUS: PASS")


if __name__ == "__main__":
    main()
