"""Compare held-out UCI native samples with generated native batches and graph them."""

import json

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from native_rate_utils import (
    DATA_METADATA_FILE,
    GRAPH_DIR,
    PROCESSED_DIR,
    SENSOR_NAMES,
    SENSORS_100HZ,
    SENSORS_10HZ,
    SENSORS_1HZ,
    TRAIN_RECORDS,
    load_json,
    load_prepared_arrays,
    write_json,
)
from test_native_rate_generator import GENERATED_FILE


STATISTICS_FILE = PROCESSED_DIR / "native_rate_statistics.csv"
QUALITY_FILE = PROCESSED_DIR / "native_rate_quality.json"


def describe(values):
    values = np.asarray(values, dtype=np.float64)
    return {
        "mean": float(values.mean()),
        "std": float(values.std()),
        "min": float(values.min()),
        "max": float(values.max()),
    }


def main():
    baselines, residual_100hz = load_prepared_arrays(100)
    _, residual_10hz = load_prepared_arrays(10)
    metadata = load_json(DATA_METADATA_FILE)
    with np.load(GENERATED_FILE, allow_pickle=False) as generated:
        generated_baselines = generated["baselines"]
        generated_100hz = generated["samples_100hz"]
        generated_10hz = generated["samples_10hz"]

    seed_record = TRAIN_RECORDS
    phase_indices = (31 + np.arange(generated_baselines.shape[0])) % 60
    rows = []
    spike_results = {}
    continuity_results = {}
    range_violations = {}
    for sensor_index, sensor in enumerate(SENSOR_NAMES):
        if sensor in SENSORS_100HZ:
            rate_hz = 100
            rate_index = SENSORS_100HZ.index(sensor)
            raw_blocks = (
                np.asarray(baselines[seed_record, phase_indices, sensor_index, np.newaxis])
                + np.asarray(residual_100hz[seed_record, phase_indices, rate_index])
            )
            generated_blocks = generated_100hz[:, rate_index]
            native_spike_applicable = True
        elif sensor in SENSORS_10HZ:
            rate_hz = 10
            rate_index = SENSORS_10HZ.index(sensor)
            raw_blocks = (
                np.asarray(baselines[seed_record, phase_indices, sensor_index, np.newaxis])
                + np.asarray(residual_10hz[seed_record, phase_indices, rate_index])
            )
            generated_blocks = generated_10hz[:, rate_index]
            native_spike_applicable = True
        else:
            rate_hz = 1
            raw_blocks = np.asarray(
                baselines[seed_record, phase_indices, sensor_index]
            )[:, np.newaxis]
            generated_blocks = generated_baselines[:, sensor_index, np.newaxis]
            native_spike_applicable = False

        raw_values = raw_blocks.reshape(-1)
        generated_values = generated_blocks.reshape(-1)
        raw_stats = describe(raw_values)
        generated_stats = describe(generated_values)
        if native_spike_applicable:
            raw_max_abs_step = float(np.abs(np.diff(raw_blocks, axis=1)).max())
            generated_max_abs_step = float(
                np.abs(np.diff(generated_blocks, axis=1)).max()
            )
            spike_pass = generated_max_abs_step <= raw_max_abs_step + 1e-6
        else:
            raw_max_abs_step = 0.0
            generated_max_abs_step = 0.0
            spike_pass = True
        raw_boundary_step = float(
            np.abs(raw_blocks[1:, 0] - raw_blocks[:-1, -1]).max()
        )
        generated_boundary_step = float(
            np.abs(generated_blocks[1:, 0] - generated_blocks[:-1, -1]).max()
        )
        spike_results[sensor] = {
            "scope": "within_second_native_waveform",
            "applicable": native_spike_applicable,
            "raw_max_abs_step": raw_max_abs_step,
            "generated_max_abs_step": generated_max_abs_step,
            "pass": spike_pass,
        }
        continuity_results[sensor] = {
            "scope": "between_second_boundary_includes_existing_v5_baseline_change",
            "raw_max_abs_step": raw_boundary_step,
            "generated_max_abs_step": generated_boundary_step,
            "pass": generated_boundary_step <= raw_boundary_step + 1e-6,
        }
        raw_min = metadata["sensor_statistics"][sensor]["raw_min"]
        raw_max = metadata["sensor_statistics"][sensor]["raw_max"]
        violation_count = int(
            np.count_nonzero(
                (generated_values < raw_min - 1e-6)
                | (generated_values > raw_max + 1e-6)
            )
        )
        range_violations[sensor] = violation_count
        rows.append(
            {
                "sensor": sensor,
                "rate_hz": rate_hz,
                **{f"raw_{name}": value for name, value in raw_stats.items()},
                **{f"generated_{name}": value for name, value in generated_stats.items()},
                "mean_error": generated_stats["mean"] - raw_stats["mean"],
                "std_error": generated_stats["std"] - raw_stats["std"],
                "raw_native_max_abs_step": raw_max_abs_step,
                "generated_native_max_abs_step": generated_max_abs_step,
                "native_spike_pass": spike_pass,
                "raw_boundary_max_abs_step": raw_boundary_step,
                "generated_boundary_max_abs_step": generated_boundary_step,
                "continuity_pass": continuity_results[sensor]["pass"],
                "raw_range_violation_count": violation_count,
            }
        )

    mean_errors = []
    for index in range(len(SENSORS_100HZ)):
        mean_errors.extend(
            np.abs(generated_100hz[:, index].mean(axis=1) - generated_baselines[:, index])
        )
    offset = len(SENSORS_100HZ)
    for index in range(len(SENSORS_10HZ)):
        mean_errors.extend(
            np.abs(generated_10hz[:, index].mean(axis=1) - generated_baselines[:, offset + index])
        )

    frame = pd.DataFrame(rows)
    PROCESSED_DIR.mkdir(parents=True, exist_ok=True)
    GRAPH_DIR.mkdir(parents=True, exist_ok=True)
    frame.to_csv(STATISTICS_FILE, index=False)
    quality = {
        "status": "PASS"
        if not any(range_violations.values()) and all(item["pass"] for item in spike_results.values())
        else "FAIL",
        "validation_source": "same-condition held-out UCI seed Record 1765 at matching cycle phases",
        "generated_seconds": int(generated_baselines.shape[0]),
        "samples_per_generated_second": 728,
        "max_native_mean_vs_v5_baseline_abs_error": float(max(mean_errors)),
        "raw_range_violations": range_violations,
        "spike_checks": spike_results,
        "between_second_continuity_checks": continuity_results,
        "nan_count": int(
            np.isnan(generated_baselines).sum()
            + np.isnan(generated_100hz).sum()
            + np.isnan(generated_10hz).sum()
        ),
        "inf_count": int(
            np.isinf(generated_baselines).sum()
            + np.isinf(generated_100hz).sum()
            + np.isinf(generated_10hz).sum()
        ),
    }
    write_json(QUALITY_FILE, quality)

    representative = (("PS1", 100), ("EPS1", 100), ("FS1", 10))
    raw_record_index = TRAIN_RECORDS
    raw_second = 31
    figure, axes = plt.subplots(3, 1, figsize=(12, 10), constrained_layout=True)
    for axis, (sensor, rate_hz) in zip(axes, representative):
        sensor_index = SENSOR_NAMES.index(sensor)
        if rate_hz == 100:
            rate_index = SENSORS_100HZ.index(sensor)
            raw_values = (
                float(baselines[raw_record_index, raw_second, sensor_index])
                + np.asarray(residual_100hz[raw_record_index, raw_second, rate_index])
            )
            generated_values = generated_100hz[0, rate_index]
        else:
            rate_index = SENSORS_10HZ.index(sensor)
            raw_values = (
                float(baselines[raw_record_index, raw_second, sensor_index])
                + np.asarray(residual_10hz[raw_record_index, raw_second, rate_index])
            )
            generated_values = generated_10hz[0, rate_index]
        x = np.arange(rate_hz) / rate_hz
        axis.plot(x, raw_values, label="Held-out UCI Raw", linewidth=1.5)
        axis.plot(x, generated_values, label="Generated Native", linewidth=1.2)
        axis.set_title(f"{sensor} native waveform - 1 second ({rate_hz}Hz)")
        axis.set_xlabel("Time within second (s)")
        axis.set_ylabel(sensor)
        axis.grid(alpha=0.25)
        axis.legend()
    one_second_graph = GRAPH_DIR / "raw_vs_generated_representative_1s.png"
    figure.savefig(one_second_graph, dpi=150)
    plt.close(figure)

    figure, axes = plt.subplots(3, 1, figsize=(14, 10), constrained_layout=True)
    for axis, (sensor, rate_hz) in zip(axes, representative):
        sensor_index = SENSOR_NAMES.index(sensor)
        if rate_hz == 100:
            rate_index = SENSORS_100HZ.index(sensor)
            raw_values = (
                np.asarray(baselines[raw_record_index, raw_second:raw_second + 10, sensor_index, np.newaxis])
                + np.asarray(residual_100hz[raw_record_index, raw_second:raw_second + 10, rate_index])
            ).reshape(-1)
            generated_values = generated_100hz[:10, rate_index].reshape(-1)
        else:
            rate_index = SENSORS_10HZ.index(sensor)
            raw_values = (
                np.asarray(baselines[raw_record_index, raw_second:raw_second + 10, sensor_index, np.newaxis])
                + np.asarray(residual_10hz[raw_record_index, raw_second:raw_second + 10, rate_index])
            ).reshape(-1)
            generated_values = generated_10hz[:10, rate_index].reshape(-1)
        x = np.arange(10 * rate_hz) / rate_hz
        axis.plot(x, raw_values, label="Held-out UCI Raw", linewidth=1.0)
        axis.plot(x, generated_values, label="Generated Native", linewidth=0.9)
        axis.set_title(f"{sensor} native waveform continuity - 10 seconds")
        axis.set_xlabel("Time (s)")
        axis.set_ylabel(sensor)
        axis.grid(alpha=0.25)
        axis.legend()
    ten_second_graph = GRAPH_DIR / "raw_vs_generated_representative_10s.png"
    figure.savefig(ten_second_graph, dpi=150)
    plt.close(figure)

    print(frame.to_string(index=False))
    print(json.dumps(quality, ensure_ascii=False, indent=2))
    print(f"Graphs: {one_second_graph}, {ten_second_graph}")
    if quality["status"] != "PASS":
        raise SystemExit(1)


if __name__ == "__main__":
    main()
