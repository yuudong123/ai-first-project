"""Validate the fixed V5 generator on 50 independent Validation seeds."""

from pathlib import Path

import joblib
import numpy as np
import pandas as pd
import tensorflow as tf
from sklearn.preprocessing import StandardScaler

from v5_generation_utils import SENSOR_COUNT, WINDOW_SIZE, generate_from_seed_batch


PROJECT_ROOT = Path(__file__).resolve().parents[2]
DATA_DIR = PROJECT_ROOT / "data" / "processed" / "simulator"
MODEL_DIR = PROJECT_ROOT / "models" / "simulator"
RAW_FILE = DATA_DIR / "uci_1hz_17sensors.npz"
MODEL_FILE = MODEL_DIR / "virtual_factory_generator_v5.keras"
INPUT_SCALER_FILE = MODEL_DIR / "input_scaler_v5.joblib"
OFFSET_SCALER_FILE = MODEL_DIR / "offset_scaler_v5.joblib"
BOUNDS_FILE = MODEL_DIR / "sensor_bounds_v5.npz"
SEED_SUMMARY_FILE = DATA_DIR / "v5_multiseed_summary.csv"
SENSOR_SUMMARY_FILE = DATA_DIR / "v5_multiseed_sensor_summary.csv"

TRAIN_RATIO = 0.8
SEED_COUNT = 50
GENERATE_SECONDS = 300
REFERENCE_RECORDS = 5


def phase_aligned_reference(records):
    """Align Raw phase 31 with the first generated second."""
    shifted = np.concatenate(
        [records[:, WINDOW_SIZE:, :], records[:, :WINDOW_SIZE, :]], axis=1
    )
    return shifted.reshape(GENERATE_SECONDS, SENSOR_COUNT)


def main():
    print("=" * 90)
    print("HydroTwin V5 - 50 Validation Seed Test")
    print("=" * 90)

    with np.load(RAW_FILE, allow_pickle=False) as npz_file:
        data = npz_file["data"].astype(np.float32)
        sensor_names = [str(name) for name in npz_file["sensor_names"]]
    validation_start = int(data.shape[0] * TRAIN_RATIO)
    seed_indices = np.arange(validation_start, validation_start + SEED_COUNT)
    seeds = data[seed_indices, :WINDOW_SIZE, :]

    model = tf.keras.models.load_model(MODEL_FILE, compile=False)
    input_scaler = joblib.load(INPUT_SCALER_FILE)
    offset_scaler = joblib.load(OFFSET_SCALER_FILE)
    with np.load(BOUNDS_FILE, allow_pickle=False) as bounds_file:
        sensor_min = bounds_file["sensor_min"]
        sensor_max = bounds_file["sensor_max"]

    generated = generate_from_seed_batch(
        model,
        input_scaler,
        offset_scaler,
        seeds,
        GENERATE_SECONDS,
        sensor_min=sensor_min,
        sensor_max=sensor_max,
        seed_local_bound_indices=[sensor_names.index("PS4")],
    )

    if generated.shape != (SEED_COUNT, GENERATE_SECONDS, SENSOR_COUNT):
        raise ValueError(f"Unexpected multi-seed output: {generated.shape}")
    if not np.isfinite(generated).all():
        raise ValueError("Multi-seed V5 output contains NaN or Inf")

    seed_rows = []
    sensor_rows = []
    for batch_index, seed_index in enumerate(seed_indices):
        raw_records = data[seed_index : seed_index + REFERENCE_RECORDS]
        raw_reference = phase_aligned_reference(raw_records)
        generated_values = generated[batch_index]

        reference_scaler = StandardScaler()
        raw_scaled = reference_scaler.fit_transform(raw_reference)
        generated_scaled = reference_scaler.transform(generated_values)

        raw_mean = raw_reference.mean(axis=0)
        raw_std = raw_reference.std(axis=0)
        generated_mean = generated_values.mean(axis=0)
        generated_std = generated_values.std(axis=0)
        generated_min = generated_values.min(axis=0)
        generated_max = generated_values.max(axis=0)
        range_pass = (generated_min >= sensor_min) & (
            generated_max <= sensor_max
        )
        mean_error_pct = np.abs(generated_mean - raw_mean) / np.maximum(
            np.abs(raw_mean), 1e-8
        ) * 100.0
        std_error_pct = np.abs(generated_std - raw_std) / np.maximum(
            raw_std, 1e-8
        ) * 100.0
        first_mean = generated_values[:60].mean(axis=0)
        previous_mean = generated_values[-120:-60].mean(axis=0)
        last_mean = generated_values[-60:].mean(axis=0)

        seed_rows.append(
            {
                "seed_record": int(seed_index),
                "mean_error_pct": mean_error_pct.mean(),
                "std_error_pct": std_error_pct.mean(),
                "standardized_mse": np.mean(
                    (generated_scaled - raw_scaled) ** 2
                ),
                "standardized_mae": np.mean(
                    np.abs(generated_scaled - raw_scaled)
                ),
                "range_pass_count": int(range_pass.sum()),
                "mean_abs_first_to_last_cycle_change": np.mean(
                    np.abs(last_mean - first_mean)
                ),
                "mean_abs_last_two_cycle_change": np.mean(
                    np.abs(last_mean - previous_mean)
                ),
            }
        )

        for sensor_index, sensor in enumerate(sensor_names):
            sensor_rows.append(
                {
                    "seed_record": int(seed_index),
                    "sensor": sensor,
                    "raw_mean": raw_mean[sensor_index],
                    "raw_std": raw_std[sensor_index],
                    "raw_min": raw_reference[:, sensor_index].min(),
                    "raw_max": raw_reference[:, sensor_index].max(),
                    "generated_mean": generated_mean[sensor_index],
                    "generated_std": generated_std[sensor_index],
                    "generated_min": generated_min[sensor_index],
                    "generated_max": generated_max[sensor_index],
                    "mean_error_pct": mean_error_pct[sensor_index],
                    "std_error_pct": std_error_pct[sensor_index],
                    "last_two_cycle_mean_change": (
                        last_mean[sensor_index] - previous_mean[sensor_index]
                    ),
                    "training_range_pass": bool(range_pass[sensor_index]),
                }
            )

    seed_frame = pd.DataFrame(seed_rows)
    sensor_frame = pd.DataFrame(sensor_rows)
    seed_frame.to_csv(SEED_SUMMARY_FILE, index=False, float_format="%.10f")
    sensor_frame.to_csv(
        SENSOR_SUMMARY_FILE, index=False, float_format="%.10f"
    )

    print(f"Raw Dataset Shape     : {data.shape}")
    print(f"Seed Count            : {SEED_COUNT}")
    print(f"Seed Range            : {seed_indices[0]}..{seed_indices[-1]}")
    print(f"Generated Shape       : {generated.shape}")
    print("NaN / Inf             : False")
    print(
        "Mean Error %         : "
        f"{seed_frame['mean_error_pct'].mean():.6f} mean / "
        f"{seed_frame['mean_error_pct'].std():.6f} std"
    )
    print(
        "Std Error %          : "
        f"{seed_frame['std_error_pct'].mean():.6f} mean / "
        f"{seed_frame['std_error_pct'].std():.6f} std"
    )
    print(
        "Standardized MSE/MAE : "
        f"{seed_frame['standardized_mse'].mean():.6f} / "
        f"{seed_frame['standardized_mae'].mean():.6f}"
    )
    print(
        "Range Pass           : "
        f"{seed_frame['range_pass_count'].min()}.."
        f"{seed_frame['range_pass_count'].max()} / 17"
    )
    print(
        "Last-two-cycle change: "
        f"{seed_frame['mean_abs_last_two_cycle_change'].mean():.9f} mean"
    )
    print(f"[SAVED] {SEED_SUMMARY_FILE}")
    print(f"[SAVED] {SENSOR_SUMMARY_FILE}")


if __name__ == "__main__":
    main()
