"""Generate and validate 300 seconds with the already-trained V5 model."""

from pathlib import Path

import joblib
import numpy as np
import pandas as pd
import tensorflow as tf

from v5_generation_utils import SENSOR_COUNT, WINDOW_SIZE, generate_from_seed_batch


PROJECT_ROOT = Path(__file__).resolve().parents[2]
DATA_DIR = PROJECT_ROOT / "data" / "processed" / "simulator"
MODEL_DIR = PROJECT_ROOT / "models" / "simulator"
RAW_FILE = DATA_DIR / "uci_1hz_17sensors.npz"
MODEL_FILE = MODEL_DIR / "virtual_factory_generator_v5.keras"
INPUT_SCALER_FILE = MODEL_DIR / "input_scaler_v5.joblib"
OFFSET_SCALER_FILE = MODEL_DIR / "offset_scaler_v5.joblib"
BOUNDS_FILE = MODEL_DIR / "sensor_bounds_v5.npz"
OUTPUT_FILE = DATA_DIR / "generated_300s_v5.csv"
SUMMARY_FILE = DATA_DIR / "v5_300s_generation_summary.csv"

TRAIN_RATIO = 0.8
GENERATE_SECONDS = 300
RAW_REFERENCE_RECORDS = 5


def main():
    print("=" * 84)
    print("HydroTwin V5 - 300-second Free Generation")
    print("=" * 84)

    with np.load(RAW_FILE, allow_pickle=False) as npz_file:
        data = npz_file["data"].astype(np.float32)
        sensor_names = [str(name) for name in npz_file["sensor_names"]]
    if data.shape != (2205, 60, 17) or len(sensor_names) != SENSOR_COUNT:
        raise ValueError(f"Unexpected Raw dataset: {data.shape}, {sensor_names}")
    if not np.isfinite(data).all():
        raise ValueError("Raw dataset contains NaN or Inf")

    model = tf.keras.models.load_model(MODEL_FILE, compile=False)
    input_scaler = joblib.load(INPUT_SCALER_FILE)
    offset_scaler = joblib.load(OFFSET_SCALER_FILE)
    with np.load(BOUNDS_FILE, allow_pickle=False) as bounds_file:
        sensor_min = bounds_file["sensor_min"]
        sensor_max = bounds_file["sensor_max"]

    validation_start = int(data.shape[0] * TRAIN_RATIO)
    seed = data[validation_start, :WINDOW_SIZE, :][np.newaxis, ...]
    raw_reference = data[
        validation_start : validation_start + RAW_REFERENCE_RECORDS
    ].reshape(GENERATE_SECONDS, SENSOR_COUNT)

    print(f"Raw Dataset Shape  : {data.shape}")
    print(f"Seed Record        : {validation_start} (zero-based)")
    print(f"Seed Shape         : {seed.shape}")
    print("Inference          : load_model() + predict(), no fit()")

    generated = generate_from_seed_batch(
        model,
        input_scaler,
        offset_scaler,
        seed,
        GENERATE_SECONDS,
        sensor_min=sensor_min,
        sensor_max=sensor_max,
        seed_local_bound_indices=[sensor_names.index("PS4")],
    )[0]

    if generated.shape != (GENERATE_SECONDS, SENSOR_COUNT):
        raise ValueError(f"Unexpected V5 output shape: {generated.shape}")
    if not np.isfinite(generated).all():
        raise ValueError("Generated V5 data contains NaN or Inf")

    raw_all = data.reshape(-1, SENSOR_COUNT)
    raw_global_min = raw_all.min(axis=0)
    raw_global_max = raw_all.max(axis=0)
    generated_min = generated.min(axis=0)
    generated_max = generated.max(axis=0)
    range_pass = (generated_min >= raw_global_min) & (
        generated_max <= raw_global_max
    )

    first_60_mean = generated[:60].mean(axis=0)
    last_60_mean = generated[-60:].mean(axis=0)
    previous_60_mean = generated[-120:-60].mean(axis=0)
    generated_std = generated.std(axis=0)
    reference_std = raw_reference.std(axis=0)
    std_ratio = generated_std / np.maximum(reference_std, 1e-8)

    summary_rows = []
    print("\n[V5 SENSOR VALIDATION]")
    for index, sensor in enumerate(sensor_names):
        row = {
            "sensor": sensor,
            "raw_normal_mean": raw_reference[:, index].mean(),
            "raw_normal_std": reference_std[index],
            "raw_normal_min": raw_reference[:, index].min(),
            "raw_normal_max": raw_reference[:, index].max(),
            "generated_mean": generated[:, index].mean(),
            "generated_std": generated_std[index],
            "generated_min": generated_min[index],
            "generated_max": generated_max[index],
            "first_60_mean": first_60_mean[index],
            "last_60_mean": last_60_mean[index],
            "last_minus_first_60_mean": (
                last_60_mean[index] - first_60_mean[index]
            ),
            "last_minus_previous_60_mean": (
                last_60_mean[index] - previous_60_mean[index]
            ),
            "std_ratio_to_raw_normal": std_ratio[index],
            "raw_global_range_pass": bool(range_pass[index]),
        }
        summary_rows.append(row)
        print(
            f"{sensor:5s} mean={row['generated_mean']:11.5f} "
            f"std={row['generated_std']:10.5f} "
            f"first-last drift={row['last_minus_first_60_mean']:10.5f} "
            f"std_ratio={row['std_ratio_to_raw_normal']:8.3f} "
            f"range={'PASS' if range_pass[index] else 'CHECK'}"
        )

    generated_frame = pd.DataFrame(generated, columns=sensor_names)
    generated_frame.insert(0, "generated_second", np.arange(1, 301))
    # Nine decimals preserve float32 training bounds when CSV is reloaded.
    generated_frame.to_csv(OUTPUT_FILE, index=False, float_format="%.12f")
    pd.DataFrame(summary_rows).to_csv(
        SUMMARY_FILE, index=False, float_format="%.10f"
    )

    first_last_drift = np.abs(last_60_mean - first_60_mean)
    stable_cycle_drift = np.abs(last_60_mean - previous_60_mean)
    convergence_count = int(np.sum(std_ratio < 0.20))
    print("\n[V5 300-SECOND CHECKS]")
    print(f"Generated Shape              : {generated.shape}")
    print("NaN / Inf                    : False")
    print(f"Raw global range pass        : {int(range_pass.sum())}/17")
    print(
        "Largest first-vs-last 60 mean change: "
        f"{first_last_drift.max():.9f}"
    )
    print(
        "Largest last-two-cycle mean change  : "
        f"{stable_cycle_drift.max():.9f}"
    )
    print(f"Sensors with std ratio < 0.2: {convergence_count}/17")
    print(f"[SAVED] {OUTPUT_FILE}")
    print(f"[SAVED] {SUMMARY_FILE}")


if __name__ == "__main__":
    main()
