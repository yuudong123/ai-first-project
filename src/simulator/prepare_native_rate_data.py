"""Prepare native-rate residual training arrays from the original UCI TXT files."""

import argparse
import hashlib
import time

import numpy as np

from native_rate_utils import (
    BASELINES_FILE,
    DATA_METADATA_FILE,
    EXPECTED_RECORDS,
    PROCESSED_DIR,
    RAW_DATA_DIR,
    RESIDUAL_100HZ_FILE,
    RESIDUAL_10HZ_FILE,
    SECONDS_PER_RECORD,
    SENSOR_HZ,
    SENSOR_NAMES,
    SENSORS_100HZ,
    SENSORS_10HZ,
    load_json,
    update_model_metadata,
    write_json,
)


def sha256_file(path):
    digest = hashlib.sha256()
    with path.open("rb") as source:
        for chunk in iter(lambda: source.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def parse_args():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--force", action="store_true", help="replace existing native data")
    return parser.parse_args()


def main():
    args = parse_args()
    output_files = (BASELINES_FILE, RESIDUAL_100HZ_FILE, RESIDUAL_10HZ_FILE, DATA_METADATA_FILE)
    existing = [str(path) for path in output_files if path.exists()]
    if existing and not args.force:
        raise FileExistsError("Native data already exists; use --force: " + ", ".join(existing))

    PROCESSED_DIR.mkdir(parents=True, exist_ok=True)
    baselines = np.lib.format.open_memmap(
        BASELINES_FILE,
        mode="w+",
        dtype=np.float32,
        shape=(EXPECTED_RECORDS, SECONDS_PER_RECORD, len(SENSOR_NAMES)),
    )
    residual_100hz = np.lib.format.open_memmap(
        RESIDUAL_100HZ_FILE,
        mode="w+",
        dtype=np.float32,
        shape=(EXPECTED_RECORDS, SECONDS_PER_RECORD, len(SENSORS_100HZ), 100),
    )
    residual_10hz = np.lib.format.open_memmap(
        RESIDUAL_10HZ_FILE,
        mode="w+",
        dtype=np.float32,
        shape=(EXPECTED_RECORDS, SECONDS_PER_RECORD, len(SENSORS_10HZ), 10),
    )

    sensor_statistics = {}
    source_hashes = {}
    started = time.perf_counter()
    for sensor_index, sensor in enumerate(SENSOR_NAMES):
        rate_hz = SENSOR_HZ[sensor]
        path = RAW_DATA_DIR / f"{sensor}.txt"
        expected_shape = (EXPECTED_RECORDS, SECONDS_PER_RECORD * rate_hz)
        print(f"[LOAD] {sensor:5s} {rate_hz:3d}Hz {path}", flush=True)
        values = np.loadtxt(path, dtype=np.float32)
        if values.shape != expected_shape:
            raise ValueError(f"{sensor}: expected {expected_shape}, got {values.shape}")
        if not np.isfinite(values).all():
            raise ValueError(f"{sensor}: raw TXT contains NaN or Inf")

        blocks = values.reshape(EXPECTED_RECORDS, SECONDS_PER_RECORD, rate_hz)
        means = blocks.mean(axis=2, dtype=np.float64).astype(np.float32)
        baselines[:, :, sensor_index] = means
        statistics = {
            "rate_hz": rate_hz,
            "raw_mean": float(values.mean(dtype=np.float64)),
            "raw_std": float(values.std(dtype=np.float64)),
            "raw_min": float(values.min()),
            "raw_max": float(values.max()),
        }
        if rate_hz > 1:
            residuals = blocks - means[:, :, np.newaxis]
            residual_mean_max_abs = float(
                np.abs(residuals.mean(axis=2, dtype=np.float64)).max()
            )
            statistics.update(
                {
                    "residual_mean": float(residuals.mean(dtype=np.float64)),
                    "residual_std": float(residuals.std(dtype=np.float64)),
                    "residual_min": float(residuals.min()),
                    "residual_max": float(residuals.max()),
                    "block_residual_mean_max_abs": residual_mean_max_abs,
                }
            )
            if rate_hz == 100:
                residual_100hz[:, :, SENSORS_100HZ.index(sensor), :] = residuals
            else:
                residual_10hz[:, :, SENSORS_10HZ.index(sensor), :] = residuals
        else:
            statistics.update(
                {
                    "residual_mean": 0.0,
                    "residual_std": 0.0,
                    "residual_min": 0.0,
                    "residual_max": 0.0,
                    "block_residual_mean_max_abs": 0.0,
                }
            )
        sensor_statistics[sensor] = statistics
        source_hashes[sensor] = sha256_file(path)

    baselines.flush()
    residual_100hz.flush()
    residual_10hz.flush()
    del baselines, residual_100hz, residual_10hz

    metadata = {
        "source": "UCI Hydraulic original TXT files",
        "raw_data_directory": str(RAW_DATA_DIR),
        "records": EXPECTED_RECORDS,
        "seconds_per_record": SECONDS_PER_RECORD,
        "sensor_names": list(SENSOR_NAMES),
        "sensor_hz": SENSOR_HZ,
        "baseline_shape": [EXPECTED_RECORDS, SECONDS_PER_RECORD, len(SENSOR_NAMES)],
        "residual_100hz_shape": [EXPECTED_RECORDS, SECONDS_PER_RECORD, len(SENSORS_100HZ), 100],
        "residual_10hz_shape": [EXPECTED_RECORDS, SECONDS_PER_RECORD, len(SENSORS_10HZ), 10],
        "residual_definition": "each 1-second native block minus that block mean",
        "record_boundaries_joined": False,
        "source_sha256": source_hashes,
        "sensor_statistics": sensor_statistics,
        "preparation_seconds": time.perf_counter() - started,
    }
    write_json(DATA_METADATA_FILE, metadata)
    update_model_metadata("training_data", metadata)
    verified = load_json(DATA_METADATA_FILE)
    print(f"Prepared native-rate data in {verified['preparation_seconds']:.3f}s")
    print(f"Baselines: {BASELINES_FILE}")
    print(f"100Hz residuals: {RESIDUAL_100HZ_FILE}")
    print(f"10Hz residuals: {RESIDUAL_10HZ_FILE}")


if __name__ == "__main__":
    main()
