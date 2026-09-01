"""Shared data, training, and runtime helpers for native-rate generation.

The native models predict only the within-second residual waveforms.  The
existing V5 model remains the source of each second's 17 sensor baselines.
"""

from __future__ import annotations

import json
import os
import time
from datetime import datetime
from pathlib import Path
from typing import Any

import joblib
import numpy as np
from sklearn.preprocessing import StandardScaler


PROJECT_ROOT = Path(__file__).resolve().parents[2]
RAW_DATA_DIR = PROJECT_ROOT / "data" / "raw" / "uci_hydraulic" / "extracted"
PROCESSED_DIR = PROJECT_ROOT / "data" / "processed" / "simulator" / "native_rate"
GRAPH_DIR = PROJECT_ROOT / "data" / "processed" / "simulator" / "graphs" / "native_rate"
MODEL_DIR = PROJECT_ROOT / "models" / "simulator"

BASELINES_FILE = PROCESSED_DIR / "native_baselines.npy"
RESIDUAL_100HZ_FILE = PROCESSED_DIR / "native_residual_100hz.npy"
RESIDUAL_10HZ_FILE = PROCESSED_DIR / "native_residual_10hz.npy"
DATA_METADATA_FILE = PROCESSED_DIR / "native_rate_data_metadata.json"
METADATA_FILE = MODEL_DIR / "native_rate_metadata.json"
MODEL_100HZ_FILE = MODEL_DIR / "native_100hz_generator.keras"
MODEL_10HZ_FILE = MODEL_DIR / "native_10hz_generator.keras"
SCALER_100HZ_FILE = MODEL_DIR / "native_100hz_scaler.joblib"
SCALER_10HZ_FILE = MODEL_DIR / "native_10hz_scaler.joblib"
HISTORY_100HZ_FILE = PROCESSED_DIR / "native_100hz_training_history.csv"
HISTORY_10HZ_FILE = PROCESSED_DIR / "native_10hz_training_history.csv"

V5_DATA_FILE = PROJECT_ROOT / "data" / "processed" / "simulator" / "uci_1hz_17sensors.npz"
V5_MODEL_FILE = MODEL_DIR / "virtual_factory_generator_v5.keras"
V5_INPUT_SCALER_FILE = MODEL_DIR / "input_scaler_v5.joblib"
V5_OFFSET_SCALER_FILE = MODEL_DIR / "offset_scaler_v5.joblib"
V5_BOUNDS_FILE = MODEL_DIR / "sensor_bounds_v5.npz"

SENSOR_NAMES = (
    "PS1", "PS2", "PS3", "PS4", "PS5", "PS6", "EPS1",
    "FS1", "FS2",
    "TS1", "TS2", "TS3", "TS4", "VS1", "CE", "CP", "SE",
)
SENSORS_100HZ = SENSOR_NAMES[:7]
SENSORS_10HZ = SENSOR_NAMES[7:9]
SENSORS_1HZ = SENSOR_NAMES[9:]
SENSOR_HZ = {
    **{sensor: 100 for sensor in SENSORS_100HZ},
    **{sensor: 10 for sensor in SENSORS_10HZ},
    **{sensor: 1 for sensor in SENSORS_1HZ},
}

EXPECTED_RECORDS = 2205
SECONDS_PER_RECORD = 60
WINDOW_SIZE = 30
TRAIN_RATIO = 0.8
TRAIN_RECORDS = int(EXPECTED_RECORDS * TRAIN_RATIO)
TARGET_SECONDS = tuple(range(SECONDS_PER_RECORD))
NATIVE_TOPIC = "hydraulic.sensor.native"
TOTAL_SAMPLES_PER_BATCH = 728


def load_json(path: Path) -> dict[str, Any]:
    with path.open("r", encoding="utf-8") as source:
        return json.load(source)


def write_json(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    with temporary.open("w", encoding="utf-8") as destination:
        json.dump(payload, destination, ensure_ascii=False, indent=2)
        destination.write("\n")
    os.replace(temporary, path)


def update_model_metadata(section: str, payload: dict[str, Any]) -> None:
    metadata: dict[str, Any]
    if METADATA_FILE.exists():
        metadata = load_json(METADATA_FILE)
    else:
        metadata = {
            "version": "native_rate_v1",
            "architecture": "separate_100hz_and_10hz_residual_generators",
            "baseline_source": "existing_v5_1hz_generator",
            "sensor_names": list(SENSOR_NAMES),
            "window_size": WINDOW_SIZE,
            "input_shape": [WINDOW_SIZE, len(SENSOR_NAMES) + 1],
            "record_boundaries_joined": False,
            "generated_data_used_for_training": False,
            "kafka_topic": NATIVE_TOPIC,
        }
    metadata[section] = payload
    write_json(METADATA_FILE, metadata)


def assert_prepared_files() -> None:
    missing = [
        str(path)
        for path in (
            BASELINES_FILE,
            RESIDUAL_100HZ_FILE,
            RESIDUAL_10HZ_FILE,
            DATA_METADATA_FILE,
        )
        if not path.exists()
    ]
    if missing:
        raise FileNotFoundError(
            "Native-rate data is not prepared. Missing: " + ", ".join(missing)
        )


def load_prepared_arrays(rate_hz: int | None = None):
    assert_prepared_files()
    baselines = np.load(BASELINES_FILE, mmap_mode="r")
    if baselines.shape != (EXPECTED_RECORDS, SECONDS_PER_RECORD, len(SENSOR_NAMES)):
        raise ValueError(f"Unexpected baseline shape: {baselines.shape}")
    if rate_hz == 100:
        residuals = np.load(RESIDUAL_100HZ_FILE, mmap_mode="r")
        expected = (EXPECTED_RECORDS, SECONDS_PER_RECORD, len(SENSORS_100HZ), 100)
    elif rate_hz == 10:
        residuals = np.load(RESIDUAL_10HZ_FILE, mmap_mode="r")
        expected = (EXPECTED_RECORDS, SECONDS_PER_RECORD, len(SENSORS_10HZ), 10)
    elif rate_hz is None:
        return baselines
    else:
        raise ValueError(f"Unsupported rate: {rate_hz}")
    if residuals.shape != expected:
        raise ValueError(f"Unexpected {rate_hz}Hz residual shape: {residuals.shape}")
    return baselines, residuals


def build_input_windows(baselines: np.ndarray, start: int, stop: int) -> np.ndarray:
    """Build 30x18 windows without ever crossing a Record boundary."""
    record_count = stop - start
    window_count = record_count * len(TARGET_SECONDS)
    inputs = np.empty(
        (window_count, WINDOW_SIZE, len(SENSOR_NAMES) + 1), dtype=np.float32
    )
    row = 0
    for record_index in range(start, stop):
        record = np.asarray(baselines[record_index], dtype=np.float32)
        for target_second in TARGET_SECONDS:
            input_seconds = (
                np.arange(target_second - WINDOW_SIZE, target_second)
                % SECONDS_PER_RECORD
            )
            inputs[row, :, : len(SENSOR_NAMES)] = record[input_seconds]
            inputs[row, :, -1] = input_seconds / float(SECONDS_PER_RECORD - 1)
            row += 1
    return inputs


def build_targets(
    residuals: np.ndarray, start: int, stop: int, rate_hz: int
) -> np.ndarray:
    selected = np.asarray(residuals[start:stop], dtype=np.float32)
    sensor_count = len(SENSORS_100HZ if rate_hz == 100 else SENSORS_10HZ)
    return selected.reshape(-1, sensor_count * rate_hz)


def scale_input_windows(
    train_inputs: np.ndarray,
    validation_inputs: np.ndarray,
) -> tuple[np.ndarray, np.ndarray, StandardScaler]:
    scaler = StandardScaler()
    sensor_count = len(SENSOR_NAMES)
    scaler.fit(train_inputs[:, :, :sensor_count].reshape(-1, sensor_count))
    for values in (train_inputs, validation_inputs):
        scaled = scaler.transform(values[:, :, :sensor_count].reshape(-1, sensor_count))
        values[:, :, :sensor_count] = scaled.reshape(values.shape[0], WINDOW_SIZE, sensor_count)
    return train_inputs, validation_inputs, scaler


def center_residuals(residuals: np.ndarray) -> np.ndarray:
    residuals = np.asarray(residuals, dtype=np.float64)
    return residuals - residuals.mean(axis=-1, keepdims=True)


def rate_configuration(rate_hz: int) -> dict[str, Any]:
    if rate_hz == 100:
        return {
            "sensors": SENSORS_100HZ,
            "model_file": MODEL_100HZ_FILE,
            "scaler_file": SCALER_100HZ_FILE,
            "history_file": HISTORY_100HZ_FILE,
        }
    if rate_hz == 10:
        return {
            "sensors": SENSORS_10HZ,
            "model_file": MODEL_10HZ_FILE,
            "scaler_file": SCALER_10HZ_FILE,
            "history_file": HISTORY_10HZ_FILE,
        }
    raise ValueError(f"Unsupported rate: {rate_hz}")


def build_residual_model(output_size: int, name: str):
    import tensorflow as tf

    model = tf.keras.Sequential(
        [
            tf.keras.layers.Input(shape=(WINDOW_SIZE, len(SENSOR_NAMES) + 1)),
            tf.keras.layers.LSTM(48),
            tf.keras.layers.Dense(96, activation="relu"),
            tf.keras.layers.Dense(output_size),
        ],
        name=name,
    )
    model.compile(optimizer="adam", loss="mse", metrics=["mae"])
    return model


def train_rate_model(
    rate_hz: int,
    epochs: int = 20,
    batch_size: int = 256,
    patience: int = 5,
    random_seed: int = 42,
) -> dict[str, Any]:
    """Train one rate-specific model from raw-TXT-derived residual arrays."""
    import pandas as pd
    import tensorflow as tf

    if epochs <= 0 or batch_size <= 0 or patience < 0:
        raise ValueError("epochs/batch_size must be positive and patience non-negative")
    np.random.seed(random_seed)
    tf.random.set_seed(random_seed)

    configuration = rate_configuration(rate_hz)
    sensors = configuration["sensors"]
    baselines, residuals = load_prepared_arrays(rate_hz)

    print(f"Building independent Record windows for {rate_hz}Hz...")
    x_train = build_input_windows(baselines, 0, TRAIN_RECORDS)
    x_validation = build_input_windows(baselines, TRAIN_RECORDS, EXPECTED_RECORDS)
    x_train, x_validation, input_scaler = scale_input_windows(x_train, x_validation)
    y_train_raw = build_targets(residuals, 0, TRAIN_RECORDS, rate_hz)
    y_validation_raw = build_targets(residuals, TRAIN_RECORDS, EXPECTED_RECORDS, rate_hz)

    target_scaler = StandardScaler()
    y_train = target_scaler.fit_transform(y_train_raw).astype(np.float32)
    y_validation = target_scaler.transform(y_validation_raw).astype(np.float32)

    model = build_residual_model(len(sensors) * rate_hz, f"native_{rate_hz}hz_generator")
    configuration["model_file"].parent.mkdir(parents=True, exist_ok=True)
    configuration["history_file"].parent.mkdir(parents=True, exist_ok=True)
    callbacks = [
        tf.keras.callbacks.EarlyStopping(
            monitor="val_loss", patience=patience, restore_best_weights=True, verbose=1
        ),
        tf.keras.callbacks.ModelCheckpoint(
            configuration["model_file"], monitor="val_loss", save_best_only=True, verbose=1
        ),
    ]

    print(f"X train/validation: {x_train.shape} / {x_validation.shape}")
    print(f"Y train/validation: {y_train.shape} / {y_validation.shape}")
    model.summary()
    started = time.perf_counter()
    history = model.fit(
        x_train,
        y_train,
        validation_data=(x_validation, y_validation),
        epochs=epochs,
        batch_size=batch_size,
        callbacks=callbacks,
        shuffle=True,
        verbose=2,
    )
    training_seconds = time.perf_counter() - started
    model.save(configuration["model_file"])
    joblib.dump(
        {
            "input_scaler": input_scaler,
            "target_scaler": target_scaler,
            "rate_hz": rate_hz,
            "sensor_names": list(sensors),
            "window_size": WINDOW_SIZE,
        },
        configuration["scaler_file"],
    )

    history_frame = pd.DataFrame(history.history)
    history_frame.insert(0, "epoch", np.arange(1, len(history_frame) + 1))
    history_frame.to_csv(configuration["history_file"], index=False)

    prediction_scaled = model.predict(x_validation, batch_size=batch_size, verbose=0)
    prediction = target_scaler.inverse_transform(prediction_scaled).reshape(
        -1, len(sensors), rate_hz
    )
    target = y_validation_raw.reshape(-1, len(sensors), rate_hz)
    prediction = center_residuals(prediction)
    mse = float(np.mean((prediction - target) ** 2))
    mae = float(np.mean(np.abs(prediction - target)))
    per_sensor = {}
    for index, sensor in enumerate(sensors):
        error = prediction[:, index] - target[:, index]
        per_sensor[sensor] = {
            "mse_raw_units": float(np.mean(error ** 2)),
            "mae_raw_units": float(np.mean(np.abs(error))),
            "generated_residual_std": float(np.std(prediction[:, index])),
            "raw_residual_std": float(np.std(target[:, index])),
        }

    best_epoch = int(history_frame["val_loss"].idxmin()) + 1
    result = {
        "rate_hz": rate_hz,
        "sensors": list(sensors),
        "input_shape": [WINDOW_SIZE, len(SENSOR_NAMES) + 1],
        "output_shape": [len(sensors), rate_hz],
        "output_count": len(sensors) * rate_hz,
        "training_records": TRAIN_RECORDS,
        "validation_records": EXPECTED_RECORDS - TRAIN_RECORDS,
        "training_windows": int(x_train.shape[0]),
        "validation_windows": int(x_validation.shape[0]),
        "epochs_requested": epochs,
        "epochs_trained": int(len(history_frame)),
        "best_epoch": best_epoch,
        "best_validation_mse_scaled": float(history_frame["val_loss"].min()),
        "validation_mse_raw_units": mse,
        "validation_mae_raw_units": mae,
        "training_seconds": training_seconds,
        "parameter_count": int(model.count_params()),
        "model_file": str(configuration["model_file"].relative_to(PROJECT_ROOT)),
        "scaler_file": str(configuration["scaler_file"].relative_to(PROJECT_ROOT)),
        "training_source": "UCI original TXT-derived residuals only",
        "record_boundaries_joined": False,
        "per_sensor": per_sensor,
    }
    update_model_metadata(f"model_{rate_hz}hz", result)
    return result


def build_runtime_input(
    baseline_window: np.ndarray,
    phase_window: np.ndarray,
    input_scaler: StandardScaler,
) -> np.ndarray:
    baseline_window = np.asarray(baseline_window, dtype=np.float32)
    phase_window = np.asarray(phase_window, dtype=np.int32)
    if baseline_window.shape != (WINDOW_SIZE, len(SENSOR_NAMES)):
        raise ValueError(f"Unexpected baseline window: {baseline_window.shape}")
    if phase_window.shape != (WINDOW_SIZE,):
        raise ValueError(f"Unexpected phase window: {phase_window.shape}")
    values = np.empty((1, WINDOW_SIZE, len(SENSOR_NAMES) + 1), dtype=np.float32)
    values[0, :, : len(SENSOR_NAMES)] = input_scaler.transform(baseline_window)
    values[0, :, -1] = phase_window / float(SECONDS_PER_RECORD - 1)
    return values


class NativeRateRuntime:
    """Loaded 100Hz and 10Hz residual generators with runtime safeguards."""

    def __init__(self):
        import tensorflow as tf

        self.model_100hz = tf.keras.models.load_model(MODEL_100HZ_FILE, compile=False)
        self.model_10hz = tf.keras.models.load_model(MODEL_10HZ_FILE, compile=False)
        self.scalers_100hz = joblib.load(SCALER_100HZ_FILE)
        self.scalers_10hz = joblib.load(SCALER_10HZ_FILE)
        metadata = load_json(DATA_METADATA_FILE)
        statistics = metadata["sensor_statistics"]
        self.raw_min = np.asarray([statistics[name]["raw_min"] for name in SENSOR_NAMES])
        self.raw_max = np.asarray([statistics[name]["raw_max"] for name in SENSOR_NAMES])
        self.residual_min_100hz = np.asarray(
            [statistics[name]["residual_min"] for name in SENSORS_100HZ]
        )
        self.residual_max_100hz = np.asarray(
            [statistics[name]["residual_max"] for name in SENSORS_100HZ]
        )
        self.residual_min_10hz = np.asarray(
            [statistics[name]["residual_min"] for name in SENSORS_10HZ]
        )
        self.residual_max_10hz = np.asarray(
            [statistics[name]["residual_max"] for name in SENSORS_10HZ]
        )

    @staticmethod
    def _range_safe_residuals(
        residuals: np.ndarray,
        baselines: np.ndarray,
        raw_min: np.ndarray,
        raw_max: np.ndarray,
        residual_min: np.ndarray,
        residual_max: np.ndarray,
    ) -> np.ndarray:
        residuals = center_residuals(residuals)
        residuals = np.minimum(
            np.maximum(residuals, residual_min[:, np.newaxis]),
            residual_max[:, np.newaxis],
        )
        residuals = center_residuals(residuals)
        for index in range(residuals.shape[0]):
            positive_peak = max(float(residuals[index].max()), 0.0)
            negative_peak = max(float(-residuals[index].min()), 0.0)
            scale = 1.0
            if positive_peak > 0:
                scale = min(scale, (raw_max[index] - baselines[index]) / positive_peak)
            if negative_peak > 0:
                scale = min(scale, (baselines[index] - raw_min[index]) / negative_peak)
            residuals[index] *= max(0.0, min(1.0, float(scale)))
        if not np.isfinite(residuals).all():
            raise ValueError("Native residual inference contains NaN or Inf")
        return residuals

    def _predict_rate(
        self,
        model,
        scaler_bundle: dict[str, Any],
        baseline_window: np.ndarray,
        phase_window: np.ndarray,
        sensor_count: int,
        rate_hz: int,
    ) -> np.ndarray:
        inputs = build_runtime_input(
            baseline_window, phase_window, scaler_bundle["input_scaler"]
        )
        predicted_scaled = model.predict(inputs, verbose=0)
        return scaler_bundle["target_scaler"].inverse_transform(
            predicted_scaled
        ).reshape(sensor_count, rate_hz)

    def generate(
        self,
        baseline: np.ndarray,
        baseline_window: np.ndarray,
        phase_window: np.ndarray,
    ) -> tuple[dict[str, list[float]], dict[str, float]]:
        baseline = np.asarray(baseline, dtype=np.float64)
        if baseline.shape != (len(SENSOR_NAMES),):
            raise ValueError(f"Unexpected V5 baseline shape: {baseline.shape}")
        if not np.isfinite(baseline).all():
            raise ValueError("V5 baseline contains NaN or Inf")

        start_100hz = time.perf_counter()
        residual_100hz = self._predict_rate(
            self.model_100hz,
            self.scalers_100hz,
            baseline_window,
            phase_window,
            len(SENSORS_100HZ),
            100,
        )
        inference_100hz = time.perf_counter() - start_100hz
        residual_100hz = self._range_safe_residuals(
            residual_100hz,
            baseline[: len(SENSORS_100HZ)],
            self.raw_min[: len(SENSORS_100HZ)],
            self.raw_max[: len(SENSORS_100HZ)],
            self.residual_min_100hz,
            self.residual_max_100hz,
        )

        start_10hz = time.perf_counter()
        residual_10hz = self._predict_rate(
            self.model_10hz,
            self.scalers_10hz,
            baseline_window,
            phase_window,
            len(SENSORS_10HZ),
            10,
        )
        inference_10hz = time.perf_counter() - start_10hz
        residual_10hz = self._range_safe_residuals(
            residual_10hz,
            baseline[len(SENSORS_100HZ) : len(SENSORS_100HZ) + len(SENSORS_10HZ)],
            self.raw_min[len(SENSORS_100HZ) : len(SENSORS_100HZ) + len(SENSORS_10HZ)],
            self.raw_max[len(SENSORS_100HZ) : len(SENSORS_100HZ) + len(SENSORS_10HZ)],
            self.residual_min_10hz,
            self.residual_max_10hz,
        )

        sensors: dict[str, list[float]] = {}
        for index, sensor in enumerate(SENSORS_100HZ):
            sensors[sensor] = (baseline[index] + residual_100hz[index]).tolist()
        offset = len(SENSORS_100HZ)
        for index, sensor in enumerate(SENSORS_10HZ):
            sensors[sensor] = (baseline[offset + index] + residual_10hz[index]).tolist()
        offset += len(SENSORS_10HZ)
        for index, sensor in enumerate(SENSORS_1HZ):
            sensors[sensor] = [float(baseline[offset + index])]

        timings = {
            "native_100hz_inference_seconds": inference_100hz,
            "native_10hz_inference_seconds": inference_10hz,
        }
        return sensors, timings


def create_native_message(sensors: dict[str, list[float]]) -> dict[str, Any]:
    rounded = {
        sensor: [round(float(value), 6) for value in sensors[sensor]]
        for sensor in SENSOR_NAMES
    }
    message = {
        "timestamp": datetime.now().astimezone().isoformat(timespec="milliseconds"),
        "sensors": rounded,
    }
    validate_native_message(message)
    return message


def validate_native_message(message: dict[str, Any]) -> None:
    if set(message) != {"timestamp", "sensors"}:
        raise ValueError(f"Unexpected native message keys: {list(message)}")
    if list(message["sensors"]) != list(SENSOR_NAMES):
        raise ValueError("Native sensor names/order changed")
    expected_lengths = {
        **{sensor: 100 for sensor in SENSORS_100HZ},
        **{sensor: 10 for sensor in SENSORS_10HZ},
        **{sensor: 1 for sensor in SENSORS_1HZ},
    }
    total = 0
    for sensor, expected_length in expected_lengths.items():
        values = np.asarray(message["sensors"][sensor], dtype=np.float64)
        if values.shape != (expected_length,):
            raise ValueError(f"{sensor}: expected length {expected_length}, got {values.shape}")
        if not np.isfinite(values).all():
            raise ValueError(f"{sensor}: NaN or Inf")
        total += values.size
    if total != TOTAL_SAMPLES_PER_BATCH:
        raise ValueError(f"Expected {TOTAL_SAMPLES_PER_BATCH} samples, got {total}")
    forbidden = {
        "prediction", "label", "model", "confidence", "drift", "normal", "feature"
    }
    if forbidden.intersection(message):
        raise ValueError("Native message contains forbidden metadata")


class V5BaselineRuntime:
    """Read-only adapter around the existing V5 artifacts for native runtime tests."""

    def __init__(self, seed_record: int = TRAIN_RECORDS):
        import tensorflow as tf

        from v5_generation_utils import build_model_inputs

        with np.load(V5_DATA_FILE, allow_pickle=False) as source:
            raw_data = source["data"].astype(np.float32)
            self.sensor_names = [str(value) for value in source["sensor_names"]]
        if self.sensor_names != list(SENSOR_NAMES):
            raise ValueError("V5 sensor names/order changed")
        if not TRAIN_RECORDS <= seed_record < EXPECTED_RECORDS:
            raise ValueError(f"Seed must be validation Record {TRAIN_RECORDS}..{EXPECTED_RECORDS - 1}")
        self.model = tf.keras.models.load_model(V5_MODEL_FILE, compile=False)
        self.input_scaler = joblib.load(V5_INPUT_SCALER_FILE)
        self.offset_scaler = joblib.load(V5_OFFSET_SCALER_FILE)
        with np.load(V5_BOUNDS_FILE, allow_pickle=False) as bounds:
            self.sensor_min = bounds["sensor_min"]
            self.sensor_max = bounds["sensor_max"]
        seed = raw_data[seed_record, :WINDOW_SIZE]
        self.anchor = seed[0].copy()
        self.sensor_window = seed[np.newaxis].copy()
        self.phase_window = np.arange(WINDOW_SIZE, dtype=np.int32)[np.newaxis]
        self.seed_min = seed.min(axis=0)
        self.seed_max = seed.max(axis=0)
        self.ps4_index = self.sensor_names.index("PS4")
        self._build_model_inputs = build_model_inputs

    def context(self) -> tuple[np.ndarray, np.ndarray]:
        return self.sensor_window[0].copy(), self.phase_window[0].copy()

    def predict_next(self) -> np.ndarray:
        inputs = self._build_model_inputs(
            self.sensor_window, self.phase_window, self.input_scaler
        )
        scaled_offset = self.model.predict(inputs, verbose=0)
        offset = self.offset_scaler.inverse_transform(scaled_offset)[0]
        next_sensor = self.anchor + offset
        next_phase = int((self.phase_window[0, -1] + 1) % SECONDS_PER_RECORD)
        if next_phase == 0:
            next_sensor = self.anchor.copy()
        next_sensor = np.minimum(np.maximum(next_sensor, self.sensor_min), self.sensor_max)
        next_sensor[self.ps4_index] = np.clip(
            next_sensor[self.ps4_index],
            self.seed_min[self.ps4_index],
            self.seed_max[self.ps4_index],
        )
        if not np.isfinite(next_sensor).all():
            raise ValueError("V5 baseline contains NaN or Inf")
        self.sensor_window = np.concatenate(
            [self.sensor_window[:, 1:], next_sensor[np.newaxis, np.newaxis]], axis=1
        )
        self.phase_window = np.concatenate(
            [self.phase_window[:, 1:], np.asarray([[next_phase]], dtype=np.int32)], axis=1
        )
        return next_sensor.astype(np.float64)
