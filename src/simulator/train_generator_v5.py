"""Train the phase-aware, anchor-stable HydroTwin Generator V5.

V5 is based on the verified Raw result: several sensors have a repeatable
shape at fixed positions inside each independent 60-second Record.  TS1, for
example, has its mean minimum at second 12 and maximum at second 40, while its
mean end-start difference is only about 0.016.

Design compared with V1-V4:

* Records remain independent; no two Raw Records are joined for training.
* A 30-second LSTM window receives the 17 scaled sensors plus cycle position.
* The target is the next value's offset from the same Record's first value.
* Circular windows stay inside one Record and teach the real 60-to-1 reset.
* Generation restores an offset around one fixed seed anchor, so prediction
  errors cannot accumulate forever as they can in recursive Delta generation.

Only the UCI Raw NPZ is read.  No generated data, label, or existing model is
used as training data.
"""

import json
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
import tensorflow as tf
from sklearn.preprocessing import StandardScaler
from tensorflow.keras import Sequential
from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint
from tensorflow.keras.layers import Dense, Input, LSTM

from v5_generation_utils import CYCLE_SECONDS, SENSOR_COUNT, WINDOW_SIZE


PROJECT_ROOT = Path(__file__).resolve().parents[2]
DATA_FILE = (
    PROJECT_ROOT
    / "data"
    / "processed"
    / "simulator"
    / "uci_1hz_17sensors.npz"
)
MODEL_DIR = PROJECT_ROOT / "models" / "simulator"
MODEL_FILE = MODEL_DIR / "virtual_factory_generator_v5.keras"
INPUT_SCALER_FILE = MODEL_DIR / "input_scaler_v5.joblib"
OFFSET_SCALER_FILE = MODEL_DIR / "offset_scaler_v5.joblib"
BOUNDS_FILE = MODEL_DIR / "sensor_bounds_v5.npz"
METADATA_FILE = MODEL_DIR / "generator_metadata_v5.json"
HISTORY_FILE = (
    PROJECT_ROOT
    / "data"
    / "processed"
    / "simulator"
    / "training_history_v5.csv"
)

EXPECTED_SHAPE = (2205, 60, 17)
TRAIN_RATIO = 0.8
EPOCHS = 50
BATCH_SIZE = 256
RANDOM_SEED = 42


np.random.seed(RANDOM_SEED)
tf.random.set_seed(RANDOM_SEED)


def load_raw_data():
    with np.load(DATA_FILE, allow_pickle=False) as npz_file:
        data = npz_file["data"].astype(np.float32)
        sensor_names = [str(name) for name in npz_file["sensor_names"]]
    if data.shape != EXPECTED_SHAPE:
        raise ValueError(f"Expected {EXPECTED_SHAPE}, actual={data.shape}")
    if len(sensor_names) != SENSOR_COUNT:
        raise ValueError(f"Expected {SENSOR_COUNT} sensors: {sensor_names}")
    if not np.isfinite(data).all():
        raise ValueError("Raw data contains NaN or Inf")
    return data, sensor_names


def create_circular_windows(records_raw, input_scaler, offset_scaler):
    """Create 60 phase-aware windows per Record without joining Records."""
    record_count = records_raw.shape[0]
    window_count = record_count * CYCLE_SECONDS
    x_data = np.empty(
        (window_count, WINDOW_SIZE, SENSOR_COUNT + 1), dtype=np.float32
    )
    target_offsets = np.empty((window_count, SENSOR_COUNT), dtype=np.float32)

    scaled_records = input_scaler.transform(
        records_raw.reshape(-1, SENSOR_COUNT)
    ).reshape(records_raw.shape)

    row = 0
    for record_index in range(record_count):
        record_raw = records_raw[record_index]
        record_scaled = scaled_records[record_index]
        anchor = record_raw[0]

        for target_phase in range(CYCLE_SECONDS):
            input_phases = (
                np.arange(target_phase - WINDOW_SIZE, target_phase)
                % CYCLE_SECONDS
            )
            x_data[row, :, :SENSOR_COUNT] = record_scaled[input_phases]
            x_data[row, :, SENSOR_COUNT] = (
                input_phases.astype(np.float32) / float(CYCLE_SECONDS - 1)
            )
            target_offsets[row] = record_raw[target_phase] - anchor
            row += 1

    y_data = offset_scaler.transform(target_offsets).astype(np.float32)
    return x_data, y_data


def main():
    print("=" * 78)
    print("HydroTwin Generator V5 - Phase-aware Anchored Offset LSTM")
    print("=" * 78)

    data, sensor_names = load_raw_data()
    train_count = int(data.shape[0] * TRAIN_RATIO)
    train_raw = data[:train_count]
    validation_raw = data[train_count:]

    print(f"Dataset Shape      : {data.shape}")
    print(f"Sensor Count       : {len(sensor_names)}")
    print(f"Train Records      : {train_raw.shape}")
    print(f"Validation Records : {validation_raw.shape}")
    print(f"Window Size        : {WINDOW_SIZE}")
    print(f"Cycle Seconds      : {CYCLE_SECONDS}")
    print("Training Source    : UCI Raw 17 Sensors Only")

    input_scaler = StandardScaler()
    input_scaler.fit(train_raw.reshape(-1, SENSOR_COUNT))

    train_offsets = train_raw - train_raw[:, 0:1, :]
    offset_scaler = StandardScaler()
    offset_scaler.fit(train_offsets.reshape(-1, SENSOR_COUNT))

    print("Creating independent circular Record windows...")
    x_train, y_train = create_circular_windows(
        train_raw, input_scaler, offset_scaler
    )
    x_validation, y_validation = create_circular_windows(
        validation_raw, input_scaler, offset_scaler
    )

    print(f"X Train            : {x_train.shape}")
    print(f"Y Train            : {y_train.shape}")
    print(f"X Validation       : {x_validation.shape}")
    print(f"Y Validation       : {y_validation.shape}")

    model = Sequential(
        [
            Input(shape=(WINDOW_SIZE, SENSOR_COUNT + 1)),
            LSTM(64),
            Dense(64, activation="relu"),
            Dense(SENSOR_COUNT),
        ]
    )
    model.compile(optimizer="adam", loss="mse", metrics=["mae"])

    print("\n[V5 MODEL SUMMARY]")
    model.summary()
    MODEL_DIR.mkdir(parents=True, exist_ok=True)
    HISTORY_FILE.parent.mkdir(parents=True, exist_ok=True)

    callbacks = [
        EarlyStopping(
            monitor="val_loss",
            patience=7,
            restore_best_weights=True,
            verbose=1,
        ),
        ModelCheckpoint(
            MODEL_FILE,
            monitor="val_loss",
            save_best_only=True,
            verbose=1,
        ),
    ]

    print("\n[V5 TRAINING START]")
    history = model.fit(
        x_train,
        y_train,
        validation_data=(x_validation, y_validation),
        epochs=EPOCHS,
        batch_size=BATCH_SIZE,
        callbacks=callbacks,
        verbose=2,
        shuffle=True,
    )

    validation_mse, validation_mae = model.evaluate(
        x_validation, y_validation, verbose=0
    )
    print("\n[V5 VALIDATION]")
    print(f"Validation MSE     : {validation_mse:.9f}")
    print(f"Validation MAE     : {validation_mae:.9f}")

    joblib.dump(input_scaler, INPUT_SCALER_FILE)
    joblib.dump(offset_scaler, OFFSET_SCALER_FILE)
    np.savez(
        BOUNDS_FILE,
        sensor_min=train_raw.reshape(-1, SENSOR_COUNT).min(axis=0),
        sensor_max=train_raw.reshape(-1, SENSOR_COUNT).max(axis=0),
    )
    model.save(MODEL_FILE)

    history_frame = pd.DataFrame(history.history)
    history_frame.insert(0, "epoch", np.arange(1, len(history_frame) + 1))
    history_frame.to_csv(HISTORY_FILE, index=False)

    best_epoch = int(history_frame["val_loss"].idxmin()) + 1
    metadata = {
        "version": "v5",
        "design": "phase_aware_anchored_offset_lstm",
        "sensor_names": sensor_names,
        "sensor_count": SENSOR_COUNT,
        "window_size": WINDOW_SIZE,
        "cycle_seconds": CYCLE_SECONDS,
        "input_feature_count": SENSOR_COUNT + 1,
        "target": "next_sensor_offset_from_same_record_start",
        "training_records": int(train_raw.shape[0]),
        "validation_records": int(validation_raw.shape[0]),
        "training_windows": int(x_train.shape[0]),
        "validation_windows": int(x_validation.shape[0]),
        "epochs_trained": int(len(history_frame)),
        "best_epoch": best_epoch,
        "validation_mse_scaled": float(validation_mse),
        "validation_mae_scaled": float(validation_mae),
        "training_source": "UCI Raw 17 Sensors Only",
        "generated_data_used_for_training": False,
        "records_joined_for_training": False,
        "sensor_bounds_source": "UCI Raw training records only",
        "seed_local_bound_sensors": ["PS4"],
        "raw_pattern_basis": {
            "ts1_position_min_second": 12,
            "ts1_position_max_second": 40,
            "ts1_position_mean_range": 0.6056730102,
            "ts1_end_minus_start_mean": 0.0160344373,
        },
    }
    with open(METADATA_FILE, "w", encoding="utf-8") as metadata_file:
        json.dump(metadata, metadata_file, ensure_ascii=False, indent=2)

    sample_index = 30
    sample_scaled = model.predict(
        x_validation[sample_index : sample_index + 1], verbose=0
    )
    sample_offset = offset_scaler.inverse_transform(sample_scaled)[0]
    sample_target_phase = 30
    sample_anchor = validation_raw[0, 0]
    sample_generated = sample_anchor + sample_offset
    sample_actual = validation_raw[0, sample_target_phase]

    print("\n[ACTUAL SENSOR VS V5 SAMPLE - VALIDATION RECORD 1765, SECOND 31]")
    for sensor, actual, generated in zip(
        sensor_names, sample_actual, sample_generated
    ):
        print(f"{sensor:5s} actual={actual:12.6f} v5={generated:12.6f}")

    for output_path in [
        MODEL_FILE,
        INPUT_SCALER_FILE,
        OFFSET_SCALER_FILE,
        BOUNDS_FILE,
        METADATA_FILE,
        HISTORY_FILE,
    ]:
        print(f"[SAVED] {output_path}")


if __name__ == "__main__":
    main()
