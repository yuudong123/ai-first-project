"""V5 Virtual Factory Producer for permanent NORMAL and Drift Demo modes.

Runtime pipelines:

    normal-live: fixed V5 NORMAL model -> Kafka Raw topic
    drift-demo: fixed V5 NORMAL model -> temperature offset -> Kafka Raw topic

The drift is intentionally injected for this project demonstration and is not
a UCI label.  Kafka messages contain only timestamp and 17 sensor values; the
private scenario phase is written to a separate local ground-truth CSV.
"""

import argparse
import csv
import json
import os
import sys
import time
from datetime import datetime
from pathlib import Path

import joblib
import numpy as np
import tensorflow as tf
from kafka import KafkaProducer


PROJECT_ROOT = Path(__file__).resolve().parents[1]
SIMULATOR_SOURCE = PROJECT_ROOT / "src" / "simulator"
if str(SIMULATOR_SOURCE) not in sys.path:
    sys.path.insert(0, str(SIMULATOR_SOURCE))

from environment_scenario import (  # noqa: E402
    EnvironmentScenarioConfig,
    apply_temperature_offset,
    phase_and_temperature_offset,
)
from v5_generation_utils import (  # noqa: E402
    CYCLE_SECONDS,
    SENSOR_COUNT,
    WINDOW_SIZE,
    build_model_inputs,
)


DATA_DIR = PROJECT_ROOT / "data" / "processed" / "simulator"
MODEL_DIR = PROJECT_ROOT / "models" / "simulator"
RAW_FILE = DATA_DIR / "uci_1hz_17sensors.npz"
MODEL_FILE = MODEL_DIR / "virtual_factory_generator_v5.keras"
INPUT_SCALER_FILE = MODEL_DIR / "input_scaler_v5.joblib"
OFFSET_SCALER_FILE = MODEL_DIR / "offset_scaler_v5.joblib"
BOUNDS_FILE = MODEL_DIR / "sensor_bounds_v5.npz"
METADATA_FILE = MODEL_DIR / "generator_metadata_v5.json"

LIVE_OUTPUT_FILE = DATA_DIR / "v5_drift_live_300s.csv"
NORMAL_REFERENCE_FILE = DATA_DIR / "v5_normal_live_reference_300s.csv"
GROUND_TRUTH_FILE = DATA_DIR / "v5_drift_scenario_ground_truth.csv"

DEFAULT_BROKER = os.getenv("KAFKA_BROKER", "localhost:9092")
DEFAULT_TOPIC = os.getenv("KAFKA_TOPIC", "hydraulic.sensor.raw")
DEFAULT_INTERVAL = float(os.getenv("V5_SEND_INTERVAL_SEC", "1.0"))
TRAIN_RATIO = 0.8


class V5NormalRuntime:
    """One-step V5 inference state.  This class never calls model.fit()."""

    def __init__(
        self,
        model,
        input_scaler,
        offset_scaler,
        sensor_min,
        sensor_max,
        seed_window,
        ps4_index,
    ):
        self.model = model
        self.input_scaler = input_scaler
        self.offset_scaler = offset_scaler
        self.sensor_min = sensor_min
        self.sensor_max = sensor_max
        self.anchor = seed_window[0].copy()
        self.sensor_window = seed_window[np.newaxis, ...].copy()
        self.phase_window = np.arange(WINDOW_SIZE, dtype=np.int32)[
            np.newaxis, ...
        ]
        self.seed_min = seed_window.min(axis=0)
        self.seed_max = seed_window.max(axis=0)
        self.ps4_index = ps4_index

    def predict_next(self):
        model_input = build_model_inputs(
            self.sensor_window, self.phase_window, self.input_scaler
        )
        offset_scaled = self.model.predict(model_input, verbose=0)
        offset = self.offset_scaler.inverse_transform(offset_scaled)[0]
        next_sensor = self.anchor + offset
        next_phase = int((self.phase_window[0, -1] + 1) % CYCLE_SECONDS)

        if next_phase == 0:
            next_sensor = self.anchor.copy()
        next_sensor = np.minimum(
            np.maximum(next_sensor, self.sensor_min), self.sensor_max
        )
        next_sensor[self.ps4_index] = min(
            max(next_sensor[self.ps4_index], self.seed_min[self.ps4_index]),
            self.seed_max[self.ps4_index],
        )
        if not np.isfinite(next_sensor).all():
            raise ValueError("V5 NORMAL prediction contains NaN or Inf")

        self.sensor_window = np.concatenate(
            [self.sensor_window[:, 1:, :], next_sensor[np.newaxis, np.newaxis, :]],
            axis=1,
        )
        self.phase_window = np.concatenate(
            [
                self.phase_window[:, 1:],
                np.asarray([[next_phase]], dtype=np.int32),
            ],
            axis=1,
        )
        return next_sensor.astype(np.float64)


def parse_args():
    parser = argparse.ArgumentParser(
        description="HydroTwin V5 live environmental drift producer"
    )
    parser.add_argument(
        "--mode",
        choices=("normal-live", "drift-demo"),
        default="normal-live",
        help="normal-live runs forever; drift-demo keeps the finite scenario",
    )
    parser.add_argument("--broker", default=DEFAULT_BROKER)
    parser.add_argument("--topic", default=DEFAULT_TOPIC)
    parser.add_argument("--interval", type=float, default=DEFAULT_INTERVAL)
    parser.add_argument("--max-temp-offset", type=float, default=4.0)
    parser.add_argument("--normal-seconds", type=int, default=60)
    parser.add_argument("--drift-seconds", type=int, default=60)
    parser.add_argument("--hold-seconds", type=int, default=60)
    parser.add_argument("--recovery-seconds", type=int, default=60)
    parser.add_argument("--final-normal-seconds", type=int, default=60)
    parser.add_argument("--seed-record", type=int)
    return parser.parse_args()


def create_message(sensor_names, sensor_values):
    return {
        "timestamp": datetime.now().astimezone().isoformat(timespec="milliseconds"),
        "sensors": {
            sensor: round(float(value), 6)
            for sensor, value in zip(sensor_names, sensor_values)
        },
    }


def validate_raw_message(message, sensor_names):
    if set(message) != {"timestamp", "sensors"}:
        raise ValueError(f"Invalid Raw message keys: {list(message)}")
    if list(message["sensors"]) != sensor_names:
        raise ValueError("Raw message sensor names/order changed")
    forbidden = {
        "drift", "scenario", "phase", "normal", "fault", "label",
        "risk", "prediction", "confidence", "generator", "model", "features",
    }
    serialized = json.dumps(message).lower()
    if any(word in serialized for word in forbidden):
        raise ValueError("Raw Kafka message contains forbidden metadata")


def main():
    args = parse_args()
    if args.interval <= 0:
        raise ValueError("--interval must be greater than zero for LIVE mode")
    config = EnvironmentScenarioConfig(
        max_temperature_offset=args.max_temp_offset,
        normal_seconds=args.normal_seconds,
        drift_seconds=args.drift_seconds,
        hold_seconds=args.hold_seconds,
        recovery_seconds=args.recovery_seconds,
        final_normal_seconds=args.final_normal_seconds,
    )

    with np.load(RAW_FILE, allow_pickle=False) as npz_file:
        raw_data = npz_file["data"].astype(np.float32)
        sensor_names = [str(name) for name in npz_file["sensor_names"]]
    if raw_data.shape != (2205, 60, 17) or len(sensor_names) != SENSOR_COUNT:
        raise ValueError(f"Unexpected Raw dataset: {raw_data.shape}")

    validation_start = int(raw_data.shape[0] * TRAIN_RATIO)
    seed_record = validation_start if args.seed_record is None else args.seed_record
    if seed_record < validation_start or seed_record >= raw_data.shape[0]:
        raise ValueError(
            f"Seed must be in Validation records {validation_start}.."
            f"{raw_data.shape[0] - 1}"
        )
    seed_window = raw_data[seed_record, :WINDOW_SIZE, :]

    model = tf.keras.models.load_model(MODEL_FILE, compile=False)
    input_scaler = joblib.load(INPUT_SCALER_FILE)
    offset_scaler = joblib.load(OFFSET_SCALER_FILE)
    with np.load(BOUNDS_FILE, allow_pickle=False) as bounds_file:
        sensor_min = bounds_file["sensor_min"]
        sensor_max = bounds_file["sensor_max"]
    with open(METADATA_FILE, "r", encoding="utf-8") as metadata_file:
        metadata = json.load(metadata_file)
    if metadata.get("version") != "v5":
        raise ValueError("Unexpected V5 metadata version")
    if metadata.get("sensor_names") != sensor_names:
        raise ValueError("V5 metadata sensor names/order changed")
    if metadata.get("window_size") != WINDOW_SIZE:
        raise ValueError("V5 metadata window size changed")
    if metadata.get("cycle_seconds") != CYCLE_SECONDS:
        raise ValueError("V5 metadata cycle length changed")

    runtime = V5NormalRuntime(
        model,
        input_scaler,
        offset_scaler,
        sensor_min,
        sensor_max,
        seed_window,
        sensor_names.index("PS4"),
    )
    producer = KafkaProducer(
        bootstrap_servers=args.broker,
        value_serializer=lambda value: json.dumps(
            value, ensure_ascii=False
        ).encode("utf-8"),
        acks="all",
        retries=5,
    )

    if args.mode == "normal-live":
        print("=" * 90)
        print("HydroTwin V5 Virtual Factory - Permanent NORMAL LIVE")
        print("=" * 90)
        print(f"Broker / Topic : {args.broker} / {args.topic}")
        print(f"Seed Record    : {seed_record}")
        print(f"Interval       : {args.interval:.3f} sec")
        print("Duration       : unlimited (until process stop)")
        print("V5 Model       : loaded")
        print("Runtime        : load_model() + scaler load + predict() only")
        print("Kafka Schema   : timestamp + 17 sensors only")
        print("DRIFT STATUS   : OFF", flush=True)

        next_send_time = time.monotonic()
        elapsed_sec = 0
        try:
            while True:
                normal_values = runtime.predict_next()
                message = create_message(sensor_names, normal_values)
                validate_raw_message(message, sensor_names)
                producer.send(args.topic, value=message).get(timeout=10)
                values = message["sensors"]
                print(
                    "[sent %6d] TS1=%8.3f TS2=%8.3f PS1=%9.3f"
                    % (
                        elapsed_sec,
                        values["TS1"],
                        values["TS2"],
                        values["PS1"],
                    ),
                    flush=True,
                )

                elapsed_sec += 1
                next_send_time += args.interval
                sleep_seconds = next_send_time - time.monotonic()
                if sleep_seconds > 0:
                    time.sleep(sleep_seconds)
        except KeyboardInterrupt:
            print("\nV5 NORMAL LIVE stopped by operator.", flush=True)
        finally:
            producer.flush()
            producer.close()
        return

    DATA_DIR.mkdir(parents=True, exist_ok=True)
    live_file = open(LIVE_OUTPUT_FILE, "w", newline="", encoding="utf-8")
    normal_file = open(
        NORMAL_REFERENCE_FILE, "w", newline="", encoding="utf-8"
    )
    truth_file = open(GROUND_TRUTH_FILE, "w", newline="", encoding="utf-8")
    live_writer = csv.writer(live_file)
    normal_writer = csv.writer(normal_file)
    truth_writer = csv.writer(truth_file)
    live_writer.writerow(["elapsed_sec", "timestamp"] + sensor_names)
    normal_writer.writerow(["elapsed_sec"] + sensor_names)
    truth_writer.writerow(["elapsed_sec", "phase", "temperature_offset"])

    print("=" * 90)
    print("HydroTwin V5 Virtual Factory + Environment Drift LIVE")
    print("=" * 90)
    print(f"Broker / Topic : {args.broker} / {args.topic}")
    print(f"Seed Record    : {seed_record}")
    print(f"Interval       : {args.interval:.3f} sec")
    print(f"Total Seconds  : {config.total_seconds}")
    print(f"Max Temp Offset: +{config.max_temperature_offset:.3f} C")
    print("Runtime        : load_model() + predict() only")
    print("Kafka Schema   : timestamp + sensors only")

    next_send_time = time.monotonic()
    last_phase = None
    try:
        for elapsed_sec in range(config.total_seconds):
            normal_values = runtime.predict_next()
            phase, offset = phase_and_temperature_offset(elapsed_sec, config)
            drift_values = apply_temperature_offset(
                normal_values, sensor_names, offset
            )
            message = create_message(sensor_names, drift_values)
            validate_raw_message(message, sensor_names)

            producer.send(args.topic, value=message).get(timeout=10)
            sent_values = [message["sensors"][sensor] for sensor in sensor_names]
            normal_rounded = [round(float(value), 6) for value in normal_values]
            live_writer.writerow(
                [elapsed_sec, message["timestamp"]] + sent_values
            )
            normal_writer.writerow([elapsed_sec] + normal_rounded)
            truth_writer.writerow([elapsed_sec, phase, f"{offset:.9f}"])
            live_file.flush()
            normal_file.flush()
            truth_file.flush()

            if phase != last_phase:
                print(f"[PHASE] elapsed={elapsed_sec:3d} {phase}")
                last_phase = phase
            print(
                f"[sent {elapsed_sec:3d}] offset={offset:6.3f} "
                f"TS1={message['sensors']['TS1']:8.3f} "
                f"TS2={message['sensors']['TS2']:8.3f} "
                f"PS1={message['sensors']['PS1']:9.3f}"
            )

            if elapsed_sec < config.total_seconds - 1:
                next_send_time += args.interval
                sleep_seconds = next_send_time - time.monotonic()
                if sleep_seconds > 0:
                    time.sleep(sleep_seconds)
    finally:
        producer.flush()
        producer.close()
        live_file.close()
        normal_file.close()
        truth_file.close()

    print("=" * 90)
    print("V5 LIVE SCENARIO COMPLETE")
    print(f"[SAVED] {LIVE_OUTPUT_FILE}")
    print(f"[SAVED] {NORMAL_REFERENCE_FILE}")
    print(f"[SAVED] {GROUND_TRUTH_FILE}")


if __name__ == "__main__":
    main()
