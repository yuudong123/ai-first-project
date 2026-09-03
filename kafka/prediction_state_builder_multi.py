"""Build three-station state snapshots with the real HydroTwin model.

The Raw Multi-Station snapshot remains read-only.  Each station owns an
independent 20-sample rolling buffer.  Once a buffer is full, the builder
calculates the 17 ordered mean features and runs the integrated LightGBM
bundle for every newly observed one-second sample.
"""

from __future__ import annotations

import argparse
import json
import math
import os
import sys
import time
from collections import deque
from datetime import datetime, timezone
from pathlib import Path
from statistics import fmean
from typing import Any, Callable, Mapping


PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.hydrotwin_pipeline import (  # noqa: E402
    MEAN_FEATURE_COLUMNS,
    MODEL_PATH,
    SENSOR_NAMES,
    load_model_bundle,
    predict,
)


EQUIPMENT_IDS = ("station-01", "station-02", "station-03")
SENSOR_IDS = tuple(SENSOR_NAMES)
FEATURE_NAMES = tuple(MEAN_FEATURE_COLUMNS)
COMPONENT_IDS = ("cooler", "valve", "pump", "accumulator")
MODEL_IDS = (*COMPONENT_IDS, "stable_flag")
PREDICTION_WINDOW_SEC = 20
DEFAULT_POLL_INTERVAL_SEC = 1.0
SCRIPT_DIR = Path(__file__).resolve().parent
DEFAULT_INPUT_PATH = SCRIPT_DIR / "latest_raw_by_equipment.json"
DEFAULT_OUTPUT_PATH = SCRIPT_DIR / "latest_state_by_equipment.json"


def build_warming_up_prediction() -> dict[str, Any]:
    return {
        "status": "warming_up",
        "result": None,
        "window_sec": PREDICTION_WINDOW_SEC,
        "updated_at": None,
    }


def create_runtime_state() -> tuple[dict, dict, dict]:
    buffers = {
        station_id: deque(maxlen=PREDICTION_WINDOW_SEC)
        for station_id in EQUIPMENT_IDS
    }
    predictions = {
        station_id: build_warming_up_prediction()
        for station_id in EQUIPMENT_IDS
    }
    last_timestamps = {station_id: None for station_id in EQUIPMENT_IDS}
    return buffers, predictions, last_timestamps


def _parse_timestamp(value: Any, station_id: str) -> datetime:
    if not isinstance(value, str) or not value.strip():
        raise ValueError(f"Raw timestamp for {station_id} must be a string")
    normalized = value[:-1] + "+00:00" if value.endswith("Z") else value
    try:
        parsed = datetime.fromisoformat(normalized)
    except ValueError as error:
        raise ValueError(
            f"Raw timestamp for {station_id} is not ISO-8601: {value}"
        ) from error
    if parsed.tzinfo is None:
        return parsed.replace(tzinfo=timezone.utc)
    return parsed.astimezone(timezone.utc)


def _validated_sensors(value: Any, station_id: str) -> dict[str, float]:
    if not isinstance(value, dict) or set(value) != set(SENSOR_IDS):
        raise ValueError(
            f"Raw sensors for {station_id} must contain exactly all 17 IDs"
        )

    sensors: dict[str, float] = {}
    for sensor_id in SENSOR_IDS:
        sensor_value = value[sensor_id]
        if isinstance(sensor_value, bool):
            raise ValueError(f"Raw sensor {station_id}/{sensor_id} is not numeric")
        try:
            numeric_value = float(sensor_value)
        except (TypeError, ValueError) as error:
            raise ValueError(
                f"Raw sensor {station_id}/{sensor_id} is not numeric"
            ) from error
        if not math.isfinite(numeric_value):
            raise ValueError(f"Raw sensor {station_id}/{sensor_id} is not finite")
        sensors[sensor_id] = numeric_value
    return sensors


def read_raw_by_equipment(path: Path) -> dict[str, dict[str, Any]]:
    """Read and validate one complete three-station Raw snapshot."""
    with path.open("r", encoding="utf-8") as input_file:
        raw_by_equipment = json.load(input_file)

    if not isinstance(raw_by_equipment, dict):
        raise ValueError("Raw state root must be a JSON object")
    if set(raw_by_equipment) != set(EQUIPMENT_IDS):
        raise ValueError("Raw state must contain exactly station-01/02/03")

    validated = {}
    for station_id in EQUIPMENT_IDS:
        station = raw_by_equipment[station_id]
        if not isinstance(station, dict):
            raise ValueError(f"Raw state for {station_id} is invalid")
        timestamp = station.get("timestamp")
        _parse_timestamp(timestamp, station_id)
        validated[station_id] = {
            "timestamp": timestamp,
            "sensors": _validated_sensors(station.get("sensors"), station_id),
        }
    return validated


def validate_model_bundle(bundle: Mapping[str, Any]) -> dict[str, Any]:
    """Fail fast when the artifact does not match the realtime contract."""
    if bundle.get("model_type") != "LightGBM":
        raise ValueError("Model bundle type must be LightGBM")
    if int(bundle.get("window_sec", 0)) != PREDICTION_WINDOW_SEC:
        raise ValueError("Model bundle window_sec must be 20")
    if tuple(bundle.get("feature_names", ())) != FEATURE_NAMES:
        raise ValueError("Model bundle feature_names/order does not match 17 means")
    if tuple(bundle.get("target_order", ())) != MODEL_IDS:
        raise ValueError("Model bundle target_order does not match five outputs")
    models = bundle.get("models")
    if not isinstance(models, Mapping) or set(models) != set(MODEL_IDS):
        raise ValueError("Model bundle must contain exactly five models")
    return dict(bundle)


def build_mean_features(samples: deque[dict[str, float]]) -> dict[str, float]:
    if len(samples) != PREDICTION_WINDOW_SEC:
        raise ValueError("Mean features require exactly 20 samples")
    return {
        feature_name: fmean(sample[sensor_id] for sample in samples)
        for sensor_id, feature_name in zip(SENSOR_IDS, FEATURE_NAMES)
    }


def _validated_prediction_result(result: Any) -> dict[str, Any]:
    if not isinstance(result, Mapping):
        raise ValueError("Model prediction must be an object")
    if set(result) != {"stable_flag", "components"}:
        raise ValueError("Model prediction must contain stable_flag and components")
    components = result["components"]
    if not isinstance(components, Mapping) or set(components) != set(COMPONENT_IDS):
        raise ValueError("Model prediction components/order is invalid")
    return {
        "stable_flag": int(result["stable_flag"]),
        "components": {
            component_id: int(components[component_id])
            for component_id in COMPONENT_IDS
        },
    }


def update_predictions(
    raw_by_equipment: Mapping[str, Mapping[str, Any]],
    buffers: dict,
    predictions: dict,
    last_timestamps: dict,
    model_bundle: Mapping[str, Any],
    prediction_time: str,
    *,
    predict_fn: Callable[..., Mapping[str, Any]] = predict,
) -> tuple[str, ...]:
    """Consume each station's new timestamp and infer on full rolling windows."""
    updated_stations = []
    for station_id in EQUIPMENT_IDS:
        station = raw_by_equipment[station_id]
        timestamp = station["timestamp"]
        parsed_timestamp = _parse_timestamp(timestamp, station_id)
        previous_timestamp = last_timestamps[station_id]

        if previous_timestamp is not None:
            previous_parsed = _parse_timestamp(previous_timestamp, station_id)
            if parsed_timestamp <= previous_parsed:
                continue

        buffers[station_id].append(dict(station["sensors"]))
        last_timestamps[station_id] = timestamp

        if len(buffers[station_id]) < PREDICTION_WINDOW_SEC:
            predictions[station_id] = build_warming_up_prediction()
            continue

        features = build_mean_features(buffers[station_id])
        result = _validated_prediction_result(
            predict_fn(features, model_bundle=model_bundle)
        )
        predictions[station_id] = {
            "status": "ready",
            "result": result,
            "window_sec": PREDICTION_WINDOW_SEC,
            "updated_at": prediction_time,
        }
        updated_stations.append(station_id)
    return tuple(updated_stations)


def build_state_snapshot(raw_by_equipment: Mapping, predictions: Mapping) -> dict:
    """Copy Raw timestamp/sensors and add only the prediction field."""
    return {
        station_id: {
            "timestamp": raw_by_equipment[station_id]["timestamp"],
            "sensors": dict(raw_by_equipment[station_id]["sensors"]),
            "prediction": dict(predictions[station_id]),
        }
        for station_id in EQUIPMENT_IDS
    }


def atomic_write_json(path: Path, value: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary_path = path.with_name(path.name + ".tmp")
    with temporary_path.open("w", encoding="utf-8") as output_file:
        json.dump(value, output_file, ensure_ascii=False, indent=2)
        output_file.write("\n")
        output_file.flush()
        os.fsync(output_file.fileno())
    os.replace(temporary_path, path)


def current_timestamp() -> str:
    return datetime.now().astimezone().isoformat(timespec="milliseconds")


def run_builder(
    input_path: Path,
    output_path: Path,
    model_path: Path = MODEL_PATH,
    poll_interval: float = DEFAULT_POLL_INTERVAL_SEC,
    max_updates: int = 0,
    *,
    model_bundle: Mapping[str, Any] | None = None,
    model_loader: Callable[[Path], Mapping[str, Any]] = load_model_bundle,
    predict_fn: Callable[..., Mapping[str, Any]] = predict,
    monotonic_fn: Callable[[], float] = time.monotonic,
    sleep_fn: Callable[[float], None] = time.sleep,
    timestamp_fn: Callable[[], str] = current_timestamp,
) -> int:
    if poll_interval <= 0:
        raise ValueError("poll_interval must be positive")
    if max_updates < 0:
        raise ValueError("max_updates cannot be negative")
    if input_path.resolve() == output_path.resolve():
        raise ValueError("input_path and output_path must be different")

    bundle = validate_model_bundle(
        model_bundle if model_bundle is not None else model_loader(model_path)
    )
    buffers, predictions, last_timestamps = create_runtime_state()
    update_count = 0

    print("HydroTwin real AI Multi-Station state builder", flush=True)
    print(f"Model          : {model_path}", flush=True)
    print(f"Input (read only): {input_path}", flush=True)
    print(f"Output (atomic): {output_path}", flush=True)

    while True:
        cycle_started_at = monotonic_fn()
        try:
            raw_by_equipment = read_raw_by_equipment(input_path)
        except (FileNotFoundError, json.JSONDecodeError, OSError, ValueError) as error:
            print(
                f"Raw snapshot unavailable ({type(error).__name__}: {error}); "
                f"retrying in {poll_interval:g}s",
                flush=True,
            )
            sleep_fn(poll_interval)
            continue

        updated_stations = update_predictions(
            raw_by_equipment,
            buffers,
            predictions,
            last_timestamps,
            bundle,
            timestamp_fn(),
            predict_fn=predict_fn,
        )
        if updated_stations:
            print(
                "Prediction updated: " + ",".join(updated_stations),
                flush=True,
            )

        state = build_state_snapshot(raw_by_equipment, predictions)
        atomic_write_json(output_path, state)
        update_count += 1
        if max_updates and update_count >= max_updates:
            return update_count

        cycle_duration = monotonic_fn() - cycle_started_at
        sleep_fn(max(0.0, poll_interval - cycle_duration))


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", type=Path, default=DEFAULT_INPUT_PATH)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT_PATH)
    parser.add_argument("--model", type=Path, default=MODEL_PATH)
    parser.add_argument(
        "--poll-interval", type=float, default=DEFAULT_POLL_INTERVAL_SEC
    )
    parser.add_argument(
        "--max-updates",
        type=int,
        default=0,
        help="Stop after this many writes; zero runs until interrupted.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    try:
        run_builder(
            input_path=args.input,
            output_path=args.output,
            model_path=args.model,
            poll_interval=args.poll_interval,
            max_updates=args.max_updates,
        )
    except KeyboardInterrupt:
        print("\nMulti-Station prediction builder stopped by operator.", flush=True)


if __name__ == "__main__":
    main()
