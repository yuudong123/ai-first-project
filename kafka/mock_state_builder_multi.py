"""Build the latest integrated state for all HydroTwin stations.

The Raw Multi-Station file is treated as read-only.  This process copies its
timestamp and sensors, adds independently tracked mock predictions, and writes
the resulting snapshot atomically.
"""

from __future__ import annotations

import argparse
import json
import os
import time
from datetime import datetime
from pathlib import Path
from typing import Callable


EQUIPMENT_IDS = ("station-01", "station-02", "station-03")
SENSOR_IDS = (
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
)
PREDICTION_WINDOW_SEC = 10
DEFAULT_POLL_INTERVAL_SEC = 1.0
SCRIPT_DIR = Path(__file__).resolve().parent
DEFAULT_INPUT_PATH = SCRIPT_DIR / "latest_raw_by_equipment.json"
DEFAULT_OUTPUT_PATH = SCRIPT_DIR / "latest_state_by_equipment.json"
MOCK_NOTICE = "MOCK PREDICTION - UI/Unity integration test only"

# MOCK PREDICTION - UI/Unity integration test only.
# One entry represents one ready 10-second block.  Replacing
# build_mock_prediction() is sufficient when the real model is connected.
MOCK_RESULT_PATTERNS = (
    {
        "station-01": "normal",
        "station-02": "abnormal",
        "station-03": "normal",
    },
    {
        "station-01": "abnormal",
        "station-02": "normal",
        "station-03": "normal",
    },
    {
        "station-01": "normal",
        "station-02": "normal",
        "station-03": "abnormal",
    },
)


def build_mock_prediction(
    station_id: str, block_index: int, prediction_time: str
) -> dict:
    """Return a UI-only prediction for one station and ready-time block."""
    if station_id not in EQUIPMENT_IDS:
        raise ValueError(f"Unknown station: {station_id}")
    if block_index < 0:
        raise ValueError("block_index cannot be negative")

    pattern = MOCK_RESULT_PATTERNS[block_index % len(MOCK_RESULT_PATTERNS)]
    return {
        "status": "ready",
        "result": pattern[station_id],
        "window_sec": PREDICTION_WINDOW_SEC,
        "updated_at": prediction_time,
    }


def build_warming_up_prediction() -> dict:
    return {
        "status": "warming_up",
        "result": None,
        "window_sec": PREDICTION_WINDOW_SEC,
        "updated_at": None,
    }


def create_prediction_state() -> tuple[dict, dict]:
    predictions = {
        station_id: build_warming_up_prediction()
        for station_id in EQUIPMENT_IDS
    }
    last_blocks = {station_id: None for station_id in EQUIPMENT_IDS}
    return predictions, last_blocks


def update_predictions(
    predictions: dict,
    last_blocks: dict,
    elapsed_seconds: float,
    prediction_time: str,
) -> tuple[str, ...]:
    """Update only stations whose independent 10-second block has changed."""
    if elapsed_seconds < PREDICTION_WINDOW_SEC:
        return ()

    block_index = int(elapsed_seconds // PREDICTION_WINDOW_SEC) - 1
    updated_stations = []
    for station_id in EQUIPMENT_IDS:
        if last_blocks[station_id] == block_index:
            continue
        predictions[station_id] = build_mock_prediction(
            station_id, block_index, prediction_time
        )
        last_blocks[station_id] = block_index
        updated_stations.append(station_id)
    return tuple(updated_stations)


def read_raw_by_equipment(path: Path) -> dict:
    """Read and validate a complete three-station Raw snapshot."""
    with path.open("r", encoding="utf-8") as input_file:
        raw_by_equipment = json.load(input_file)

    if not isinstance(raw_by_equipment, dict):
        raise ValueError("Raw state root must be a JSON object")

    for station_id in EQUIPMENT_IDS:
        station = raw_by_equipment.get(station_id)
        if not isinstance(station, dict):
            raise ValueError(f"Raw state is missing {station_id}")
        if "timestamp" not in station or not isinstance(station.get("sensors"), dict):
            raise ValueError(f"Raw state for {station_id} is invalid")
        if set(station["sensors"]) != set(SENSOR_IDS):
            raise ValueError(
                f"Raw sensors for {station_id} must contain exactly all 17 IDs"
            )

    return raw_by_equipment


def build_state_snapshot(raw_by_equipment: dict, predictions: dict) -> dict:
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
    """Write through ``<filename>.tmp`` and atomically replace the target."""
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
    poll_interval: float = DEFAULT_POLL_INTERVAL_SEC,
    max_updates: int = 0,
    *,
    monotonic_fn: Callable[[], float] = time.monotonic,
    sleep_fn: Callable[[float], None] = time.sleep,
    timestamp_fn: Callable[[], str] = current_timestamp,
) -> int:
    """Continuously copy Raw state and refresh mock prediction windows."""
    if poll_interval <= 0:
        raise ValueError("poll_interval must be positive")
    if max_updates < 0:
        raise ValueError("max_updates cannot be negative")
    if input_path.resolve() == output_path.resolve():
        raise ValueError("input_path and output_path must be different")

    predictions, last_blocks = create_prediction_state()
    started_at = monotonic_fn()
    update_count = 0

    print(MOCK_NOTICE, flush=True)
    print(f"Input  (read only): {input_path}", flush=True)
    print(f"Output (atomic)   : {output_path}", flush=True)

    while True:
        cycle_started_at = monotonic_fn()
        prediction_time = timestamp_fn()
        updated_stations = update_predictions(
            predictions,
            last_blocks,
            cycle_started_at - started_at,
            prediction_time,
        )
        if updated_stations:
            block_number = next(
                last_blocks[station_id] for station_id in updated_stations
            )
            print(
                f"{MOCK_NOTICE}: block={block_number} "
                f"stations={','.join(updated_stations)} updated_at={prediction_time}",
                flush=True,
            )

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
            poll_interval=args.poll_interval,
            max_updates=args.max_updates,
        )
    except KeyboardInterrupt:
        print("\nMulti-Station state builder stopped by operator.", flush=True)


if __name__ == "__main__":
    main()
