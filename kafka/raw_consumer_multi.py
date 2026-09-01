"""Consume Multi-Station Raw messages into one latest-state JSON file."""

from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path

from kafka import KafkaConsumer


PROJECT_ROOT = Path(__file__).resolve().parents[1]
SIMULATOR_SOURCE = PROJECT_ROOT / "src" / "simulator"
if str(SIMULATOR_SOURCE) not in sys.path:
    sys.path.insert(0, str(SIMULATOR_SOURCE))

from v5_multi_station_utils import (  # noqa: E402
    EQUIPMENT_IDS,
    validate_multi_raw_message,
)


DEFAULT_BROKER = os.getenv("KAFKA_BROKER", "localhost:9092")
DEFAULT_TOPIC = os.getenv("KAFKA_MULTI_TOPIC", "hydraulic.sensor.multi.raw")
LATEST_FILE = Path(__file__).resolve().parent / "latest_raw_by_equipment.json"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--broker", default=DEFAULT_BROKER)
    parser.add_argument("--topic", default=DEFAULT_TOPIC)
    parser.add_argument("--output", type=Path, default=LATEST_FILE)
    parser.add_argument(
        "--group-id", default="hydraulic-raw-multi-consumer"
    )
    parser.add_argument("--max-messages", type=int, default=0)
    parser.add_argument("--quiet", action="store_true")
    return parser.parse_args()


def save_latest_by_equipment(path: Path, states: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary_path = path.with_name(path.name + ".tmp")
    with temporary_path.open("w", encoding="utf-8") as output_file:
        json.dump(states, output_file, ensure_ascii=False, indent=2)
        output_file.flush()
        os.fsync(output_file.fileno())
    os.replace(temporary_path, path)


def update_latest(states: dict, message: dict, sensor_names: list[str]) -> dict:
    validate_multi_raw_message(message, sensor_names)
    equipment_id = message["equipment_id"]
    states[equipment_id] = {
        "timestamp": message["timestamp"],
        "sensors": message["sensors"],
    }
    return states


def main() -> None:
    args = parse_args()
    if args.max_messages < 0:
        raise ValueError("--max-messages cannot be negative")
    sensor_names = [
        "PS1", "PS2", "PS3", "PS4", "PS5", "PS6", "EPS1", "FS1",
        "FS2", "TS1", "TS2", "TS3", "TS4", "VS1", "CE", "CP", "SE",
    ]
    states = {}
    consumer = KafkaConsumer(
        args.topic,
        bootstrap_servers=args.broker,
        auto_offset_reset="latest",
        enable_auto_commit=True,
        group_id=args.group_id,
        value_deserializer=lambda value: json.loads(value.decode("utf-8")),
    )
    print("=" * 76)
    print("HydroTwin Multi-Station Raw Consumer")
    print(f"Broker / Topic: {args.broker} / {args.topic}")
    print(f"Output        : {args.output}")
    print(f"Equipment     : {', '.join(EQUIPMENT_IDS)}")
    print("=" * 76, flush=True)

    count = 0
    try:
        for record in consumer:
            message = record.value
            update_latest(states, message, sensor_names)
            save_latest_by_equipment(args.output, states)
            count += 1
            if not args.quiet:
                sensors = message["sensors"]
                print(
                    f"[receive {count:6d}] {message['timestamp']} "
                    f"{message['equipment_id']} PS1={sensors['PS1']} "
                    f"FS1={sensors['FS1']} TS1={sensors['TS1']} "
                    f"VS1={sensors['VS1']}",
                    flush=True,
                )
            if args.max_messages and count >= args.max_messages:
                break
    except KeyboardInterrupt:
        print("\nMulti-Station Raw Consumer stopped by operator.", flush=True)
    finally:
        consumer.close()
    print(f"Messages consumed: {count}", flush=True)


if __name__ == "__main__":
    main()
