#!/usr/bin/env python3
"""Queue manual drift commands and display producer drift status."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Mapping


PROJECT_ROOT = Path(__file__).resolve().parents[1]
SIMULATOR_SOURCE = PROJECT_ROOT / "src" / "simulator"
if str(SIMULATOR_SOURCE) not in sys.path:
    sys.path.insert(0, str(SIMULATOR_SOURCE))

from drift_injector import enqueue_control_command  # noqa: E402


DEFAULT_CONTROL_FILE = PROJECT_ROOT / "kafka" / "run" / "drift_control.json"
DEFAULT_STATUS_FILE = PROJECT_ROOT / "kafka" / "run" / "drift_status.json"
EQUIPMENT_IDS = ("station-01", "station-02", "station-03")
TARGETS = (*EQUIPMENT_IDS, "all")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--control-file", type=Path, default=DEFAULT_CONTROL_FILE)
    parser.add_argument("--status-file", type=Path, default=DEFAULT_STATUS_FILE)
    subparsers = parser.add_subparsers(dest="command", required=True)
    for action in ("trigger", "reset"):
        action_parser = subparsers.add_parser(action)
        action_parser.add_argument("target", choices=TARGETS)
    subparsers.add_parser("status")
    return parser.parse_args()


def read_status(path: Path) -> Mapping:
    with path.open(encoding="utf-8") as input_file:
        payload = json.load(input_file)
    if not isinstance(payload, dict) or not isinstance(payload.get("stations"), dict):
        raise ValueError("Status JSON must contain a stations object")
    return payload


def format_status(payload: Mapping) -> str:
    stations = payload.get("stations")
    if not isinstance(stations, Mapping):
        raise ValueError("Status payload does not contain stations")
    lines = []
    for equipment_id in EQUIPMENT_IDS:
        station = stations.get(equipment_id)
        if not isinstance(station, Mapping):
            raise ValueError(f"Status is missing {equipment_id}")
        state = station.get("state")
        temperature_offset = station.get(
            "temperature_offset", station.get("offset")
        )
        progress = station.get("progress", 0.0)
        pressure_percent = station.get("pressure_percent", 0.0)
        target_temp_offset = station.get("target_temp_offset", 0.0)
        target_pressure_percent = station.get("target_pressure_percent", 0.0)
        event_duration_sec = station.get("event_duration_sec", 0.0)
        if not isinstance(state, str) or not all(
            isinstance(value, (int, float))
            for value in (
                progress,
                temperature_offset,
                pressure_percent,
                target_temp_offset,
                target_pressure_percent,
                event_duration_sec,
            )
        ):
            raise ValueError(f"Status is invalid for {equipment_id}")
        line = (
            f"{equipment_id}  {state:<9}  progress={float(progress):.3f} "
            f"temp_offset={float(temperature_offset):.3f}C "
            f"pressure={float(pressure_percent):.3f}%"
        )
        if state != "NORMAL":
            line += (
                f" target_temp={float(target_temp_offset):.3f}C"
                f" target_pressure={float(target_pressure_percent):.3f}%"
                f" duration={float(event_duration_sec):.3f}s"
            )
        lines.append(line)
    return "\n".join(lines)


def main() -> int:
    args = parse_args()
    if args.command == "status":
        try:
            print(format_status(read_status(args.status_file)))
        except FileNotFoundError:
            print(
                f"No drift status found at {args.status_file}; is the producer running?",
                file=sys.stderr,
            )
            return 1
        except (OSError, ValueError, json.JSONDecodeError) as error:
            print(f"Cannot read drift status safely: {error}", file=sys.stderr)
            return 1
        return 0

    try:
        command_id = enqueue_control_command(
            args.control_file, args.command, args.target
        )
    except (OSError, ValueError, json.JSONDecodeError, TimeoutError) as error:
        print(f"Cannot queue drift command safely: {error}", file=sys.stderr)
        return 1
    print(f"queued {args.command} {args.target} command_id={command_id}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
