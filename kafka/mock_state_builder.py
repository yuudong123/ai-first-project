"""Build latest_state.json from live Raw sensors and a temporary prediction.

MOCK PREDICTION - 실제 AI 모델 연결 전 UI 통합 테스트용

The Raw input is read-only.  Replace ``build_mock_prediction`` with real
model inference later while keeping the state assembly and atomic writer.
"""

from __future__ import annotations

import argparse
import json
import logging
import os
import signal
import time
from datetime import datetime
from pathlib import Path
from typing import Any


KAFKA_DIR = Path(__file__).resolve().parent
DEFAULT_RAW_FILE = KAFKA_DIR / "latest_raw.json"
DEFAULT_STATE_FILE = KAFKA_DIR / "latest_state.json"
DEFAULT_LOG_FILE = KAFKA_DIR / "logs" / "mock_state_builder.log"
WINDOW_SECONDS = 10
SENSOR_COUNT = 17
MOCK_NOTICE = "MOCK PREDICTION - 실제 AI 모델 연결 전 UI 통합 테스트용"


class RawStateError(ValueError):
    """Raised when the current Raw snapshot cannot produce a valid state."""


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--raw-file", type=Path, default=DEFAULT_RAW_FILE)
    parser.add_argument("--output", type=Path, default=DEFAULT_STATE_FILE)
    parser.add_argument("--log-file", type=Path, default=DEFAULT_LOG_FILE)
    parser.add_argument("--interval", type=float, default=1.0)
    parser.add_argument(
        "--max-seconds",
        type=float,
        default=0.0,
        help="Validation duration; zero runs until stopped",
    )
    return parser.parse_args()


def configure_logging(log_file: Path) -> logging.Logger:
    log_file.parent.mkdir(parents=True, exist_ok=True)
    logger = logging.getLogger("hydrotwin.mock_state_builder")
    logger.setLevel(logging.INFO)
    logger.handlers.clear()
    formatter = logging.Formatter(
        "%(asctime)s %(levelname)s %(message)s",
        datefmt="%Y-%m-%dT%H:%M:%S%z",
    )
    for handler in (logging.StreamHandler(), logging.FileHandler(log_file)):
        handler.setFormatter(formatter)
        logger.addHandler(handler)
    return logger


def judgment_timestamp() -> str:
    return datetime.now().astimezone().isoformat(timespec="milliseconds")


def build_mock_prediction(window_index: int, updated_at: str | None) -> dict:
    """Return the temporary UI prediction for one ten-second window."""
    if window_index <= 0:
        return {
            "status": "warming_up",
            "result": None,
            "window_sec": WINDOW_SECONDS,
            "updated_at": None,
        }
    return {
        "status": "ready",
        "result": "normal" if window_index % 2 == 1 else "abnormal",
        "window_sec": WINDOW_SECONDS,
        "updated_at": updated_at,
    }


def read_latest_raw(path: Path) -> dict:
    with path.open(encoding="utf-8") as raw_file:
        raw = json.load(raw_file)
    if not isinstance(raw, dict):
        raise RawStateError("latest_raw.json root must be an object")
    if "timestamp" not in raw or "sensors" not in raw:
        raise RawStateError("latest_raw.json requires timestamp and sensors")
    if not isinstance(raw["timestamp"], str):
        raise RawStateError("latest_raw.json timestamp must be a string")
    if not isinstance(raw["sensors"], dict):
        raise RawStateError("latest_raw.json sensors must be an object")
    if len(raw["sensors"]) != SENSOR_COUNT:
        raise RawStateError(
            f"latest_raw.json requires {SENSOR_COUNT} sensors, "
            f"actual={len(raw['sensors'])}"
        )
    return raw


def build_latest_state(raw: dict, prediction: dict) -> dict[str, Any]:
    """Copy Raw timestamp/sensors unchanged and append only prediction."""
    return {
        "timestamp": raw["timestamp"],
        "sensors": raw["sensors"],
        "prediction": prediction,
    }


def atomic_write_json(path: Path, value: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary_path = path.with_name(path.name + ".tmp")
    with temporary_path.open("w", encoding="utf-8") as output_file:
        json.dump(value, output_file, ensure_ascii=False, indent=2)
        output_file.flush()
        os.fsync(output_file.fileno())
    os.replace(temporary_path, path)


def run(args: argparse.Namespace) -> None:
    if args.interval <= 0:
        raise ValueError("--interval must be greater than zero")
    if args.max_seconds < 0:
        raise ValueError("--max-seconds cannot be negative")

    logger = configure_logging(args.log_file)
    stop_requested = False

    def request_stop(signum, _frame) -> None:
        nonlocal stop_requested
        stop_requested = True
        logger.info("Stop requested by signal %s", signum)

    signal.signal(signal.SIGINT, request_stop)
    signal.signal(signal.SIGTERM, request_stop)

    started_at = time.monotonic()
    next_update_at = started_at
    prediction_window = 0
    prediction = build_mock_prediction(prediction_window, None)
    logger.info(MOCK_NOTICE)
    logger.info(
        "Started raw=%s output=%s sensor_interval=%.3fs prediction_window=%ds",
        args.raw_file,
        args.output,
        args.interval,
        WINDOW_SECONDS,
    )
    logger.info("Prediction warming_up result=null updated_at=null")

    while not stop_requested:
        elapsed = time.monotonic() - started_at
        if args.max_seconds and elapsed > args.max_seconds:
            break

        current_window = int(elapsed // WINDOW_SECONDS)
        if current_window != prediction_window:
            prediction_window = current_window
            prediction = build_mock_prediction(
                prediction_window, judgment_timestamp()
            )
            logger.info(
                "Prediction updated status=%s result=%s updated_at=%s",
                prediction["status"],
                prediction["result"],
                prediction["updated_at"],
            )

        try:
            raw = read_latest_raw(args.raw_file)
            state = build_latest_state(raw, prediction)
            atomic_write_json(args.output, state)
            logger.info(
                "State updated raw_timestamp=%s prediction=%s prediction_updated_at=%s",
                state["timestamp"],
                prediction["status"],
                prediction["updated_at"],
            )
        except (FileNotFoundError, json.JSONDecodeError, OSError, RawStateError) as error:
            logger.warning("Raw read/state write failed; retrying: %s", error)

        next_update_at += args.interval
        sleep_seconds = next_update_at - time.monotonic()
        if sleep_seconds > 0:
            time.sleep(sleep_seconds)

    logger.info("Stopped after %.3f seconds", time.monotonic() - started_at)


def main() -> None:
    run(parse_args())


if __name__ == "__main__":
    main()
