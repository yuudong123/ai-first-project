"""Runtime helpers for three independent V5 virtual factory stations.

This module only performs inference with an already-loaded V5 model.  Each
``V5StationRuntime`` owns its anchor, rolling window, cycle position, and
current sensor values; the model and fitted scalers are shared read-only.
"""

from __future__ import annotations

import json
from dataclasses import dataclass
from datetime import datetime
from typing import Mapping, Sequence

import numpy as np

from v5_generation_utils import (
    CYCLE_SECONDS,
    SENSOR_COUNT,
    WINDOW_SIZE,
    build_model_inputs,
)


EQUIPMENT_IDS = ("station-01", "station-02", "station-03")
DEFAULT_SEED_RECORDS = {
    "station-01": 1790,
    "station-02": 1793,
    "station-03": 1795,
}
NORMAL_PROFILE = np.asarray([100.0, 100.0, 0.0, 130.0, 0.0])
FORBIDDEN_RAW_FIELDS = {
    "confidence",
    "drift",
    "feature",
    "features",
    "label",
    "model",
    "prediction",
    "state",
}


@dataclass(frozen=True)
class SeedValidation:
    equipment_id: str
    record: int
    profile: tuple[float, ...]


class V5StationRuntime:
    """One station's independent anchored-offset inference state."""

    def __init__(
        self,
        *,
        equipment_id: str,
        seed_record: int,
        model,
        input_scaler,
        offset_scaler,
        sensor_min: np.ndarray,
        sensor_max: np.ndarray,
        seed_window: np.ndarray,
        ps4_index: int,
    ) -> None:
        if equipment_id not in EQUIPMENT_IDS:
            raise ValueError(f"Unsupported equipment_id: {equipment_id}")
        seed_window = np.asarray(seed_window, dtype=np.float32)
        if seed_window.shape != (WINDOW_SIZE, SENSOR_COUNT):
            raise ValueError(f"Unexpected seed window shape: {seed_window.shape}")
        if not np.isfinite(seed_window).all():
            raise ValueError(f"Seed {seed_record} contains NaN or Inf")

        self.equipment_id = equipment_id
        self.seed_record = int(seed_record)
        self.model = model
        self.input_scaler = input_scaler
        self.offset_scaler = offset_scaler
        self.sensor_min = np.asarray(sensor_min, dtype=np.float64).copy()
        self.sensor_max = np.asarray(sensor_max, dtype=np.float64).copy()
        self.anchor = seed_window[0].astype(np.float64, copy=True)
        self.sensor_window = seed_window[np.newaxis, ...].copy()
        self.phase_window = np.arange(WINDOW_SIZE, dtype=np.int32)[
            np.newaxis, ...
        ]
        self.seed_min = seed_window.min(axis=0).astype(np.float64)
        self.seed_max = seed_window.max(axis=0).astype(np.float64)
        self.ps4_index = int(ps4_index)
        self.current_sensors = seed_window[-1].astype(np.float64, copy=True)

    @property
    def cycle_position(self) -> int:
        return int(self.phase_window[0, -1])

    def predict_next(self) -> np.ndarray:
        model_input = build_model_inputs(
            self.sensor_window,
            self.phase_window,
            self.input_scaler,
        )
        offset_scaled = self.model.predict(model_input, verbose=0)
        offset = self.offset_scaler.inverse_transform(offset_scaled)[0]
        next_sensor = self.anchor + offset
        next_phase = (self.cycle_position + 1) % CYCLE_SECONDS

        if next_phase == 0:
            next_sensor = self.anchor.copy()
        next_sensor = np.minimum(
            np.maximum(next_sensor, self.sensor_min), self.sensor_max
        )
        next_sensor[self.ps4_index] = np.clip(
            next_sensor[self.ps4_index],
            self.seed_min[self.ps4_index],
            self.seed_max[self.ps4_index],
        )
        if not np.isfinite(next_sensor).all():
            raise ValueError(
                f"{self.equipment_id} prediction contains NaN or Inf"
            )

        self.sensor_window = np.concatenate(
            [
                self.sensor_window[:, 1:, :],
                next_sensor[np.newaxis, np.newaxis, :],
            ],
            axis=1,
        )
        self.phase_window = np.concatenate(
            [
                self.phase_window[:, 1:],
                np.asarray([[next_phase]], dtype=np.int32),
            ],
            axis=1,
        )
        self.current_sensors = next_sensor.astype(np.float64, copy=True)
        return self.current_sensors.copy()


def validate_seed_records(
    raw_data: np.ndarray,
    profiles: np.ndarray,
    seed_records: Mapping[str, int],
    sensor_min: np.ndarray,
    sensor_max: np.ndarray,
    validation_start: int,
) -> list[SeedValidation]:
    """Validate fixed seeds as stable, fully healthy UCI validation records."""
    if tuple(seed_records) != EQUIPMENT_IDS:
        raise ValueError("Seed mapping must contain station-01..station-03 in order")
    if len(set(seed_records.values())) != len(EQUIPMENT_IDS):
        raise ValueError("Every station must use a distinct seed record")

    results = []
    for equipment_id, record in seed_records.items():
        if not validation_start <= record < raw_data.shape[0]:
            raise ValueError(f"{equipment_id} seed is outside validation: {record}")
        profile = np.asarray(profiles[record], dtype=np.float64)
        if not np.array_equal(profile, NORMAL_PROFILE):
            raise ValueError(
                f"{equipment_id} seed {record} is not fully NORMAL: {profile}"
            )
        window = np.asarray(raw_data[record, :WINDOW_SIZE], dtype=np.float64)
        if not np.isfinite(window).all():
            raise ValueError(f"{equipment_id} seed contains NaN or Inf")
        if not ((window >= sensor_min) & (window <= sensor_max)).all():
            raise ValueError(f"{equipment_id} seed exceeds V5 sensor bounds")
        results.append(
            SeedValidation(
                equipment_id=equipment_id,
                record=int(record),
                profile=tuple(float(value) for value in profile),
            )
        )
    return results


def create_multi_raw_message(
    equipment_id: str,
    timestamp: str,
    sensor_names: Sequence[str],
    sensor_values: Sequence[float],
) -> dict:
    if equipment_id not in EQUIPMENT_IDS:
        raise ValueError(f"Unsupported equipment_id: {equipment_id}")
    return {
        "equipment_id": equipment_id,
        "timestamp": timestamp,
        "sensors": {
            sensor: round(float(value), 6)
            for sensor, value in zip(sensor_names, sensor_values)
        },
    }


def current_timestamp() -> str:
    return datetime.now().astimezone().isoformat(timespec="milliseconds")


def clip_for_six_decimal_raw_range(
    sensor_values: Sequence[float],
    raw_sensor_min: Sequence[float],
    raw_sensor_max: Sequence[float],
) -> np.ndarray:
    """Keep six-decimal Kafka values strictly inside the UCI Raw range."""
    values = np.asarray(sensor_values, dtype=np.float64)
    raw_min = np.asarray(raw_sensor_min, dtype=np.float64)
    raw_max = np.asarray(raw_sensor_max, dtype=np.float64)
    decimal_scale = 1_000_000.0
    safe_min = np.ceil(raw_min * decimal_scale) / decimal_scale
    safe_max = np.floor(raw_max * decimal_scale) / decimal_scale
    safe_min = np.where(safe_min < raw_min, safe_min + 1e-6, safe_min)
    safe_max = np.where(safe_max > raw_max, safe_max - 1e-6, safe_max)
    if np.any(safe_min > safe_max):
        raise ValueError("A sensor Raw range cannot be represented at six decimals")
    return np.clip(np.round(values, 6), safe_min, safe_max)


def validate_multi_raw_message(message: Mapping, sensor_names: Sequence[str]) -> None:
    if set(message) != {"equipment_id", "timestamp", "sensors"}:
        raise ValueError(f"Invalid Multi Raw message keys: {list(message)}")
    if message["equipment_id"] not in EQUIPMENT_IDS:
        raise ValueError(f"Invalid equipment_id: {message['equipment_id']}")
    sensors = message["sensors"]
    if list(sensors) != list(sensor_names):
        raise ValueError("Multi Raw sensor names/order changed")
    values = np.asarray(list(sensors.values()), dtype=np.float64)
    if values.shape != (SENSOR_COUNT,) or not np.isfinite(values).all():
        raise ValueError("Multi Raw sensors contain invalid values")
    serialized = json.dumps(message, ensure_ascii=False).lower()
    if any(field in serialized for field in FORBIDDEN_RAW_FIELDS):
        raise ValueError("Multi Raw message contains forbidden metadata")


def assert_station_values_differ(values_by_equipment: Mapping[str, np.ndarray]) -> None:
    if tuple(values_by_equipment) != EQUIPMENT_IDS:
        raise ValueError("All three station values are required in station order")
    for left_index, left_id in enumerate(EQUIPMENT_IDS):
        for right_id in EQUIPMENT_IDS[left_index + 1 :]:
            if np.array_equal(
                values_by_equipment[left_id], values_by_equipment[right_id]
            ):
                raise ValueError(f"Identical station output: {left_id} and {right_id}")
