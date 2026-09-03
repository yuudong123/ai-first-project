"""Environmental temperature/pressure drift and file-based runtime control.

The injector is deliberately independent from the V5 model runtime.  It only
changes the final TS1..TS4 and PS1..PS6 values and keeps drift metadata outside
Kafka.
"""

from __future__ import annotations

import hashlib
import json
import math
import os
import random
import time
import uuid
from contextlib import contextmanager
from dataclasses import dataclass
from datetime import datetime, timezone
from enum import Enum
from pathlib import Path
from typing import Callable, Iterable, Mapping, Sequence


TEMPERATURE_SENSORS = ("TS1", "TS2", "TS3", "TS4")
PRESSURE_SENSORS = ("PS1", "PS2", "PS3", "PS4", "PS5", "PS6")
DRIFT_SENSORS = PRESSURE_SENSORS + TEMPERATURE_SENSORS
TEMP_MIN_OFFSET = 0.0
TEMP_MAX_OFFSET = 4.0
PRESSURE_MIN_PERCENT = 0.0
PRESSURE_MAX_PERCENT = 3.0
DRIFT_MIN_DURATION_SEC = 20.0
DRIFT_MAX_DURATION_SEC = 60.0
RISING_DURATION_RATIO = 0.35
MAX_HOLD_DURATION_RATIO = 0.30
FALLING_DURATION_RATIO = 0.35
DRIFT_MODES = ("off", "manual", "auto")
CONTROL_TARGET_ALL = "all"
CONTROL_ACTIONS = ("trigger", "reset")


class DriftState(str, Enum):
    NORMAL = "NORMAL"
    RISING = "RISING"
    MAX_HOLD = "MAX_HOLD"
    FALLING = "FALLING"


@dataclass(frozen=True)
class DriftConfig:
    # Keep the deployed temperature field names as the canonical Python API.
    min_offset: float = TEMP_MIN_OFFSET
    max_offset: float = TEMP_MAX_OFFSET
    # Retained so existing deployments can keep passing the legacy option.
    # Sampled event duration now contains all three active phases.
    step_per_sec: float = 0.1
    max_hold_sec: float = 30.0
    auto_normal_min_sec: float = 120.0
    auto_normal_max_sec: float = 240.0
    seed: int = 0
    # Appended to preserve positional construction of the original config.
    pressure_min_percent: float = PRESSURE_MIN_PERCENT
    pressure_max_percent: float = PRESSURE_MAX_PERCENT
    min_duration_sec: float = DRIFT_MIN_DURATION_SEC
    max_duration_sec: float = DRIFT_MAX_DURATION_SEC

    def __post_init__(self) -> None:
        numeric_values = (
            self.min_offset,
            self.max_offset,
            self.pressure_min_percent,
            self.pressure_max_percent,
            self.min_duration_sec,
            self.max_duration_sec,
            self.step_per_sec,
            self.max_hold_sec,
            self.auto_normal_min_sec,
            self.auto_normal_max_sec,
        )
        if not all(math.isfinite(value) for value in numeric_values):
            raise ValueError("Drift configuration values must be finite")
        if self.min_offset > self.max_offset:
            raise ValueError("Drift min_offset cannot exceed max_offset")
        if self.pressure_min_percent < 0:
            raise ValueError("Drift pressure_min_percent cannot be negative")
        if self.pressure_min_percent > self.pressure_max_percent:
            raise ValueError(
                "Drift pressure_min_percent cannot exceed pressure_max_percent"
            )
        if self.min_duration_sec <= 0:
            raise ValueError("Drift minimum duration must be greater than zero")
        if self.min_duration_sec > self.max_duration_sec:
            raise ValueError("Drift minimum duration cannot exceed maximum")
        if self.step_per_sec <= 0:
            raise ValueError("Drift step_per_sec must be greater than zero")
        if self.max_hold_sec < 0:
            raise ValueError("Drift max_hold_sec cannot be negative")
        if self.auto_normal_min_sec < 0:
            raise ValueError("Auto normal minimum cannot be negative")
        if self.auto_normal_min_sec > self.auto_normal_max_sec:
            raise ValueError("Auto normal minimum cannot exceed maximum")


def _station_seed(base_seed: int, equipment_id: str) -> int:
    payload = f"{base_seed}:{equipment_id}".encode("utf-8")
    return int.from_bytes(hashlib.sha256(payload).digest()[:8], "big")


class StationDriftController:
    """Elapsed-time state machine owned by exactly one station."""

    def __init__(
        self,
        equipment_id: str,
        config: DriftConfig,
        *,
        automatic: bool = False,
        now: float | None = None,
    ) -> None:
        self.equipment_id = equipment_id
        self.config = config
        self.automatic = bool(automatic)
        self.state = DriftState.NORMAL
        self.current_offset = 0.0
        self.target_temp_offset = 0.0
        self.target_pressure_percent = 0.0
        self.event_duration_sec = 0.0
        self.rising_duration_sec = 0.0
        self.max_hold_duration_sec = 0.0
        self.falling_duration_sec = 0.0
        self._progress = 0.0
        self.hold_elapsed_sec = 0.0
        self._rng = random.Random(_station_seed(config.seed, equipment_id))
        self._last_update = time.monotonic() if now is None else float(now)
        self.next_auto_at = self._schedule_auto(self._last_update)

    def _normal_duration(self) -> float:
        return self._rng.uniform(
            self.config.auto_normal_min_sec,
            self.config.auto_normal_max_sec,
        )

    def _schedule_auto(self, now: float) -> float | None:
        if not self.automatic:
            return None
        return now + self._normal_duration()

    def _set_progress(self, progress: float) -> None:
        self._progress = min(1.0, max(0.0, float(progress)))
        self.current_offset = self.target_temp_offset * self._progress

    def _sample_event(self) -> None:
        self.target_temp_offset = self._rng.uniform(
            self.config.min_offset,
            self.config.max_offset,
        )
        self.target_pressure_percent = self._rng.uniform(
            self.config.pressure_min_percent,
            self.config.pressure_max_percent,
        )
        self.event_duration_sec = self._rng.uniform(
            self.config.min_duration_sec,
            self.config.max_duration_sec,
        )
        nominal_hold_duration = (
            self.event_duration_sec * MAX_HOLD_DURATION_RATIO
        )
        self.max_hold_duration_sec = min(
            nominal_hold_duration,
            self.config.max_hold_sec,
        )
        ramp_duration = self.event_duration_sec - self.max_hold_duration_sec
        ramp_ratio = RISING_DURATION_RATIO + FALLING_DURATION_RATIO
        self.rising_duration_sec = (
            ramp_duration * RISING_DURATION_RATIO / ramp_ratio
        )
        self.falling_duration_sec = (
            self.event_duration_sec
            - self.rising_duration_sec
            - self.max_hold_duration_sec
        )
        self._set_progress(0.0)

    def _begin_event(self) -> None:
        self._sample_event()
        self.state = DriftState.RISING
        self.hold_elapsed_sec = 0.0
        self.next_auto_at = None

    def _finish_event(self, now: float) -> None:
        self.state = DriftState.NORMAL
        self.target_temp_offset = 0.0
        self.target_pressure_percent = 0.0
        self.event_duration_sec = 0.0
        self.rising_duration_sec = 0.0
        self.max_hold_duration_sec = 0.0
        self.falling_duration_sec = 0.0
        self._set_progress(0.0)
        self.hold_elapsed_sec = 0.0
        self.next_auto_at = self._schedule_auto(now)

    def _clamp(self) -> None:
        self._set_progress(self._progress)

    @property
    def progress(self) -> float:
        """Return the shared temperature/pressure drift progress in [0, 1]."""
        return self._progress

    @property
    def temperature_offset(self) -> float:
        return self.current_offset

    @property
    def pressure_percent(self) -> float:
        return self.target_pressure_percent * self.progress

    def trigger(self, now: float | None = None) -> bool:
        now = time.monotonic() if now is None else float(now)
        self.update(now)
        if self.state is not DriftState.NORMAL:
            return False
        self._begin_event()
        self._last_update = now
        return True

    def reset(self, now: float | None = None) -> None:
        now = time.monotonic() if now is None else float(now)
        self._finish_event(now)
        self._last_update = now

    def update(self, now: float | None = None) -> float:
        now = time.monotonic() if now is None else float(now)
        if now <= self._last_update:
            self._clamp()
            return self.current_offset

        remaining = now - self._last_update
        cursor = self._last_update
        transitions = 0
        while remaining > 1e-12:
            transitions += 1
            if transitions > 1000:
                raise RuntimeError("Drift state machine exceeded transition limit")

            if self.state is DriftState.NORMAL:
                self._set_progress(0.0)
                if self.next_auto_at is None or self.next_auto_at > now:
                    break
                wait = max(0.0, self.next_auto_at - cursor)
                if wait > remaining:
                    break
                remaining -= wait
                cursor += wait
                self._begin_event()
                continue

            if self.state is DriftState.RISING:
                duration = (1.0 - self.progress) * self.rising_duration_sec
                if remaining + 1e-12 < duration:
                    self._set_progress(
                        self.progress + remaining / self.rising_duration_sec
                    )
                    cursor += remaining
                    remaining = 0.0
                else:
                    self._set_progress(1.0)
                    cursor += duration
                    remaining -= duration
                    self.state = DriftState.MAX_HOLD
                    self.hold_elapsed_sec = 0.0
                continue

            if self.state is DriftState.MAX_HOLD:
                self._set_progress(1.0)
                hold_left = max(
                    0.0,
                    self.max_hold_duration_sec - self.hold_elapsed_sec,
                )
                if remaining + 1e-12 < hold_left:
                    self.hold_elapsed_sec += remaining
                    cursor += remaining
                    remaining = 0.0
                else:
                    self.hold_elapsed_sec = self.max_hold_duration_sec
                    cursor += hold_left
                    remaining -= hold_left
                    self.state = DriftState.FALLING
                continue

            duration = self.progress * self.falling_duration_sec
            if remaining + 1e-12 < duration:
                self._set_progress(
                    self.progress - remaining / self.falling_duration_sec
                )
                cursor += remaining
                remaining = 0.0
            else:
                cursor += duration
                remaining -= duration
                self._finish_event(cursor)

        self._last_update = now
        self._clamp()
        return self.current_offset

    def status(self) -> dict:
        return {
            "state": self.state.value,
            # Retained for consumers of the original temperature-only status.
            "offset": self.current_offset,
            "progress": self.progress,
            "temperature_offset": self.temperature_offset,
            "pressure_percent": self.pressure_percent,
            "target_temp_offset": self.target_temp_offset,
            "target_pressure_percent": self.target_pressure_percent,
            "event_duration_sec": self.event_duration_sec,
            "next_auto_in_sec": (
                None
                if self.next_auto_at is None
                else max(0.0, self.next_auto_at - self._last_update)
            ),
        }


def atomic_write_json(path: Path, payload: Mapping) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.{uuid.uuid4().hex}.tmp")
    try:
        with temporary.open("w", encoding="utf-8", newline="\n") as output_file:
            json.dump(payload, output_file, ensure_ascii=False, indent=2)
            output_file.write("\n")
            output_file.flush()
            os.fsync(output_file.fileno())
        os.replace(temporary, path)
    finally:
        try:
            temporary.unlink()
        except FileNotFoundError:
            pass


@contextmanager
def _control_lock(path: Path, timeout_sec: float = 2.0):
    lock_path = Path(f"{path}.lock")
    lock_path.parent.mkdir(parents=True, exist_ok=True)
    deadline = time.monotonic() + timeout_sec
    descriptor = None
    while descriptor is None:
        try:
            descriptor = os.open(
                lock_path,
                os.O_CREAT | os.O_EXCL | os.O_WRONLY,
                0o600,
            )
            os.write(descriptor, f"{os.getpid()}\n".encode("ascii"))
        except FileExistsError:
            try:
                stale = time.time() - lock_path.stat().st_mtime > 30.0
                if stale:
                    lock_path.unlink()
                    continue
            except FileNotFoundError:
                continue
            if time.monotonic() >= deadline:
                raise TimeoutError(f"Timed out waiting for control lock: {lock_path}")
            time.sleep(0.01)
    try:
        yield
    finally:
        os.close(descriptor)
        try:
            lock_path.unlink()
        except FileNotFoundError:
            pass


def _read_control_payload(path: Path) -> dict:
    if not path.exists():
        return {"version": 1, "commands": []}
    with path.open(encoding="utf-8") as input_file:
        payload = json.load(input_file)
    if not isinstance(payload, dict) or not isinstance(payload.get("commands"), list):
        raise ValueError("Control JSON must contain a commands list")
    return payload


def enqueue_control_command(path: Path, action: str, target: str) -> str:
    if action not in CONTROL_ACTIONS:
        raise ValueError(f"Unsupported control action: {action}")
    command_id = uuid.uuid4().hex
    command = {
        "id": command_id,
        "action": action,
        "target": target,
        "created_at": datetime.now(timezone.utc).isoformat(timespec="milliseconds"),
    }
    with _control_lock(path):
        payload = _read_control_payload(path)
        payload["version"] = 1
        payload["commands"].append(command)
        atomic_write_json(path, payload)
    return command_id


def consume_control_commands(path: Path) -> tuple[list[dict], list[str]]:
    path = Path(path)
    try:
        with _control_lock(path):
            try:
                payload = _read_control_payload(path)
            except (OSError, ValueError, json.JSONDecodeError) as error:
                return [], [f"Ignoring malformed drift control JSON: {error}"]

            commands = []
            warnings = []
            for item in payload["commands"]:
                if not isinstance(item, dict):
                    warnings.append("Discarded non-object drift control command")
                    continue
                command_id = item.get("id")
                action = item.get("action")
                target = item.get("target")
                if not isinstance(command_id, str) or not command_id:
                    warnings.append("Discarded drift command without an id")
                    continue
                if action not in CONTROL_ACTIONS:
                    warnings.append(f"Discarded unsupported drift action: {action}")
                    continue
                if not isinstance(target, str) or not target:
                    warnings.append("Discarded drift command without a target")
                    continue
                commands.append(item)
            if payload["commands"]:
                atomic_write_json(path, {"version": 1, "commands": []})
            return commands, warnings
    except (OSError, TimeoutError) as error:
        return [], [f"Drift control polling failed safely: {error}"]


class DriftInjector:
    """Own station controllers, consume commands, and inject synchronized drift."""

    def __init__(
        self,
        equipment_ids: Iterable[str],
        sensor_names: Sequence[str],
        config: DriftConfig,
        *,
        mode: str = "off",
        control_path: Path | None = None,
        status_path: Path | None = None,
        now: float | None = None,
    ) -> None:
        if mode not in DRIFT_MODES:
            raise ValueError(f"Unsupported drift mode: {mode}")
        self.equipment_ids = tuple(equipment_ids)
        if not self.equipment_ids or len(set(self.equipment_ids)) != len(
            self.equipment_ids
        ):
            raise ValueError("Drift equipment IDs must be unique and non-empty")
        self.sensor_names = tuple(sensor_names)
        missing = set(DRIFT_SENSORS).difference(self.sensor_names)
        if missing:
            raise ValueError(f"Missing drift sensors: {sorted(missing)}")
        self.temperature_indexes = tuple(
            self.sensor_names.index(sensor) for sensor in TEMPERATURE_SENSORS
        )
        self.pressure_indexes = tuple(
            self.sensor_names.index(sensor) for sensor in PRESSURE_SENSORS
        )
        self.config = config
        self.mode = mode
        self.control_path = None if control_path is None else Path(control_path)
        self.status_path = None if status_path is None else Path(status_path)
        start = time.monotonic() if now is None else float(now)
        self.controllers = {
            equipment_id: StationDriftController(
                equipment_id,
                config,
                automatic=mode == "auto",
                now=start,
            )
            for equipment_id in self.equipment_ids
        }

    def process_control(self, now: float) -> list[str]:
        if self.control_path is None:
            return []
        commands, messages = consume_control_commands(self.control_path)
        for command in commands:
            target = command["target"]
            if target == CONTROL_TARGET_ALL:
                targets = self.equipment_ids
            elif target in self.controllers:
                targets = (target,)
            else:
                messages.append(
                    f"[{command['id']}] ignored unknown station: {target}"
                )
                continue

            if command["action"] == "reset":
                for equipment_id in targets:
                    self.controllers[equipment_id].reset(now)
                messages.append(
                    f"[{command['id']}] reset applied to {', '.join(targets)}"
                )
                continue

            if self.mode == "off":
                messages.append(
                    f"[{command['id']}] trigger ignored because drift mode is off"
                )
                continue
            changed = []
            active = []
            for equipment_id in targets:
                if self.controllers[equipment_id].trigger(now):
                    changed.append(equipment_id)
                else:
                    active.append(equipment_id)
            if changed:
                messages.append(
                    f"[{command['id']}] trigger applied to {', '.join(changed)}"
                )
            if active:
                messages.append(
                    f"[{command['id']}] already active: {', '.join(active)}"
                )
        return messages

    def write_status(self) -> None:
        if self.status_path is None:
            return
        payload = {
            "version": 1,
            "mode": self.mode,
            "updated_at": datetime.now(timezone.utc).isoformat(timespec="milliseconds"),
            "stations": {
                equipment_id: self.controllers[equipment_id].status()
                for equipment_id in self.equipment_ids
            },
        }
        atomic_write_json(self.status_path, payload)

    def process_cycle(
        self,
        values_by_equipment: Mapping[str, Sequence[float]],
        *,
        now: float | None = None,
        warning_handler: Callable[[str], None] | None = None,
    ) -> dict[str, object]:
        import numpy as np

        now = time.monotonic() if now is None else float(now)
        for message in self.process_control(now):
            if warning_handler is not None:
                warning_handler(message)
        for controller in self.controllers.values():
            controller.update(now)

        result = {}
        for equipment_id in self.equipment_ids:
            values = np.asarray(
                values_by_equipment[equipment_id], dtype=np.float64
            ).copy()
            if values.shape != (len(self.sensor_names),):
                raise ValueError(
                    f"Unexpected sensor shape for {equipment_id}: {values.shape}"
                )
            if self.mode != "off":
                controller = self.controllers[equipment_id]
                values[list(self.temperature_indexes)] += (
                    controller.temperature_offset
                )
                values[list(self.pressure_indexes)] *= (
                    1.0 + controller.pressure_percent / 100.0
                )
            if not np.isfinite(values).all():
                raise ValueError(f"Drift output contains NaN or Inf: {equipment_id}")
            result[equipment_id] = values
        try:
            self.write_status()
        except OSError as error:
            if warning_handler is not None:
                warning_handler(f"Drift status write failed safely: {error}")
        return result
