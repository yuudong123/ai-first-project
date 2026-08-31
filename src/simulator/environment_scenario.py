"""Presentation-only environmental temperature drift for HydroTwin.

This scenario is intentionally injected for a live demonstration.  It is not
an attempt to reproduce a UCI fault label or ground-truth equipment failure.
Only TS1, TS2, TS3, and TS4 are changed.
"""

from dataclasses import dataclass

import numpy as np


TEMPERATURE_SENSORS = ("TS1", "TS2", "TS3", "TS4")


@dataclass(frozen=True)
class EnvironmentScenarioConfig:
    max_temperature_offset: float = 4.0
    normal_seconds: int = 60
    drift_seconds: int = 60
    hold_seconds: int = 60
    recovery_seconds: int = 60
    final_normal_seconds: int = 60

    def __post_init__(self):
        durations = [
            self.normal_seconds,
            self.drift_seconds,
            self.hold_seconds,
            self.recovery_seconds,
            self.final_normal_seconds,
        ]
        if any(duration <= 0 for duration in durations):
            raise ValueError("Every scenario phase duration must be positive")
        if self.max_temperature_offset < 0:
            raise ValueError("Maximum temperature offset must be non-negative")

    @property
    def total_seconds(self):
        return (
            self.normal_seconds
            + self.drift_seconds
            + self.hold_seconds
            + self.recovery_seconds
            + self.final_normal_seconds
        )


def _linear_progress(position, duration):
    if duration == 1:
        return 1.0
    return position / float(duration - 1)


def phase_and_temperature_offset(elapsed_sec, config):
    """Return the private evaluation phase and offset for one elapsed second."""
    if elapsed_sec < 0 or elapsed_sec >= config.total_seconds:
        raise ValueError(
            f"elapsed_sec must be 0..{config.total_seconds - 1}: {elapsed_sec}"
        )

    normal_end = config.normal_seconds
    drift_end = normal_end + config.drift_seconds
    hold_end = drift_end + config.hold_seconds
    recovery_end = hold_end + config.recovery_seconds

    if elapsed_sec < normal_end:
        return "NORMAL", 0.0
    if elapsed_sec < drift_end:
        position = elapsed_sec - normal_end
        offset = config.max_temperature_offset * _linear_progress(
            position, config.drift_seconds
        )
        return "GRADUAL_DRIFT", offset
    if elapsed_sec < hold_end:
        return "HIGH_TEMP_HOLD", config.max_temperature_offset
    if elapsed_sec < recovery_end:
        position = elapsed_sec - hold_end
        offset = config.max_temperature_offset * (
            1.0 - _linear_progress(position, config.recovery_seconds)
        )
        return "RECOVERY", offset
    return "FINAL_NORMAL", 0.0


def apply_temperature_offset(sensor_values, sensor_names, offset):
    """Add the environmental offset to TS1-TS4 and no other sensor."""
    result = np.asarray(sensor_values, dtype=np.float64).copy()
    for sensor in TEMPERATURE_SENSORS:
        result[sensor_names.index(sensor)] += offset
    return result
