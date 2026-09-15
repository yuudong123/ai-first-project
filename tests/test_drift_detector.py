import numpy as np
import pytest

from src.monitoring.drift_detector import (
    DriftConfig,
    RollingDriftDetector,
    fit_reference,
)


SENSORS = ("S1", "S2")


def reference_rows():
    return [
        {
            "S1": float(np.sin(index / 4) * 0.1),
            "S2": float(10 + np.cos(index / 5) * 0.2),
        }
        for index in range(100)
    ]


def test_detector_warms_up_then_detects_persistent_mean_shift():
    reference = fit_reference(reference_rows(), SENSORS, min_samples=30)
    detector = RollingDriftDetector(
        reference,
        DriftConfig(
            window_size=4,
            mean_shift_threshold=2.0,
            out_of_bounds_ratio_threshold=0.75,
            min_affected_sensors=1,
            consecutive_windows=2,
        ),
    )

    assert detector.update({"S1": 0.0, "S2": 10.0})["status"] == "warming_up"
    detector.update({"S1": 5.0, "S2": 10.0})
    detector.update({"S1": 5.0, "S2": 10.0})
    suspected = detector.update({"S1": 5.0, "S2": 10.0})
    detected = detector.update({"S1": 5.0, "S2": 10.0})

    assert suspected["status"] == "suspected"
    assert detected["status"] == "drift"
    assert detected["drift_detected"] is True
    assert detected["sensor_scores"]["S1"]["mean_offset"] > 0
    assert "S1" in detected["affected_sensors"]


def test_detector_rejects_missing_sensor():
    reference = fit_reference(reference_rows(), SENSORS, min_samples=30)
    detector = RollingDriftDetector(reference)
    with pytest.raises(ValueError, match="누락된 센서"):
        detector.update({"S1": 0.0})
