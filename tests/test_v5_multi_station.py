import importlib.util
import sys
from pathlib import Path

import numpy as np
import pytest


PROJECT_ROOT = Path(__file__).resolve().parents[1]
SIMULATOR_SOURCE = PROJECT_ROOT / "src" / "simulator"
sys.path.insert(0, str(SIMULATOR_SOURCE))

from v5_multi_station_utils import (  # noqa: E402
    EQUIPMENT_IDS,
    V5StationRuntime,
    assert_station_values_differ,
    clip_for_six_decimal_raw_range,
    create_multi_raw_message,
    validate_multi_raw_message,
)


class IdentityScaler:
    def transform(self, values):
        return values

    def inverse_transform(self, values):
        return values


class ZeroOffsetModel:
    def predict(self, values, verbose=0):
        return np.zeros((values.shape[0], 17), dtype=np.float32)


def make_runtime(equipment_id, anchor, model=None):
    seed = np.tile(np.asarray(anchor, dtype=np.float32), (30, 1))
    return V5StationRuntime(
        equipment_id=equipment_id,
        seed_record=1790,
        model=model or ZeroOffsetModel(),
        input_scaler=IdentityScaler(),
        offset_scaler=IdentityScaler(),
        sensor_min=np.full(17, -1000.0),
        sensor_max=np.full(17, 1000.0),
        seed_window=seed,
        ps4_index=3,
    )


def test_station_runtime_state_is_independent():
    shared_model = ZeroOffsetModel()
    first = make_runtime("station-01", np.arange(17), shared_model)
    second = make_runtime("station-02", np.arange(17) + 100, shared_model)
    second_window_before = second.sensor_window.copy()

    first_values = first.predict_next()

    assert np.array_equal(first_values, np.arange(17))
    assert np.array_equal(second.sensor_window, second_window_before)
    assert first.sensor_window is not second.sensor_window
    assert first.anchor is not second.anchor
    assert first.model is second.model


def test_multi_raw_contract_has_only_allowed_fields():
    sensors = [f"S{index}" for index in range(17)]
    message = create_multi_raw_message(
        "station-01", "2026-09-01T12:00:00+09:00", sensors, np.arange(17)
    )
    validate_multi_raw_message(message, sensors)
    assert set(message) == {"equipment_id", "timestamp", "sensors"}
    assert len(message["sensors"]) == 17

    message["prediction"] = "NORMAL"
    with pytest.raises(ValueError):
        validate_multi_raw_message(message, sensors)


def test_station_difference_check_rejects_identical_values():
    values = {
        equipment_id: np.arange(17, dtype=np.float64) + index
        for index, equipment_id in enumerate(EQUIPMENT_IDS)
    }
    assert_station_values_differ(values)
    values["station-03"] = values["station-01"].copy()
    with pytest.raises(ValueError):
        assert_station_values_differ(values)


def test_six_decimal_values_stay_strictly_inside_raw_bounds():
    raw_min = np.asarray([134.84780883789062] * 17)
    raw_max = np.asarray([191.6153106689453] * 17)
    values = clip_for_six_decimal_raw_range(
        np.asarray([200.0] * 17), raw_min, raw_max
    )
    assert (values >= raw_min).all()
    assert (values <= raw_max).all()
    assert values[0] == 191.615310


def test_consumer_keeps_latest_state_by_equipment():
    module_path = PROJECT_ROOT / "kafka" / "raw_consumer_multi.py"
    spec = importlib.util.spec_from_file_location("raw_consumer_multi", module_path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    sensor_names = [f"S{index}" for index in range(17)]
    states = {}
    for index, equipment_id in enumerate(EQUIPMENT_IDS):
        message = create_multi_raw_message(
            equipment_id,
            f"2026-09-01T12:00:0{index}+09:00",
            sensor_names,
            np.arange(17) + index,
        )
        module.update_latest(states, message, sensor_names)
    assert tuple(states) == EQUIPMENT_IDS
    assert all(set(value) == {"timestamp", "sensors"} for value in states.values())
