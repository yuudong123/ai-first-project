import importlib.util
import json
from pathlib import Path

import pytest


PROJECT_ROOT = Path(__file__).resolve().parents[1]
MODULE_PATH = PROJECT_ROOT / "kafka" / "mock_state_builder_multi.py"
SPEC = importlib.util.spec_from_file_location("mock_state_builder_multi", MODULE_PATH)
MODULE = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(MODULE)


def make_raw(timestamp="2026-09-02T10:00:00+09:00", offset=0.0):
    return {
        station_id: {
            "timestamp": timestamp,
            "sensors": {
                sensor_id: offset + station_index * 100 + sensor_index
                for sensor_index, sensor_id in enumerate(MODULE.SENSOR_IDS)
            },
        }
        for station_index, station_id in enumerate(MODULE.EQUIPMENT_IDS)
    }


def write_json(path, value):
    path.write_text(json.dumps(value), encoding="utf-8")


def test_snapshot_has_exact_schema_and_preserves_all_raw_sensors():
    raw = make_raw()
    predictions, _ = MODULE.create_prediction_state()

    state = MODULE.build_state_snapshot(raw, predictions)

    assert tuple(state) == MODULE.EQUIPMENT_IDS
    for station_id in MODULE.EQUIPMENT_IDS:
        assert set(state[station_id]) == {"timestamp", "sensors", "prediction"}
        assert state[station_id]["timestamp"] == raw[station_id]["timestamp"]
        assert state[station_id]["sensors"] == raw[station_id]["sensors"]
        assert tuple(state[station_id]["sensors"]) == MODULE.SENSOR_IDS
        assert set(state[station_id]["prediction"]) == {
            "status",
            "result",
            "window_sec",
            "updated_at",
        }


def test_zero_through_nine_seconds_are_warming_up():
    predictions, last_blocks = MODULE.create_prediction_state()

    assert MODULE.update_predictions(
        predictions, last_blocks, 9.999, "not-used"
    ) == ()
    assert all(
        prediction
        == {
            "status": "warming_up",
            "result": None,
            "window_sec": 10,
            "updated_at": None,
        }
        for prediction in predictions.values()
    )


def test_ready_patterns_rotate_at_ten_second_boundaries_independently():
    predictions, last_blocks = MODULE.create_prediction_state()
    observed = {station_id: [] for station_id in MODULE.EQUIPMENT_IDS}

    expected_by_time = {
        10: ("normal", "abnormal", "normal"),
        20: ("abnormal", "normal", "normal"),
        30: ("normal", "normal", "abnormal"),
        40: ("normal", "abnormal", "normal"),
    }
    for elapsed, expected in expected_by_time.items():
        prediction_time = f"t{elapsed}"
        updated = MODULE.update_predictions(
            predictions, last_blocks, elapsed, prediction_time
        )
        assert updated == MODULE.EQUIPMENT_IDS
        assert tuple(
            predictions[station_id]["result"]
            for station_id in MODULE.EQUIPMENT_IDS
        ) == expected
        assert all(
            predictions[station_id]["status"] == "ready"
            and predictions[station_id]["updated_at"] == prediction_time
            for station_id in MODULE.EQUIPMENT_IDS
        )
        for station_id in MODULE.EQUIPMENT_IDS:
            observed[station_id].append(predictions[station_id]["result"])

    assert len({tuple(results) for results in observed.values()}) == 3


def test_prediction_timestamp_changes_only_on_new_ten_second_block():
    predictions, last_blocks = MODULE.create_prediction_state()

    MODULE.update_predictions(predictions, last_blocks, 10, "t10")
    before = {
        station_id: dict(prediction)
        for station_id, prediction in predictions.items()
    }
    assert MODULE.update_predictions(
        predictions, last_blocks, 19.999, "t19"
    ) == ()
    assert predictions == before

    MODULE.update_predictions(predictions, last_blocks, 20, "t20")
    assert all(
        prediction["updated_at"] == "t20"
        for prediction in predictions.values()
    )


def test_sensor_snapshot_can_refresh_each_second_without_prediction_refresh():
    predictions, last_blocks = MODULE.create_prediction_state()
    MODULE.update_predictions(predictions, last_blocks, 10, "prediction-t10")
    first_raw = make_raw(timestamp="sensor-t10", offset=0.0)
    second_raw = make_raw(timestamp="sensor-t11", offset=0.5)

    first = MODULE.build_state_snapshot(first_raw, predictions)
    second = MODULE.build_state_snapshot(second_raw, predictions)

    for station_id in MODULE.EQUIPMENT_IDS:
        assert first[station_id]["timestamp"] == "sensor-t10"
        assert second[station_id]["timestamp"] == "sensor-t11"
        assert first[station_id]["sensors"] != second[station_id]["sensors"]
        assert first[station_id]["prediction"] == second[station_id]["prediction"]
        assert second[station_id]["prediction"]["updated_at"] == "prediction-t10"
    assert MODULE.DEFAULT_POLL_INTERVAL_SEC == 1.0


def test_atomic_write_uses_named_temporary_file_and_os_replace(
    tmp_path, monkeypatch
):
    output_path = tmp_path / "latest_state_by_equipment.json"
    replacement_calls = []
    real_replace = MODULE.os.replace

    def recording_replace(source, destination):
        replacement_calls.append((source, destination))
        real_replace(source, destination)

    monkeypatch.setattr(MODULE.os, "replace", recording_replace)
    value = {"test": True}

    MODULE.atomic_write_json(output_path, value)

    assert replacement_calls == [
        (tmp_path / "latest_state_by_equipment.json.tmp", output_path)
    ]
    assert json.loads(output_path.read_text(encoding="utf-8")) == value
    assert not (tmp_path / "latest_state_by_equipment.json.tmp").exists()


@pytest.mark.parametrize("initial_content", [None, "{"])
def test_builder_retries_missing_or_invalid_raw_input(
    tmp_path, initial_content
):
    input_path = tmp_path / "latest_raw_by_equipment.json"
    output_path = tmp_path / "latest_state_by_equipment.json"
    if initial_content is not None:
        input_path.write_text(initial_content, encoding="utf-8")

    now = [0.0]
    sleep_calls = []

    def sleep_and_restore_input(seconds):
        sleep_calls.append(seconds)
        now[0] += seconds
        write_json(input_path, make_raw())

    writes = MODULE.run_builder(
        input_path,
        output_path,
        poll_interval=0.01,
        max_updates=1,
        monotonic_fn=lambda: now[0],
        sleep_fn=sleep_and_restore_input,
        timestamp_fn=lambda: "prediction-time",
    )

    assert writes == 1
    assert sleep_calls == [0.01]
    assert output_path.exists()


def test_successful_builder_run_never_modifies_raw_input(tmp_path):
    input_path = tmp_path / "latest_raw_by_equipment.json"
    output_path = tmp_path / "latest_state_by_equipment.json"
    write_json(input_path, make_raw())
    raw_before = input_path.read_bytes()

    MODULE.run_builder(
        input_path,
        output_path,
        max_updates=1,
        monotonic_fn=lambda: 0.0,
        sleep_fn=lambda _seconds: None,
        timestamp_fn=lambda: "prediction-time",
    )

    assert input_path.read_bytes() == raw_before
    output = json.loads(output_path.read_text(encoding="utf-8"))
    raw = json.loads(raw_before)
    assert all(
        output[station_id]["sensors"] == raw[station_id]["sensors"]
        for station_id in MODULE.EQUIPMENT_IDS
    )


def test_builder_rejects_using_raw_input_as_output(tmp_path):
    raw_path = tmp_path / "latest_raw_by_equipment.json"

    with pytest.raises(ValueError, match="must be different"):
        MODULE.run_builder(raw_path, raw_path, max_updates=1)


def test_mock_notice_is_explicit_in_code_and_runtime_log(tmp_path, capsys):
    input_path = tmp_path / "latest_raw_by_equipment.json"
    output_path = tmp_path / "latest_state_by_equipment.json"
    write_json(input_path, make_raw())

    MODULE.run_builder(
        input_path,
        output_path,
        max_updates=1,
        monotonic_fn=lambda: 0.0,
        sleep_fn=lambda _seconds: None,
        timestamp_fn=lambda: "prediction-time",
    )

    assert MODULE.MOCK_NOTICE in MODULE_PATH.read_text(encoding="utf-8")
    assert MODULE.MOCK_NOTICE in capsys.readouterr().out
