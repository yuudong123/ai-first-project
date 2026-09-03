import importlib.util
import json
import sys
from pathlib import Path

import numpy as np
import pytest


PROJECT_ROOT = Path(__file__).resolve().parents[1]
SIMULATOR_SOURCE = PROJECT_ROOT / "src" / "simulator"
sys.path.insert(0, str(SIMULATOR_SOURCE))

from drift_injector import (  # noqa: E402
    FALLING_DURATION_RATIO,
    MAX_HOLD_DURATION_RATIO,
    PRESSURE_SENSORS,
    RISING_DURATION_RATIO,
    TEMPERATURE_SENSORS,
    DriftConfig,
    DriftInjector,
    DriftState,
    StationDriftController,
    enqueue_control_command,
)
from v5_multi_station_utils import (  # noqa: E402
    EQUIPMENT_IDS,
    create_multi_raw_message,
    validate_multi_raw_message,
)


SENSOR_NAMES = [
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
]
FAST_EVENT_DURATION_SEC = 40.0 / 7.0


def fast_config(**overrides):
    values = {
        "min_offset": 1.0,
        "max_offset": 1.0,
        "pressure_min_percent": 3.0,
        "pressure_max_percent": 3.0,
        "min_duration_sec": FAST_EVENT_DURATION_SEC,
        "max_duration_sec": FAST_EVENT_DURATION_SEC,
        "step_per_sec": 0.5,
        "max_hold_sec": 2.0,
        "auto_normal_min_sec": 10.0,
        "auto_normal_max_sec": 20.0,
        "seed": 123,
    }
    values.update(overrides)
    return DriftConfig(**values)


def station_values():
    return {
        equipment_id: np.arange(17, dtype=np.float64) + index * 100.0
        for index, equipment_id in enumerate(EQUIPMENT_IDS)
    }


def test_rise_hold_fall_transitions_and_clamps():
    controller = StationDriftController("station-01", fast_config(), now=0.0)
    assert controller.trigger(0.0)

    rising_end = controller.rising_duration_sec
    hold_end = rising_end + controller.max_hold_duration_sec
    event_end = controller.event_duration_sec

    assert controller.update(rising_end / 2.0) == pytest.approx(0.5)
    assert controller.state is DriftState.RISING
    assert controller.update(rising_end) == pytest.approx(1.0)
    assert controller.state is DriftState.MAX_HOLD
    assert controller.update(
        rising_end + controller.max_hold_duration_sec / 2.0
    ) == pytest.approx(1.0)
    assert controller.state is DriftState.MAX_HOLD
    assert controller.update(hold_end) == pytest.approx(1.0)
    assert controller.state is DriftState.FALLING
    assert controller.update(
        hold_end + controller.falling_duration_sec / 2.0
    ) == pytest.approx(0.5)
    assert controller.state is DriftState.FALLING
    assert controller.update(event_end) == pytest.approx(0.0)
    assert controller.state is DriftState.NORMAL

    controller.update(1000.0)
    assert 0.0 <= controller.current_offset <= fast_config().max_offset


def test_large_elapsed_time_clamps_exactly_at_both_limits():
    controller = StationDriftController("station-01", fast_config(), now=0.0)
    controller.trigger(0.0)
    controller.update(2.5)
    assert controller.current_offset == fast_config().max_offset
    assert controller.state is DriftState.MAX_HOLD
    controller.update(100.0)
    assert controller.current_offset == 0.0
    assert controller.state is DriftState.NORMAL


def test_temperature_and_pressure_share_progress_and_other_sensors_are_unchanged():
    injector = DriftInjector(
        EQUIPMENT_IDS, SENSOR_NAMES, fast_config(), mode="manual", now=0.0
    )
    assert injector.controllers["station-02"].trigger(0.0)
    original = station_values()
    result = injector.process_cycle(original, now=1.0)

    temperature_indexes = {SENSOR_NAMES.index(name) for name in TEMPERATURE_SENSORS}
    pressure_indexes = {SENSOR_NAMES.index(name) for name in PRESSURE_SENSORS}
    for equipment_id in EQUIPMENT_IDS:
        for index in range(17):
            expected = original[equipment_id][index]
            if equipment_id == "station-02" and index in temperature_indexes:
                expected += 0.5
            if equipment_id == "station-02" and index in pressure_indexes:
                expected *= 1.015
            assert result[equipment_id][index] == pytest.approx(expected)
    unchanged_indexes = [
        index
        for index, sensor in enumerate(SENSOR_NAMES)
        if sensor not in PRESSURE_SENSORS + TEMPERATURE_SENSORS
    ]
    assert (
        result["station-02"][unchanged_indexes].tobytes()
        == original["station-02"][unchanged_indexes].tobytes()
    )
    assert injector.controllers["station-01"].state is DriftState.NORMAL
    assert injector.controllers["station-03"].state is DriftState.NORMAL


def test_all_pressure_sensors_use_same_percentage_across_different_scales():
    injector = DriftInjector(
        EQUIPMENT_IDS, SENSOR_NAMES, fast_config(), mode="manual", now=0.0
    )
    original = station_values()
    original["station-01"][:6] = [0.1, 1.0, 10.0, 100.0, 1000.0, 12345.0]
    assert injector.controllers["station-01"].trigger(0.0)

    result = injector.process_cycle(original, now=1.0)

    for sensor in PRESSURE_SENSORS:
        index = SENSOR_NAMES.index(sensor)
        percent = (
            result["station-01"][index] / original["station-01"][index] - 1.0
        ) * 100
        assert percent == pytest.approx(1.5)


def test_fixed_targets_map_shared_progress_to_temperature_and_pressure():
    config = fast_config(
        min_offset=4.0,
        max_offset=4.0,
        pressure_min_percent=3.0,
        pressure_max_percent=3.0,
    )
    injector = DriftInjector(
        EQUIPMENT_IDS, SENSOR_NAMES, config, mode="manual", now=0.0
    )
    controller = injector.controllers["station-01"]
    original = station_values()
    assert controller.trigger(0.0)

    halfway = injector.process_cycle(original, now=1.0)
    assert controller.progress == pytest.approx(0.5)
    assert controller.temperature_offset == pytest.approx(2.0)
    assert controller.pressure_percent == pytest.approx(1.5)
    for sensor in TEMPERATURE_SENSORS:
        index = SENSOR_NAMES.index(sensor)
        assert halfway["station-01"][index] == pytest.approx(
            original["station-01"][index] + 2.0
        )

    maximum = injector.process_cycle(original, now=2.0)
    assert controller.progress == pytest.approx(1.0)
    assert controller.temperature_offset == pytest.approx(4.0)
    assert controller.pressure_percent == pytest.approx(3.0)
    for sensor in PRESSURE_SENSORS:
        index = SENSOR_NAMES.index(sensor)
        assert maximum["station-01"][index] == pytest.approx(
            original["station-01"][index] * 1.03
        )


def test_pressure_percentage_is_clamped_to_zero_and_three_percent():
    injector = DriftInjector(
        EQUIPMENT_IDS, SENSOR_NAMES, fast_config(), mode="manual", now=0.0
    )
    controller = injector.controllers["station-01"]
    assert controller.trigger(0.0)

    observed = []
    for now in (0.0, 0.5, 1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 100.0):
        injector.process_cycle(station_values(), now=now)
        observed.append(controller.pressure_percent)

    assert min(observed) >= 0.0
    assert max(observed) <= 3.0
    assert 3.0 in observed
    assert controller.state is DriftState.NORMAL
    assert controller.pressure_percent == 0.0


def test_trigger_one_and_trigger_all_are_consumed_once(tmp_path):
    control_path = tmp_path / "drift_control.json"
    injector = DriftInjector(
        EQUIPMENT_IDS,
        SENSOR_NAMES,
        fast_config(),
        mode="manual",
        control_path=control_path,
        now=0.0,
    )
    enqueue_control_command(control_path, "trigger", "station-02")
    messages = []
    injector.process_cycle(station_values(), now=0.0, warning_handler=messages.append)
    assert injector.controllers["station-02"].state is DriftState.RISING
    assert injector.controllers["station-01"].state is DriftState.NORMAL
    assert injector.controllers["station-03"].state is DriftState.NORMAL
    assert any("trigger applied to station-02" in message for message in messages)
    assert json.loads(control_path.read_text(encoding="utf-8"))["commands"] == []

    drifted = injector.process_cycle(station_values(), now=1.0)
    assert drifted["station-02"][SENSOR_NAMES.index("TS1")] == pytest.approx(109.5)
    assert drifted["station-02"][SENSOR_NAMES.index("PS1")] == pytest.approx(101.5)

    injector.controllers["station-02"].reset(1.0)
    injector.process_cycle(station_values(), now=1.0)
    assert injector.controllers["station-02"].state is DriftState.NORMAL

    enqueue_control_command(control_path, "trigger", "all")
    injector.process_cycle(station_values(), now=2.0)
    assert all(
        controller.state is DriftState.RISING
        for controller in injector.controllers.values()
    )


def test_reset_one_and_reset_all(tmp_path):
    control_path = tmp_path / "drift_control.json"
    injector = DriftInjector(
        EQUIPMENT_IDS,
        SENSOR_NAMES,
        fast_config(),
        mode="manual",
        control_path=control_path,
        now=0.0,
    )
    for controller in injector.controllers.values():
        controller.trigger(0.0)
        controller.update(1.0)

    enqueue_control_command(control_path, "reset", "station-02")
    reset_result = injector.process_cycle(station_values(), now=1.0)
    assert injector.controllers["station-02"].state is DriftState.NORMAL
    assert injector.controllers["station-02"].current_offset == 0.0
    assert np.array_equal(reset_result["station-02"], station_values()["station-02"])
    assert injector.controllers["station-01"].state is DriftState.RISING
    assert injector.controllers["station-03"].state is DriftState.RISING

    enqueue_control_command(control_path, "reset", "all")
    injector.process_cycle(station_values(), now=2.0)
    assert all(
        controller.state is DriftState.NORMAL
        and controller.current_offset == 0.0
        and controller.pressure_percent == 0.0
        for controller in injector.controllers.values()
    )


def test_off_mode_is_value_identical_and_keeps_kafka_schema(tmp_path):
    original = station_values()
    control_path = tmp_path / "drift_control.json"
    injector = DriftInjector(
        EQUIPMENT_IDS,
        SENSOR_NAMES,
        fast_config(),
        mode="off",
        control_path=control_path,
        now=0.0,
    )
    enqueue_control_command(control_path, "trigger", "all")
    result = injector.process_cycle(original, now=100.0)

    for equipment_id in EQUIPMENT_IDS:
        assert np.array_equal(result[equipment_id], original[equipment_id])
        assert result[equipment_id].tobytes() == original[equipment_id].tobytes()
        message = create_multi_raw_message(
            equipment_id,
            "2026-09-02T12:00:00.000+09:00",
            SENSOR_NAMES,
            result[equipment_id],
        )
        validate_multi_raw_message(message, SENSOR_NAMES)
        assert set(message) == {"equipment_id", "timestamp", "sensors"}
        assert set(message["sensors"]) == set(SENSOR_NAMES)
        assert not {
            "drift",
            "drift_state",
            "offset",
            "progress",
            "temperature_offset",
            "pressure_percent",
            "ground_truth",
        }.intersection(message)
    assert all(
        controller.state is DriftState.NORMAL
        for controller in injector.controllers.values()
    )


def test_active_drift_keeps_kafka_schema_unchanged():
    injector = DriftInjector(
        EQUIPMENT_IDS, SENSOR_NAMES, fast_config(), mode="manual", now=0.0
    )
    injector.controllers["station-01"].trigger(0.0)
    result = injector.process_cycle(station_values(), now=1.0)
    message = create_multi_raw_message(
        "station-01",
        "2026-09-02T12:00:00.000+09:00",
        SENSOR_NAMES,
        result["station-01"],
    )

    validate_multi_raw_message(message, SENSOR_NAMES)
    assert set(message) == {"equipment_id", "timestamp", "sensors"}
    assert list(message["sensors"]) == SENSOR_NAMES


def test_auto_mode_is_reproducible_and_station_schedules_are_independent():
    first = DriftInjector(
        EQUIPMENT_IDS, SENSOR_NAMES, fast_config(), mode="auto", now=0.0
    )
    second = DriftInjector(
        EQUIPMENT_IDS, SENSOR_NAMES, fast_config(), mode="auto", now=0.0
    )
    first_deadlines = [
        first.controllers[equipment_id].next_auto_at
        for equipment_id in EQUIPMENT_IDS
    ]
    second_deadlines = [
        second.controllers[equipment_id].next_auto_at
        for equipment_id in EQUIPMENT_IDS
    ]
    assert first_deadlines == second_deadlines
    assert len(set(first_deadlines)) == len(EQUIPMENT_IDS)

    earliest = min(first_deadlines)
    earliest_station = EQUIPMENT_IDS[first_deadlines.index(earliest)]
    first.process_cycle(station_values(), now=earliest)
    assert sum(
        controller.state is DriftState.RISING
        for controller in first.controllers.values()
    ) == 1
    result = first.process_cycle(station_values(), now=earliest + 0.5)
    controller = first.controllers[earliest_station]
    assert controller.progress == pytest.approx(0.25)
    assert controller.temperature_offset == pytest.approx(0.25)
    assert controller.pressure_percent == pytest.approx(0.75)
    ps2_index = SENSOR_NAMES.index("PS2")
    assert result[earliest_station][ps2_index] == pytest.approx(
        station_values()[earliest_station][ps2_index] * 1.0075
    )


def test_status_file_and_human_readable_status(tmp_path):
    status_path = tmp_path / "drift_status.json"
    injector = DriftInjector(
        EQUIPMENT_IDS,
        SENSOR_NAMES,
        fast_config(),
        mode="manual",
        status_path=status_path,
        now=0.0,
    )
    injector.controllers["station-02"].trigger(0.0)
    injector.process_cycle(station_values(), now=1.0)

    module_path = PROJECT_ROOT / "kafka" / "drift_control.py"
    spec = importlib.util.spec_from_file_location("drift_control", module_path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    output = module.format_status(json.loads(status_path.read_text(encoding="utf-8")))
    assert "station-01  NORMAL" in output
    assert "station-02  RISING" in output
    assert "progress=0.500" in output
    assert "temp_offset=0.500C" in output
    assert "pressure=1.500%" in output
    assert "target_temp=1.000C" in output
    assert "target_pressure=3.000%" in output
    assert f"duration={FAST_EVENT_DURATION_SEC:.3f}s" in output

    legacy_payload = {
        "stations": {
            equipment_id: {"state": "NORMAL", "offset": 0.0}
            for equipment_id in EQUIPMENT_IDS
        }
    }
    legacy_output = module.format_status(legacy_payload)
    assert "progress=0.000 temp_offset=0.000C pressure=0.000%" in legacy_output


def test_malformed_control_json_does_not_stop_injection(tmp_path):
    control_path = tmp_path / "drift_control.json"
    control_path.write_text("{this is not JSON", encoding="utf-8")
    injector = DriftInjector(
        EQUIPMENT_IDS,
        SENSOR_NAMES,
        fast_config(),
        mode="manual",
        control_path=control_path,
        now=0.0,
    )
    messages = []
    result = injector.process_cycle(
        station_values(), now=1.0, warning_handler=messages.append
    )
    assert tuple(result) == EQUIPMENT_IDS
    assert any("malformed" in message for message in messages)
    assert all(
        controller.state is DriftState.NORMAL
        for controller in injector.controllers.values()
    )


@pytest.mark.parametrize(
    "overrides",
    [
        {"min_offset": 2.0, "max_offset": 1.0},
        {"step_per_sec": 0.0},
        {"max_hold_sec": -1.0},
        {"auto_normal_min_sec": 3.0, "auto_normal_max_sec": 2.0},
        {"pressure_min_percent": -0.1},
        {"pressure_min_percent": 3.1, "pressure_max_percent": 3.0},
        {"min_duration_sec": 0.0},
        {"min_duration_sec": 3.0, "max_duration_sec": 2.0},
    ],
)
def test_invalid_config_is_rejected(overrides):
    with pytest.raises(ValueError):
        fast_config(**overrides)


def ranged_config(**overrides):
    values = {
        "min_offset": 1.0,
        "max_offset": 4.0,
        "pressure_min_percent": 0.5,
        "pressure_max_percent": 3.0,
        "min_duration_sec": 20.0,
        "max_duration_sec": 60.0,
        "max_hold_sec": 30.0,
        "auto_normal_min_sec": 10.0,
        "auto_normal_max_sec": 20.0,
        "seed": 9876,
    }
    values.update(overrides)
    return DriftConfig(**values)


def event_sample(controller):
    return (
        controller.target_temp_offset,
        controller.target_pressure_percent,
        controller.event_duration_sec,
    )


def test_event_targets_and_duration_are_randomized_within_bounds_and_reproducible():
    config = ranged_config()
    first = StationDriftController("station-01", config, now=0.0)
    replay = StationDriftController("station-01", config, now=0.0)
    observed = []
    now = 0.0

    for _ in range(3):
        assert first.trigger(now)
        assert replay.trigger(now)
        sample = event_sample(first)
        observed.append(sample)
        assert sample == event_sample(replay)
        assert config.min_offset <= sample[0] <= config.max_offset
        assert config.pressure_min_percent <= sample[1] <= config.pressure_max_percent
        assert config.min_duration_sec <= sample[2] <= config.max_duration_sec
        assert (
            first.rising_duration_sec
            + first.max_hold_duration_sec
            + first.falling_duration_sec
        ) == pytest.approx(sample[2])
        assert sample[0] < config.max_offset
        assert sample[1] < config.pressure_max_percent
        now += sample[2]
        first.update(now)
        replay.update(now)
        assert first.state is DriftState.NORMAL
        assert replay.state is DriftState.NORMAL

    assert len(set(observed)) == 3


def test_station_event_samples_are_independent_and_share_one_progress():
    config = ranged_config()
    injector = DriftInjector(
        EQUIPMENT_IDS, SENSOR_NAMES, config, mode="manual", now=0.0
    )
    for controller in injector.controllers.values():
        assert controller.trigger(0.0)

    samples = {
        equipment_id: event_sample(controller)
        for equipment_id, controller in injector.controllers.items()
    }
    assert len(set(samples.values())) == len(EQUIPMENT_IDS)

    controller = injector.controllers["station-02"]
    result = injector.process_cycle(
        station_values(), now=controller.rising_duration_sec / 2.0
    )
    assert controller.progress == pytest.approx(0.5)
    assert controller.temperature_offset == pytest.approx(
        controller.target_temp_offset * controller.progress
    )
    assert controller.pressure_percent == pytest.approx(
        controller.target_pressure_percent * controller.progress
    )
    ts1_index = SENSOR_NAMES.index("TS1")
    ps1_index = SENSOR_NAMES.index("PS1")
    assert result["station-02"][ts1_index] == pytest.approx(
        station_values()["station-02"][ts1_index]
        + controller.target_temp_offset * 0.5
    )
    assert result["station-02"][ps1_index] == pytest.approx(
        station_values()["station-02"][ps1_index]
        * (1.0 + controller.target_pressure_percent * 0.5 / 100.0)
    )


def test_reset_clears_sampled_event_metadata_and_progress():
    controller = StationDriftController(
        "station-01", ranged_config(), now=0.0
    )
    controller.trigger(0.0)
    controller.update(5.0)
    controller.reset(5.0)

    assert controller.state is DriftState.NORMAL
    assert controller.progress == 0.0
    assert controller.temperature_offset == 0.0
    assert controller.pressure_percent == 0.0
    assert controller.target_temp_offset == 0.0
    assert controller.target_pressure_percent == 0.0
    assert controller.event_duration_sec == 0.0
    assert controller.rising_duration_sec == 0.0
    assert controller.max_hold_duration_sec == 0.0
    assert controller.falling_duration_sec == 0.0


def test_auto_mode_samples_new_values_for_each_event():
    config = ranged_config(
        auto_normal_min_sec=1.0,
        auto_normal_max_sec=1.0,
    )
    controller = StationDriftController(
        "station-01", config, automatic=True, now=0.0
    )

    controller.update(1.0)
    assert controller.state is DriftState.RISING
    first_sample = event_sample(controller)

    event_end = 1.0 + controller.event_duration_sec
    controller.update(event_end)
    assert controller.state is DriftState.NORMAL
    controller.update(controller.next_auto_at)
    assert controller.state is DriftState.RISING
    second_sample = event_sample(controller)

    assert second_sample != first_sample


@pytest.mark.parametrize(
    ("event_duration", "expected_phases"),
    [
        (20.0, (7.0, 6.0, 7.0)),
        (40.0, (14.0, 12.0, 14.0)),
        (60.0, (21.0, 18.0, 21.0)),
    ],
)
def test_total_event_duration_is_split_across_all_phases(
    event_duration, expected_phases
):
    config = ranged_config(
        min_duration_sec=event_duration,
        max_duration_sec=event_duration,
        max_hold_sec=30.0,
    )
    controller = StationDriftController("station-01", config, now=0.0)
    controller.trigger(0.0)

    phases = (
        controller.rising_duration_sec,
        controller.max_hold_duration_sec,
        controller.falling_duration_sec,
    )
    assert phases == pytest.approx(expected_phases)
    assert sum(phases) == pytest.approx(controller.event_duration_sec)
    assert controller.rising_duration_sec == pytest.approx(
        event_duration * RISING_DURATION_RATIO
    )
    assert controller.max_hold_duration_sec == pytest.approx(
        event_duration * MAX_HOLD_DURATION_RATIO
    )
    assert controller.falling_duration_sec == pytest.approx(
        event_duration * FALLING_DURATION_RATIO
    )

    controller.update(event_duration - 0.001)
    assert controller.state is DriftState.FALLING
    controller.update(event_duration)
    assert controller.state is DriftState.NORMAL


def test_hold_cap_redistributes_time_without_extending_total_duration():
    controller = StationDriftController(
        "station-01",
        ranged_config(
            min_duration_sec=20.0,
            max_duration_sec=20.0,
            max_hold_sec=2.0,
        ),
        now=0.0,
    )
    controller.trigger(0.0)

    assert controller.rising_duration_sec == pytest.approx(9.0)
    assert controller.max_hold_duration_sec == pytest.approx(2.0)
    assert controller.falling_duration_sec == pytest.approx(9.0)
    assert (
        controller.rising_duration_sec
        + controller.max_hold_duration_sec
        + controller.falling_duration_sec
    ) == pytest.approx(20.0)
