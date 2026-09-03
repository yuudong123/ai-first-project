import importlib.util
import json
from datetime import datetime, timedelta, timezone
from pathlib import Path

import pytest


PROJECT_ROOT = Path(__file__).resolve().parents[1]
MODULE_PATH = PROJECT_ROOT / "kafka" / "prediction_state_builder_multi.py"
SPEC = importlib.util.spec_from_file_location(
    "prediction_state_builder_multi", MODULE_PATH
)
MODULE = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(MODULE)


def iso(second):
    return (
        datetime(2026, 9, 2, tzinfo=timezone.utc) + timedelta(seconds=second)
    ).isoformat()


def make_raw(second=0, station_seconds=None, drift_offset=0.0):
    station_seconds = station_seconds or {}
    return {
        station_id: {
            "timestamp": iso(station_seconds.get(station_id, second)),
            "sensors": {
                sensor_id: (
                    station_index * 100
                    + sensor_index
                    + second
                    + drift_offset
                )
                for sensor_index, sensor_id in enumerate(MODULE.SENSOR_IDS)
            },
        }
        for station_index, station_id in enumerate(MODULE.EQUIPMENT_IDS)
    }


def fake_bundle():
    return {
        "model_type": "LightGBM",
        "window_sec": 20,
        "feature_names": list(MODULE.FEATURE_NAMES),
        "target_order": list(MODULE.MODEL_IDS),
        "component_order": list(MODULE.COMPONENT_IDS),
        "models": {model_id: object() for model_id in MODULE.MODEL_IDS},
    }


def fake_result():
    return {
        "stable_flag": 0,
        "components": {
            "cooler": 100,
            "valve": 100,
            "pump": 0,
            "accumulator": 130,
        },
    }


def update(raw, runtime, predictor, prediction_time="inference-time"):
    buffers, predictions, last_timestamps = runtime
    return MODULE.update_predictions(
        raw,
        buffers,
        predictions,
        last_timestamps,
        fake_bundle(),
        prediction_time,
        predict_fn=predictor,
    )


def test_zero_through_nineteen_samples_are_warming_up():
    runtime = MODULE.create_runtime_state()
    calls = []

    for second in range(19):
        updated = update(
            make_raw(second),
            runtime,
            lambda *args, **kwargs: calls.append((args, kwargs)),
        )
        assert updated == ()

    assert calls == []
    assert all(
        prediction
        == {
            "status": "warming_up",
            "result": None,
            "window_sec": 20,
            "updated_at": None,
        }
        for prediction in runtime[1].values()
    )


def test_exactly_twenty_samples_are_ready_with_ordered_means():
    runtime = MODULE.create_runtime_state()
    feature_rows = []

    def predictor(feature_row, model_bundle):
        assert model_bundle["window_sec"] == 20
        feature_rows.append(feature_row)
        return fake_result()

    for second in range(20):
        updated = update(make_raw(second), runtime, predictor, "ready-at")

    assert updated == MODULE.EQUIPMENT_IDS
    assert len(feature_rows) == 3
    assert tuple(feature_rows[0]) == MODULE.FEATURE_NAMES
    assert feature_rows[0]["PS1_mean"] == pytest.approx(9.5)
    assert feature_rows[0]["SE_mean"] == pytest.approx(25.5)
    for prediction in runtime[1].values():
        assert prediction == {
            "status": "ready",
            "result": fake_result(),
            "window_sec": 20,
            "updated_at": "ready-at",
        }


def test_station_buffers_are_independent_and_ignore_duplicate_timestamps():
    runtime = MODULE.create_runtime_state()
    calls = []

    def predictor(feature_row, model_bundle):
        calls.append(feature_row)
        return fake_result()

    for second in range(20):
        raw = make_raw(
            second,
            station_seconds={
                "station-01": second,
                "station-02": 0,
                "station-03": 0,
            },
        )
        update(raw, runtime, predictor)

    buffers, predictions, _ = runtime
    assert len(buffers["station-01"]) == 20
    assert len(buffers["station-02"]) == 1
    assert len(buffers["station-03"]) == 1
    assert predictions["station-01"]["status"] == "ready"
    assert predictions["station-02"]["status"] == "warming_up"
    assert predictions["station-03"]["status"] == "warming_up"
    assert len(calls) == 1


def test_full_buffer_predicts_for_each_new_rolling_sample():
    runtime = MODULE.create_runtime_state()
    calls = []

    def predictor(feature_row, model_bundle):
        calls.append(feature_row)
        return fake_result()

    for second in range(21):
        update(make_raw(second), runtime, predictor, f"t{second}")

    assert len(calls) == 6
    assert calls[3]["PS1_mean"] == pytest.approx(10.5)
    assert all(value["updated_at"] == "t20" for value in runtime[1].values())


def test_snapshot_preserves_raw_schema_values_and_embeds_full_result():
    raw = make_raw(drift_offset=4.25)
    predictions = {
        station_id: {
            "status": "ready",
            "result": fake_result(),
            "window_sec": 20,
            "updated_at": "now",
        }
        for station_id in MODULE.EQUIPMENT_IDS
    }

    state = MODULE.build_state_snapshot(raw, predictions)

    assert tuple(state) == MODULE.EQUIPMENT_IDS
    for station_id in MODULE.EQUIPMENT_IDS:
        assert set(state[station_id]) == {"timestamp", "sensors", "prediction"}
        assert state[station_id]["timestamp"] == raw[station_id]["timestamp"]
        assert state[station_id]["sensors"] == raw[station_id]["sensors"]
        assert tuple(state[station_id]["sensors"]) == MODULE.SENSOR_IDS
        assert state[station_id]["prediction"]["result"] == fake_result()


@pytest.mark.parametrize(
    "mutate, message",
    [
        (lambda raw: raw.pop("station-03"), "exactly station-01/02/03"),
        (
            lambda raw: raw["station-01"]["sensors"].pop("SE"),
            "exactly all 17 IDs",
        ),
        (
            lambda raw: raw["station-01"]["sensors"].update({"SE": "bad"}),
            "is not numeric",
        ),
    ],
)
def test_malformed_or_missing_raw_snapshot_is_rejected(tmp_path, mutate, message):
    raw = make_raw()
    mutate(raw)
    path = tmp_path / "raw.json"
    path.write_text(json.dumps(raw), encoding="utf-8")

    with pytest.raises(ValueError, match=message):
        MODULE.read_raw_by_equipment(path)


def test_model_bundle_contract_is_source_of_truth():
    validated = MODULE.validate_model_bundle(fake_bundle())
    assert validated["window_sec"] == 20
    assert tuple(validated["feature_names"]) == MODULE.FEATURE_NAMES
    assert tuple(validated["models"]) == MODULE.MODEL_IDS

    wrong = fake_bundle()
    wrong["feature_names"] = list(reversed(MODULE.FEATURE_NAMES))
    with pytest.raises(ValueError, match="feature_names/order"):
        MODULE.validate_model_bundle(wrong)


def test_model_is_loaded_only_once_during_builder_loop(tmp_path):
    input_path = tmp_path / "raw.json"
    output_path = tmp_path / "state.json"
    input_path.write_text(json.dumps(make_raw()), encoding="utf-8")
    loads = []

    def loader(path):
        loads.append(path)
        return fake_bundle()

    writes = MODULE.run_builder(
        input_path,
        output_path,
        model_path=Path("model.joblib"),
        poll_interval=0.01,
        max_updates=3,
        model_loader=loader,
        predict_fn=lambda *_args, **_kwargs: fake_result(),
        monotonic_fn=lambda: 0.0,
        sleep_fn=lambda _seconds: None,
        timestamp_fn=lambda: "now",
    )

    assert writes == 3
    assert loads == [Path("model.joblib")]
    assert json.loads(output_path.read_text(encoding="utf-8"))[
        "station-01"
    ]["prediction"]["status"] == "warming_up"


def test_atomic_write_replaces_named_temporary_file(tmp_path):
    output_path = tmp_path / "state.json"
    MODULE.atomic_write_json(output_path, {"ok": True})
    assert json.loads(output_path.read_text(encoding="utf-8")) == {"ok": True}
    assert not (tmp_path / "state.json.tmp").exists()


def test_real_model_bundle_and_predict_smoke():
    bundle = MODULE.validate_model_bundle(MODULE.load_model_bundle())
    raw = json.loads(
        (PROJECT_ROOT / "kafka" / "latest_raw_by_equipment.json").read_text(
            encoding="utf-8"
        )
    )
    feature_row = {
        feature_name: float(raw["station-01"]["sensors"][sensor_id])
        for sensor_id, feature_name in zip(MODULE.SENSOR_IDS, MODULE.FEATURE_NAMES)
    }

    result = MODULE.predict(feature_row, model_bundle=bundle)

    assert set(result) == {"stable_flag", "components"}
    assert tuple(result["components"]) == MODULE.COMPONENT_IDS
    assert result["stable_flag"] in bundle["class_labels"]["stable_flag"]
    for component_id in MODULE.COMPONENT_IDS:
        assert result["components"][component_id] in bundle["class_labels"][
            component_id
        ]


def test_twenty_sample_builder_window_calls_real_model():
    bundle = MODULE.validate_model_bundle(MODULE.load_model_bundle())
    runtime = MODULE.create_runtime_state()

    for second in range(20):
        buffers, predictions, last_timestamps = runtime
        updated = MODULE.update_predictions(
            make_raw(second),
            buffers,
            predictions,
            last_timestamps,
            bundle,
            "real-inference-time",
        )

    assert updated == MODULE.EQUIPMENT_IDS
    for prediction in runtime[1].values():
        assert prediction["status"] == "ready"
        assert prediction["window_sec"] == 20
        assert prediction["updated_at"] == "real-inference-time"
        assert set(prediction["result"]) == {"stable_flag", "components"}
        assert set(prediction["result"]["components"]) == set(
            MODULE.COMPONENT_IDS
        )
