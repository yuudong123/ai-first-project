import importlib.util
import json
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parents[1]
MODULE_PATH = PROJECT_ROOT / "kafka" / "mock_state_builder.py"
SPEC = importlib.util.spec_from_file_location("mock_state_builder", MODULE_PATH)
MODULE = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(MODULE)


def test_mock_prediction_windows():
    assert MODULE.build_mock_prediction(0, None) == {
        "status": "warming_up",
        "result": None,
        "window_sec": 10,
        "updated_at": None,
    }
    assert MODULE.build_mock_prediction(1, "normal-time")["result"] == "normal"
    assert MODULE.build_mock_prediction(2, "abnormal-time")["result"] == "abnormal"


def test_latest_state_has_exact_schema_and_preserves_raw_values():
    raw = {
        "timestamp": "2026-09-01T12:42:18+09:00",
        "sensors": {f"S{index}": index + 0.25 for index in range(17)},
        "ignored_raw_field": "not forwarded",
    }
    prediction = MODULE.build_mock_prediction(1, "2026-09-01T12:42:20+09:00")
    state = MODULE.build_latest_state(raw, prediction)

    assert set(state) == {"timestamp", "sensors", "prediction"}
    assert state["timestamp"] == raw["timestamp"]
    assert state["sensors"] == raw["sensors"]
    assert state["sensors"] is raw["sensors"]
    assert set(state["prediction"]) == {
        "status", "result", "window_sec", "updated_at"
    }


def test_atomic_write_replaces_complete_json(tmp_path):
    output = tmp_path / "latest_state.json"
    value = {"timestamp": "test", "sensors": {}, "prediction": {}}
    MODULE.atomic_write_json(output, value)

    assert json.loads(output.read_text(encoding="utf-8")) == value
    assert not output.with_name("latest_state.json.tmp").exists()
