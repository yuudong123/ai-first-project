from pathlib import Path

import pytest

from src.model.retrain import (
    RetrainConfig,
    extract_seasonal_offsets,
    load_drift_report,
    promotion_decision,
)


def config(tmp_path: Path) -> RetrainConfig:
    return RetrainConfig(
        production_model_path=tmp_path / "production.joblib",
        candidate_dir=tmp_path / "candidates",
        report_dir=tmp_path / "reports",
        min_mean_macro_f1=0.80,
        max_mean_macro_f1_drop=0.00,
        max_target_macro_f1_drop=0.02,
    )


def test_promotion_accepts_non_regressing_candidate(tmp_path):
    accepted, reasons = promotion_decision(
        {"pump": 0.94, "valve": 0.93},
        {"pump": 0.93, "valve": 0.92},
        config(tmp_path),
    )
    assert accepted is True
    assert reasons == []


def test_promotion_rejects_target_regression(tmp_path):
    accepted, reasons = promotion_decision(
        {"pump": 0.90, "valve": 0.70},
        {"pump": 0.91, "valve": 0.90},
        config(tmp_path),
    )
    assert accepted is False
    assert any("valve" in reason for reason in reasons)


def test_promotion_requires_each_environment_to_pass(tmp_path):
    accepted, reasons = promotion_decision(
        {
            "original/pump": 1.00,
            "original/valve": 1.00,
            "seasonal/pump": 0.70,
            "seasonal/valve": 0.70,
        },
        None,
        config(tmp_path),
    )
    assert accepted is False
    assert any("seasonal 환경 평균" in reason for reason in reasons)


def test_extracts_affected_signed_offsets_from_detector_report():
    offsets = extract_seasonal_offsets({
        "sensor_scores": {
            "TS1": {"affected": True, "mean_offset": 4.0},
            "PS1": {"affected": True, "mean_offset": -1.5},
            "FS1": {"affected": True, "mean_offset": 99.0},
            "TS2": {"affected": False, "mean_offset": 2.0},
        }
    })
    assert offsets == {"TS1": 4.0, "PS1": -1.5}


def test_rejects_non_seasonal_sensor_offset():
    with pytest.raises(ValueError, match="압력·온도 센서"):
        extract_seasonal_offsets({"sensor_offsets": {"FS1": 1.0}})


def test_rejects_unconfirmed_drift_event():
    with pytest.raises(ValueError, match="확정된 드리프트"):
        extract_seasonal_offsets({
            "status": "suspected",
            "sensor_scores": {
                "TS1": {"affected": True, "mean_offset": 1.0},
            },
        })


def test_loads_latest_jsonl_drift_record(tmp_path):
    report = tmp_path / "observations.jsonl"
    report.write_text(
        '{"drift":{"sensor_offsets":{"TS1":1.0}}}\n'
        '{"drift":{"sensor_offsets":{"TS1":2.0}}}\n',
        encoding="utf-8",
    )
    loaded = load_drift_report(report)
    assert loaded["drift"]["sensor_offsets"]["TS1"] == 2.0


def test_invalid_metric_cannot_be_promoted(tmp_path):
    accepted, _ = promotion_decision({'pump':float('nan')},None,config(tmp_path))
    assert not accepted


def test_gain_in_seasonal_environment_cannot_hide_original_regression(tmp_path):
    accepted, _ = promotion_decision(
        {'original/pump':.90,'seasonal/pump':.99},
        {'original/pump':.91,'seasonal/pump':.80},config(tmp_path))
    assert not accepted
