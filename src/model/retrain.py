"""HydroTwin 상태 예측 모델의 계절성 드리프트 재학습을 관리한다.

감지된 온도·압력 offset을 검증된 기존 학습 특징에 적용한다. 후보 모델은
원본 데이터와 offset 적용 데이터를 함께 학습하며, 두 환경의 성능 기준을
모두 통과할 때만 운영 모델로 승격한다. 실시간 모델의 예측 결과는 정답
라벨로 사용하지 않는다.

의존 방향은 이 모듈에서 ``hydrotwin_pipeline``을 호출하는 방향으로만 둔다.
핵심 학습 파이프라인이 운영용 재학습 코드를 불러오지 않게 하여 순환 의존을
방지한다.
"""

from __future__ import annotations

import argparse
import importlib
import json
import math
import os
import shutil
from dataclasses import asdict, dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Mapping


PROJECT_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_PRODUCTION_MODEL = (
    PROJECT_ROOT / "models" / "predict" / "integrated_lgbm.joblib"
)
DEFAULT_CANDIDATE_DIR = PROJECT_ROOT / "models" / "predict" / "candidates"
DEFAULT_REPORT_DIR = PROJECT_ROOT / "artifacts" / "retraining"
SEASONAL_SENSOR_NAMES = (
    "PS1", "PS2", "PS3", "PS4", "PS5", "PS6",
    "TS1", "TS2", "TS3", "TS4",
)


@dataclass(frozen=True)
class RetrainConfig:
    production_model_path: Path = DEFAULT_PRODUCTION_MODEL
    candidate_dir: Path = DEFAULT_CANDIDATE_DIR
    report_dir: Path = DEFAULT_REPORT_DIR
    final_window_sec: int = 10
    min_mean_macro_f1: float = 0.80
    max_mean_macro_f1_drop: float = 0.00
    max_target_macro_f1_drop: float = 0.02
    auto_promote: bool = True

    def __post_init__(self) -> None:
        if self.final_window_sec not in {10, 20, 30, 60}:
            raise ValueError("final_window_sec는 10, 20, 30, 60 중 하나여야 합니다.")
        for name in (
            "min_mean_macro_f1",
            "max_mean_macro_f1_drop",
            "max_target_macro_f1_drop",
        ):
            if getattr(self, name) < 0:
                raise ValueError(f"{name}은 음수일 수 없습니다.")


def _pipeline_module():
    return importlib.import_module("src.hydrotwin_pipeline")


def _metric_map(rows: Any) -> dict[str, float]:
    if hasattr(rows, "to_dict"):
        records = rows.to_dict(orient="records")
    else:
        records = list(rows)
    return {
        str(row["target"]): float(row["macro_f1"])
        for row in records
    }


def extract_seasonal_offsets(
    drift_context: Mapping[str, Any],
) -> dict[str, float]:
    """드리프트 결과에서 부호 있는 온도·압력 offset을 추출한다."""
    context: Mapping[str, Any] = drift_context
    if isinstance(context.get("drift"), Mapping):
        context = context["drift"]

    status = context.get("status")
    if status is not None and status != "drift":
        raise ValueError(
            f"재학습에는 확정된 드리프트 이벤트가 필요합니다. status={status}"
        )

    direct = context.get("sensor_offsets")
    if isinstance(direct, Mapping):
        raw_offsets = direct
    else:
        scores = context.get("sensor_scores")
        if not isinstance(scores, Mapping):
            raise ValueError(
                "드리프트 결과에 sensor_offsets 또는 sensor_scores가 필요합니다."
            )
        raw_offsets = {
            sensor: score["mean_offset"]
            for sensor, score in scores.items()
            if (
                sensor in SEASONAL_SENSOR_NAMES
                and isinstance(score, Mapping)
                and score.get("affected", False)
                and "mean_offset" in score
            )
        }

    unknown = set(raw_offsets) - set(SEASONAL_SENSOR_NAMES)
    if unknown:
        raise ValueError(
            "계절성 증강은 압력·온도 센서만 허용합니다: "
            f"{sorted(unknown)}"
        )

    offsets = {sensor: float(value) for sensor, value in raw_offsets.items()}
    if not offsets:
        raise ValueError("계절성 압력·온도 offset을 찾지 못했습니다.")
    if any(not math.isfinite(value) for value in offsets.values()):
        raise ValueError("계절성 offset은 유한한 숫자여야 합니다.")
    if all(abs(value) < 1e-12 for value in offsets.values()):
        raise ValueError("계절성 offset 중 하나 이상은 0이 아니어야 합니다.")
    return offsets


def _context_metrics(
    original: Mapping[str, float],
    seasonal: Mapping[str, float],
) -> dict[str, float]:
    return {
        **{f"original/{target}": value for target, value in original.items()},
        **{f"seasonal/{target}": value for target, value in seasonal.items()},
    }


def promotion_decision(
    candidate_metrics: Mapping[str, float],
    incumbent_metrics: Mapping[str, float] | None,
    config: RetrainConfig,
) -> tuple[bool, list[str]]:
    """후보 모델의 절대 성능 및 기존 모델 대비 성능 저하를 검사한다."""
    if not candidate_metrics:
        return False, ["후보 모델의 성능 지표가 비어 있습니다."]
    if any(not math.isfinite(v) or not 0 <= v <= 1 for v in candidate_metrics.values()):
        return False, ["후보 성능 지표가 유효한 0~1 범위의 숫자가 아닙니다."]
    if incumbent_metrics and set(candidate_metrics) != set(incumbent_metrics):
        return False, ["후보 모델과 운영 모델의 평가 대상이 다릅니다."]

    reasons = []
    candidate_mean = sum(candidate_metrics.values()) / len(candidate_metrics)
    if candidate_mean < config.min_mean_macro_f1:
        reasons.append(
            f"후보 모델 평균 macro_f1 {candidate_mean:.6f}가 기준 "
            f"{config.min_mean_macro_f1:.6f}보다 낮습니다."
        )

    environments: dict[str, list[float]] = {}
    for name, value in candidate_metrics.items():
        if "/" in name:
            environment, _ = name.split("/", 1)
            environments.setdefault(environment, []).append(value)
    for environment, values in sorted(environments.items()):
        environment_mean = sum(values) / len(values)
        if environment_mean < config.min_mean_macro_f1:
            reasons.append(
                f"{environment} 환경 평균 macro_f1 {environment_mean:.6f}가 "
                f"기준 {config.min_mean_macro_f1:.6f}보다 낮습니다."
            )
        if incumbent_metrics:
            old_values = [v for name,v in incumbent_metrics.items() if name.startswith(environment+'/')]
            if old_values and environment_mean < sum(old_values)/len(old_values)-config.max_mean_macro_f1_drop:
                reasons.append(f"{environment} 환경 평균 성능이 운영 모델보다 낮아졌습니다.")

    incumbent = dict(incumbent_metrics or {})
    if incumbent:
        common_targets = sorted(set(candidate_metrics) & set(incumbent))
        if not common_targets:
            reasons.append("후보 모델과 운영 모델에 공통 예측 대상이 없습니다.")
        else:
            incumbent_mean = sum(
                incumbent[target] for target in common_targets
            ) / len(common_targets)
            compared_candidate_mean = sum(
                candidate_metrics[target] for target in common_targets
            ) / len(common_targets)
            if (
                compared_candidate_mean
                < incumbent_mean - config.max_mean_macro_f1_drop
            ):
                reasons.append(
                    "후보 모델의 평균 macro_f1이 운영 모델보다 낮아졌습니다: "
                    f"후보={compared_candidate_mean:.6f}, "
                    f"운영={incumbent_mean:.6f}"
                )
            for target in common_targets:
                if (
                    candidate_metrics[target]
                    < incumbent[target] - config.max_target_macro_f1_drop
                ):
                    reasons.append(
                        f"{target} macro_f1이 운영 모델보다 낮아졌습니다: "
                        f"후보={candidate_metrics[target]:.6f}, "
                        f"운영={incumbent[target]:.6f}"
                    )

    return not reasons, reasons


def _incumbent_metrics(
    model_path: Path,
    *,
    profile: Any,
    splits: Mapping[str, Any],
    processed_dir: Path | str | None,
    seasonal_offsets: Mapping[str, float],
) -> tuple[dict[str, float], dict[str, float]]:
    if not model_path.exists():
        return {}, {}
    pipeline = _pipeline_module()
    bundle = pipeline.load_model_bundle(model_path)
    original, _ = pipeline.evaluate_bundle(
        bundle=bundle,
        profile=profile,
        splits=splits,
        processed_dir=processed_dir,
    )
    seasonal, _ = pipeline.evaluate_bundle(
        bundle=bundle,
        profile=profile,
        splits=splits,
        processed_dir=processed_dir,
        sensor_offsets=seasonal_offsets,
    )
    return _metric_map(original), _metric_map(seasonal)


def _atomic_json(path: Path, value: Mapping[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary_path = path.with_name(path.name + ".tmp")
    temporary_path.write_text(
        json.dumps(value, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    os.replace(temporary_path, path)


def load_drift_report(path: Path | str) -> dict[str, Any]:
    """JSON 결과 하나 또는 JSONL 관측 로그의 최신 행을 불러온다."""
    text = Path(path).read_text(encoding="utf-8").strip()
    if not text:
        raise ValueError(f"드리프트 결과 파일이 비어 있습니다: {path}")
    try:
        value = json.loads(text)
    except json.JSONDecodeError:
        lines = [line for line in text.splitlines() if line.strip()]
        value = json.loads(lines[-1])
    if not isinstance(value, dict):
        raise ValueError("드리프트 결과의 최상위 값은 JSON 객체여야 합니다.")
    return value


def promote_candidate(candidate_path: Path, production_path: Path) -> Path | None:
    """기존 모델을 시각별로 백업하고 후보 모델을 원자적으로 승격한다."""
    production_path.parent.mkdir(parents=True, exist_ok=True)
    backup_path = None
    if production_path.exists():
        timestamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
        backup_path = production_path.with_name(
            f"{production_path.stem}.{timestamp}.backup{production_path.suffix}"
        )
        shutil.copy2(production_path, backup_path)

    temporary_path = production_path.with_name(production_path.name + ".tmp")
    shutil.copy2(candidate_path, temporary_path)
    os.replace(temporary_path, production_path)
    return backup_path


def run_retraining(
    *,
    config: RetrainConfig | None = None,
    profile: Any = None,
    splits: Mapping[str, Any] | None = None,
    processed_dir: Path | str | None = None,
    drift_context: Mapping[str, Any] | None = None,
    seasonal_offsets: Mapping[str, float] | None = None,
) -> dict[str, Any]:
    """기존 검증 데이터를 계절 offset으로 증강하고 후보 모델을 평가·승격한다."""
    cfg = config or RetrainConfig()
    pipeline = _pipeline_module()
    offsets = (
        {sensor: float(value) for sensor, value in seasonal_offsets.items()}
        if seasonal_offsets is not None
        else extract_seasonal_offsets(drift_context or {})
    )
    # 직접 전달받은 offset에도 드리프트 결과에서 추출할 때와 같은 검증을 적용한다.
    offsets = extract_seasonal_offsets({"sensor_offsets": offsets})
    training_profile = profile if profile is not None else pipeline.load_profile()
    training_splits = (
        dict(splits)
        if splits is not None
        else pipeline.make_splits(training_profile)
    )
    pipeline.validate_splits(training_profile, training_splits)

    run_id = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%S%fZ")
    candidate_path = cfg.candidate_dir / f"integrated_lgbm_{run_id}.joblib"
    incumbent_original, incumbent_seasonal = _incumbent_metrics(
        cfg.production_model_path,
        profile=training_profile,
        splits=training_splits,
        processed_dir=processed_dir,
        seasonal_offsets=offsets,
    )

    _, metrics_frame = pipeline.train_integrated_lgbm(
        profile=training_profile,
        splits=training_splits,
        processed_dir=processed_dir,
        model_path=candidate_path,
        final_window_sec=cfg.final_window_sec,
        training_sensor_offsets=offsets,
    )
    candidate_original = _metric_map(metrics_frame)
    candidate_bundle = pipeline.load_model_bundle(candidate_path)
    seasonal_metrics_frame, _ = pipeline.evaluate_bundle(
        bundle=candidate_bundle,
        profile=training_profile,
        splits=training_splits,
        processed_dir=processed_dir,
        sensor_offsets=offsets,
    )
    candidate_seasonal = _metric_map(seasonal_metrics_frame)
    candidate = _context_metrics(candidate_original, candidate_seasonal)
    incumbent = _context_metrics(incumbent_original, incumbent_seasonal)
    accepted, reasons = promotion_decision(candidate, incumbent, cfg)

    backup_path = None
    promoted = bool(accepted and cfg.auto_promote)
    if promoted:
        backup_path = promote_candidate(
            candidate_path,
            cfg.production_model_path,
        )

    report = {
        "run_id": run_id,
        "created_at": datetime.now(timezone.utc).isoformat(),
        "label_source": "verified_historical_profile",
        "live_predictions_used_as_labels": False,
        "seasonal_sensor_offsets": offsets,
        "candidate_path": str(candidate_path),
        "production_model_path": str(cfg.production_model_path),
        "backup_path": str(backup_path) if backup_path else None,
        "accepted": accepted,
        "promoted": promoted,
        "rejection_reasons": reasons,
        "candidate_macro_f1": candidate,
        "incumbent_macro_f1": incumbent,
        "candidate_metrics_by_environment": {
            "original": candidate_original,
            "seasonal": candidate_seasonal,
        },
        "incumbent_metrics_by_environment": {
            "original": incumbent_original,
            "seasonal": incumbent_seasonal,
        },
        "drift_context": dict(drift_context or {}),
        "config": {
            **asdict(cfg),
            "production_model_path": str(cfg.production_model_path),
            "candidate_dir": str(cfg.candidate_dir),
            "report_dir": str(cfg.report_dir),
        },
    }
    report_path = cfg.report_dir / f"retrain_{run_id}.json"
    _atomic_json(report_path, report)
    report["report_path"] = str(report_path)
    return report


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--profile-path",
        type=Path,
        default=None,
        help="검증된 기존 라벨이 들어 있는 profile.txt 경로",
    )
    parser.add_argument("--processed-dir", type=Path, default=None)
    parser.add_argument("--production-model", type=Path, default=DEFAULT_PRODUCTION_MODEL)
    parser.add_argument("--candidate-dir", type=Path, default=DEFAULT_CANDIDATE_DIR)
    parser.add_argument("--report-dir", type=Path, default=DEFAULT_REPORT_DIR)
    parser.add_argument("--window-sec", type=int, default=10)
    parser.add_argument("--min-mean-macro-f1", type=float, default=0.80)
    parser.add_argument("--no-promote", action="store_true")
    parser.add_argument("--drift-report", type=Path, default=None)
    parser.add_argument(
        "--sensor-offset",
        action="append",
        default=[],
        metavar="SENSOR=VALUE",
        help="부호 있는 계절성 offset. 여러 PS/TS 센서는 옵션을 반복합니다.",
    )
    args = parser.parse_args()

    drift_context = None
    if args.drift_report is not None:
        drift_context = load_drift_report(args.drift_report)

    cli_offsets = {}
    for item in args.sensor_offset:
        if "=" not in item:
            parser.error(f"--sensor-offset 형식이 잘못되었습니다: {item}")
        sensor, value = item.split("=", 1)
        cli_offsets[sensor.strip()] = float(value)
    if not cli_offsets and drift_context is None:
        parser.error(
            "--drift-report 또는 하나 이상의 --sensor-offset이 필요합니다."
        )

    config = RetrainConfig(
        production_model_path=args.production_model,
        candidate_dir=args.candidate_dir,
        report_dir=args.report_dir,
        final_window_sec=args.window_sec,
        min_mean_macro_f1=args.min_mean_macro_f1,
        auto_promote=not args.no_promote,
    )
    profile = (
        _pipeline_module().load_profile(args.profile_path)
        if args.profile_path is not None
        else None
    )
    result = run_retraining(
        config=config,
        profile=profile,
        processed_dir=args.processed_dir,
        drift_context=drift_context,
        seasonal_offsets=cli_offsets or None,
    )
    print(json.dumps(result, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
