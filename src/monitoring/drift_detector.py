"""1초 단위 센서값을 수집하고 지속적인 데이터 드리프트를 감지한다.

이 모듈은 Kafka와 예측 모델에 직접 의존하지 않는다. Kafka Consumer가
메시지를 받을 때마다 ``DriftMonitor.update``를 한 번 호출하면 된다.

기준 통계는 실시간 스트림과 동일하게 센서별 초당 값 하나로 집계한 정상·안정
데이터로 생성해야 한다. 제한된 offset은 과거 최솟값과 최댓값 안에 머물 수
있으므로, 표준화 평균 이동과 강건한 기준 범위 이탈 비율을 함께 사용한다.
"""

from __future__ import annotations

import json
import math
import os
from collections import deque
from dataclasses import asdict, dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Iterable, Mapping, Sequence

import numpy as np


DEFAULT_SENSOR_NAMES = (
    "PS1", "PS2", "PS3", "PS4", "PS5", "PS6", "EPS1",
    "FS1", "FS2", "TS1", "TS2", "TS3", "TS4", "VS1",
    "CE", "CP", "SE",
)


@dataclass(frozen=True)
class SensorReference:
    mean: float
    std: float
    lower_bound: float
    upper_bound: float


@dataclass(frozen=True)
class DriftReference:
    sensor_names: tuple[str, ...]
    sample_count: int
    lower_quantile: float
    upper_quantile: float
    sensors: dict[str, SensorReference]
    created_at: str

    def to_dict(self) -> dict[str, Any]:
        return {
            "sensor_names": list(self.sensor_names),
            "sample_count": self.sample_count,
            "lower_quantile": self.lower_quantile,
            "upper_quantile": self.upper_quantile,
            "sensors": {
                name: asdict(reference)
                for name, reference in self.sensors.items()
            },
            "created_at": self.created_at,
        }

    @classmethod
    def from_dict(cls, value: Mapping[str, Any]) -> "DriftReference":
        sensor_names = tuple(value["sensor_names"])
        sensors = {
            name: SensorReference(**value["sensors"][name])
            for name in sensor_names
        }
        return cls(
            sensor_names=sensor_names,
            sample_count=int(value["sample_count"]),
            lower_quantile=float(value["lower_quantile"]),
            upper_quantile=float(value["upper_quantile"]),
            sensors=sensors,
            created_at=str(value["created_at"]),
        )

    def save(self, path: Path | str) -> Path:
        output_path = Path(path)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        temporary_path = output_path.with_name(output_path.name + ".tmp")
        temporary_path.write_text(
            json.dumps(self.to_dict(), ensure_ascii=False, indent=2),
            encoding="utf-8",
        )
        os.replace(temporary_path, output_path)
        return output_path

    @classmethod
    def load(cls, path: Path | str) -> "DriftReference":
        value = json.loads(Path(path).read_text(encoding="utf-8"))
        return cls.from_dict(value)


@dataclass(frozen=True)
class DriftConfig:
    window_size: int = 60
    mean_shift_threshold: float = 1.0
    out_of_bounds_ratio_threshold: float = 0.20
    min_affected_sensors: int = 1
    consecutive_windows: int = 3
    std_floor: float = 1e-6

    def __post_init__(self) -> None:
        if self.window_size < 2:
            raise ValueError("window_size는 2 이상이어야 합니다.")
        if self.mean_shift_threshold <= 0:
            raise ValueError("mean_shift_threshold는 양수여야 합니다.")
        if not 0 <= self.out_of_bounds_ratio_threshold <= 1:
            raise ValueError(
                "out_of_bounds_ratio_threshold는 0 이상 1 이하여야 합니다."
            )
        if self.min_affected_sensors < 1:
            raise ValueError("min_affected_sensors는 1 이상이어야 합니다.")
        if self.consecutive_windows < 1:
            raise ValueError("consecutive_windows는 1 이상이어야 합니다.")


def _sensor_mapping(row: Mapping[str, Any]) -> Mapping[str, Any]:
    sensors = row.get("sensors")
    if isinstance(sensors, Mapping):
        return sensors
    return row


def _validated_values(
    sensors: Mapping[str, Any],
    sensor_names: Sequence[str],
) -> dict[str, float]:
    missing = set(sensor_names) - set(sensors)
    if missing:
        raise ValueError(f"누락된 센서가 있습니다: {sorted(missing)}")

    result: dict[str, float] = {}
    for name in sensor_names:
        value = float(sensors[name])
        if not math.isfinite(value):
            raise ValueError(f"{name} 값은 유한한 숫자여야 합니다.")
        result[name] = value
    return result


def fit_reference(
    rows: Iterable[Mapping[str, Any]],
    sensor_names: Sequence[str] = DEFAULT_SENSOR_NAMES,
    lower_quantile: float = 0.01,
    upper_quantile: float = 0.99,
    min_samples: int = 30,
) -> DriftReference:
    """정상·안정 상태의 1초 단위 데이터로 기준 통계를 생성한다."""
    if not 0 <= lower_quantile < upper_quantile <= 1:
        raise ValueError("분위수는 0 <= lower < upper <= 1을 만족해야 합니다.")

    names = tuple(sensor_names)
    collected = [
        _validated_values(_sensor_mapping(row), names)
        for row in rows
    ]
    if len(collected) < min_samples:
        raise ValueError(
            f"기준 통계에는 최소 {min_samples}개 샘플이 필요합니다. "
            f"현재 샘플 수: {len(collected)}"
        )

    matrix = np.asarray(
        [[row[name] for name in names] for row in collected],
        dtype=np.float64,
    )
    references = {}
    for index, name in enumerate(names):
        values = matrix[:, index]
        references[name] = SensorReference(
            mean=float(np.mean(values)),
            std=float(np.std(values)),
            lower_bound=float(np.quantile(values, lower_quantile)),
            upper_bound=float(np.quantile(values, upper_quantile)),
        )

    return DriftReference(
        sensor_names=names,
        sample_count=len(collected),
        lower_quantile=lower_quantile,
        upper_quantile=upper_quantile,
        sensors=references,
        created_at=datetime.now(timezone.utc).isoformat(),
    )


class RollingDriftDetector:
    """1초 단위 센서값의 이동 구간에서 드리프트를 감지한다."""

    def __init__(
        self,
        reference: DriftReference,
        config: DriftConfig | None = None,
    ) -> None:
        self.reference = reference
        self.config = config or DriftConfig()
        self._window: deque[dict[str, float]] = deque(
            maxlen=self.config.window_size
        )
        self._consecutive_candidates = 0

    @property
    def observed_samples(self) -> int:
        return len(self._window)

    def reset(self) -> None:
        self._window.clear()
        self._consecutive_candidates = 0

    def update(
        self,
        sensors: Mapping[str, Any],
        timestamp: str | None = None,
    ) -> dict[str, Any]:
        values = _validated_values(sensors, self.reference.sensor_names)
        self._window.append(values)
        checked_at = timestamp or datetime.now(timezone.utc).isoformat()

        if len(self._window) < self.config.window_size:
            return {
                "status": "warming_up",
                "drift_detected": False,
                "candidate_detected": False,
                "observed_samples": len(self._window),
                "required_samples": self.config.window_size,
                "consecutive_drift_windows": self._consecutive_candidates,
                "affected_sensors": [],
                "sensor_scores": {},
                "checked_at": checked_at,
            }

        matrix = np.asarray(
            [
                [row[name] for name in self.reference.sensor_names]
                for row in self._window
            ],
            dtype=np.float64,
        )
        sensor_scores: dict[str, dict[str, Any]] = {}
        affected_sensors = []

        for index, name in enumerate(self.reference.sensor_names):
            current = matrix[:, index]
            baseline = self.reference.sensors[name]
            scale = max(abs(baseline.std), self.config.std_floor)
            current_mean = float(np.mean(current))
            mean_offset = current_mean - baseline.mean
            mean_shift = abs(mean_offset) / scale
            outside_ratio = float(
                np.mean(
                    (current < baseline.lower_bound)
                    | (current > baseline.upper_bound)
                )
            )
            affected = (
                mean_shift >= self.config.mean_shift_threshold
                or outside_ratio >= self.config.out_of_bounds_ratio_threshold
            )
            normalized_score = max(
                mean_shift / self.config.mean_shift_threshold,
                outside_ratio / max(
                    self.config.out_of_bounds_ratio_threshold,
                    self.config.std_floor,
                ),
            )
            sensor_scores[name] = {
                "reference_mean": round(baseline.mean, 6),
                "current_mean": round(current_mean, 6),
                "mean_offset": round(mean_offset, 6),
                "mean_shift_score": round(mean_shift, 6),
                "out_of_bounds_ratio": round(outside_ratio, 6),
                "drift_score": round(normalized_score, 6),
                "affected": affected,
            }
            if affected:
                affected_sensors.append(name)

        candidate_detected = (
            len(affected_sensors) >= self.config.min_affected_sensors
        )
        if candidate_detected:
            self._consecutive_candidates += 1
        else:
            self._consecutive_candidates = 0

        drift_detected = (
            self._consecutive_candidates >= self.config.consecutive_windows
        )
        status = (
            "drift"
            if drift_detected
            else "suspected"
            if candidate_detected
            else "stable"
        )

        return {
            "status": status,
            "drift_detected": drift_detected,
            "candidate_detected": candidate_detected,
            "observed_samples": len(self._window),
            "required_samples": self.config.window_size,
            "consecutive_drift_windows": self._consecutive_candidates,
            "affected_sensors": affected_sensors,
            "sensor_scores": sensor_scores,
            "checked_at": checked_at,
        }


class JsonlObservationStore:
    """검증된 실시간 센서값과 드리프트 결과를 JSONL에 누적 저장한다."""

    def __init__(self, path: Path | str) -> None:
        self.path = Path(path)
        self.path.parent.mkdir(parents=True, exist_ok=True)

    def append(
        self,
        timestamp: str,
        sensors: Mapping[str, float],
        drift: Mapping[str, Any],
    ) -> None:
        value = {
            "timestamp": timestamp,
            "sensors": dict(sensors),
            "drift": dict(drift),
        }
        with self.path.open("a", encoding="utf-8") as output:
            output.write(json.dumps(value, ensure_ascii=False) + "\n")


class DriftMonitor:
    """Kafka Consumer에서 메시지마다 호출하는 통합 인터페이스다."""

    def __init__(
        self,
        detector: RollingDriftDetector,
        store: JsonlObservationStore | None = None,
    ) -> None:
        self.detector = detector
        self.store = store

    def update(
        self,
        sensors: Mapping[str, Any],
        timestamp: str | None = None,
    ) -> dict[str, Any]:
        checked_at = timestamp or datetime.now(timezone.utc).isoformat()
        validated = _validated_values(
            sensors,
            self.detector.reference.sensor_names,
        )
        result = self.detector.update(validated, checked_at)
        if self.store is not None:
            self.store.append(checked_at, validated, result)
        return result
