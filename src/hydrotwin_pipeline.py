"""
HydroTwin 통합 데이터/모델 파이프라인.

최종 프로젝트 기준
------------------
- 센서 17개
- 입력 특징: 각 센서 mean 1개씩, 총 17개
- 메인 분할: accumulator 90/100/115/130 기준 Stratified 70/15/15
- 모든 타깃이 동일한 train/validation/test cycle_id 사용
- 예측 타깃:
    stable_flag
    cooler
    valve
    pump
    accumulator
- 최종 모델: LightGBM 5개 분류기를 하나의 joblib 번들로 저장
- 모델 파일:
    models/predict/integrated_lgbm.joblib
- 최종 예측 반환 형식:
    {
      "stable_flag": 0,
      "components": {
        "cooler": 100,
        "valve": 100,
        "pump": 0,
        "accumulator": 130
      }
    }

주의
----
stable_flag는 X 입력 특징이 아니라 별도의 y 예측 타깃이다.
"""

from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
from typing import Any, Iterable, Mapping

import joblib
import numpy as np
import pandas as pd

from lightgbm import LGBMClassifier

from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import (
    accuracy_score,
    confusion_matrix,
    f1_score,
)
from sklearn.model_selection import train_test_split


# ============================================================
# 1. 프로젝트 설정
# ============================================================

SENSOR_NAMES = [
    "PS1", "PS2", "PS3", "PS4", "PS5", "PS6",
    "EPS1",
    "FS1", "FS2",
    "TS1", "TS2", "TS3", "TS4",
    "VS1",
    "CE", "CP", "SE",
]

SAMPLING_RATES = {
    "PS1": 100,
    "PS2": 100,
    "PS3": 100,
    "PS4": 100,
    "PS5": 100,
    "PS6": 100,
    "EPS1": 100,
    "FS1": 10,
    "FS2": 10,
    "TS1": 1,
    "TS2": 1,
    "TS3": 1,
    "TS4": 1,
    "VS1": 1,
    "CE": 1,
    "CP": 1,
    "SE": 1,
}

PROFILE_COLUMNS = [
    "cooler",
    "valve",
    "pump",
    "accumulator",
    "stable_flag",
]

COMPONENT_ORDER = [
    "cooler",
    "valve",
    "pump",
    "accumulator",
]

TARGET_ORDER = COMPONENT_ORDER + [
    "stable_flag",
]

MEAN_FEATURE_COLUMNS = [
    f"{sensor}_mean"
    for sensor in SENSOR_NAMES
]

WINDOW_SECONDS = [
    10,
    20,
    30,
    60,
]

EXPECTED_CYCLES = 2205
RANDOM_STATE = 42


def _default_project_root() -> Path:
    """
    src/hydrotwin_pipeline.py 기준으로 프로젝트 루트를 찾는다.
    필요하면 HYDROTWIN_ROOT 환경변수로 덮어쓸 수 있다.
    """
    env_root = os.getenv("HYDROTWIN_ROOT")

    if env_root:
        return Path(env_root).expanduser().resolve()

    return Path(__file__).resolve().parents[1]


PROJECT_ROOT = _default_project_root()

RAW_DIR = (
    PROJECT_ROOT
    / "data"
    / "raw"
    / "uci_hydraulic"
    / "extracted"
)

PROCESSED_DIR = (
    PROJECT_ROOT
    / "data"
    / "processed"
)

MODEL_DIR = (
    PROJECT_ROOT
    / "models"
    / "predict"
)

MODEL_PATH = (
    MODEL_DIR
    / "integrated_lgbm.joblib"
)

SPLIT_PATH = (
    PROCESSED_DIR
    / "split_ids_accumulator_stratified.json"
)


# ============================================================
# 2. 원본 데이터 로드 / 검증
# ============================================================

def load_profile(
    profile_path: Path | str | None = None,
) -> pd.DataFrame:
    """profile.txt를 읽어 cycle_id를 추가한다."""
    path = Path(
        profile_path
        if profile_path is not None
        else RAW_DIR / "profile.txt"
    )

    if not path.exists():
        raise FileNotFoundError(
            f"profile.txt를 찾을 수 없습니다: {path}"
        )

    profile = pd.read_csv(
        path,
        sep=r"\s+",
        header=None,
    )

    if profile.shape[1] != len(PROFILE_COLUMNS):
        raise ValueError(
            "profile.txt 컬럼 수가 예상과 다릅니다. "
            f"예상={len(PROFILE_COLUMNS)}, 실제={profile.shape[1]}"
        )

    profile.columns = PROFILE_COLUMNS

    profile.insert(
        0,
        "cycle_id",
        range(
            1,
            len(profile) + 1,
        ),
    )

    for col in PROFILE_COLUMNS:
        profile[col] = pd.to_numeric(
            profile[col],
            errors="raise",
        ).astype(int)

    return profile


def load_sensor(
    sensor: str,
    raw_dir: Path | str | None = None,
) -> pd.DataFrame:
    """센서 TXT 한 개를 DataFrame으로 읽는다."""
    if sensor not in SENSOR_NAMES:
        raise ValueError(
            f"알 수 없는 센서입니다: {sensor}"
        )

    raw_path = Path(
        raw_dir
        if raw_dir is not None
        else RAW_DIR
    )

    path = raw_path / f"{sensor}.txt"

    if not path.exists():
        raise FileNotFoundError(
            f"센서 파일을 찾을 수 없습니다: {path}"
        )

    return pd.read_csv(
        path,
        sep=r"\s+",
        header=None,
    )


def _count_lines(path: Path) -> int:
    with path.open(
        "r",
        encoding="utf-8",
        errors="ignore",
    ) as f:
        return sum(
            1
            for line in f
            if line.strip()
        )


def validate_raw_files(
    raw_dir: Path | str | None = None,
    expected_cycles: int = EXPECTED_CYCLES,
) -> dict[str, int]:
    """
    센서 17개와 profile.txt가 모두 존재하고
    동일한 사이클 수를 갖는지 확인한다.

    센서 전체 값을 메모리에 올리지 않고 줄 수로 검사한다.
    """
    raw_path = Path(
        raw_dir
        if raw_dir is not None
        else RAW_DIR
    )

    counts: dict[str, int] = {}

    profile_path = raw_path / "profile.txt"

    if not profile_path.exists():
        raise FileNotFoundError(
            f"파일 없음: {profile_path}"
        )

    counts["profile"] = _count_lines(
        profile_path
    )

    for sensor in SENSOR_NAMES:
        path = raw_path / f"{sensor}.txt"

        if not path.exists():
            raise FileNotFoundError(
                f"파일 없음: {path}"
            )

        counts[sensor] = _count_lines(
            path
        )

    unique_counts = set(
        counts.values()
    )

    if len(unique_counts) != 1:
        raise ValueError(
            "센서/profile 사이클 수가 서로 다릅니다: "
            f"{counts}"
        )

    actual_cycles = next(
        iter(unique_counts)
    )

    if expected_cycles is not None:
        if actual_cycles != expected_cycles:
            raise ValueError(
                "사이클 수가 예상과 다릅니다. "
                f"예상={expected_cycles}, 실제={actual_cycles}"
            )

    return counts


def load_raw(
    raw_dir: Path | str | None = None,
    sensors: Iterable[str] | None = None,
) -> dict[str, Any]:
    """
    원본 데이터를 메모리로 읽는다.

    대용량이므로 보통은 extract_all_features()처럼
    센서 파일을 한 개씩 처리하는 함수를 권장한다.
    """
    raw_path = Path(
        raw_dir
        if raw_dir is not None
        else RAW_DIR
    )

    selected = list(
        sensors
        if sensors is not None
        else SENSOR_NAMES
    )

    profile = load_profile(
        raw_path / "profile.txt"
    )

    sensor_data = {
        sensor: load_sensor(
            sensor,
            raw_path,
        )
        for sensor in selected
    }

    return {
        "profile": profile,
        "sensors": sensor_data,
    }


# ============================================================
# 3. Stratified 공통 분할
# ============================================================

def make_splits(
    profile: pd.DataFrame,
    random_state: int = RANDOM_STATE,
) -> dict[str, list[int]]:
    """
    accumulator 원래 4개 클래스(90/100/115/130)를 기준으로
    Stratified 70/15/15 분할을 만든다.

    결과 크기:
    - Train: 1543
    - Validation: 331
    - Test: 331
    """
    required = {
        "cycle_id",
        "accumulator",
    }

    missing = required - set(
        profile.columns
    )

    if missing:
        raise ValueError(
            f"profile에 필요한 컬럼이 없습니다: {sorted(missing)}"
        )

    dev_ids, test_ids = train_test_split(
        profile["cycle_id"],
        test_size=0.15,
        random_state=random_state,
        stratify=profile["accumulator"],
    )

    dev_profile = profile[
        profile["cycle_id"].isin(
            dev_ids
        )
    ].copy()

    train_ids, val_ids = train_test_split(
        dev_profile["cycle_id"],
        test_size=0.15 / 0.85,
        random_state=random_state,
        stratify=dev_profile["accumulator"],
    )

    splits = {
        "train_ids": sorted(
            map(
                int,
                train_ids,
            )
        ),
        "val_ids": sorted(
            map(
                int,
                val_ids,
            )
        ),
        "test_ids": sorted(
            map(
                int,
                test_ids,
            )
        ),
    }

    validate_splits(
        profile,
        splits,
    )

    return splits


def validate_splits(
    profile: pd.DataFrame,
    splits: Mapping[str, Iterable[int]],
) -> None:
    """분할 중복, 전체 합집합, 축압기 클래스 보존을 검사한다."""
    train_set = set(
        map(
            int,
            splits["train_ids"],
        )
    )

    val_set = set(
        map(
            int,
            splits["val_ids"],
        )
    )

    test_set = set(
        map(
            int,
            splits["test_ids"],
        )
    )

    if train_set & val_set:
        raise ValueError(
            "Train/Validation cycle_id가 겹칩니다."
        )

    if train_set & test_set:
        raise ValueError(
            "Train/Test cycle_id가 겹칩니다."
        )

    if val_set & test_set:
        raise ValueError(
            "Validation/Test cycle_id가 겹칩니다."
        )

    all_ids = set(
        map(
            int,
            profile["cycle_id"],
        )
    )

    if (
        train_set
        | val_set
        | test_set
    ) != all_ids:
        raise ValueError(
            "분할 합집합이 전체 cycle_id와 일치하지 않습니다."
        )

    expected_acc_classes = set(
        map(
            int,
            profile["accumulator"].unique(),
        )
    )

    for split_name, id_set in [
        ("Train", train_set),
        ("Validation", val_set),
        ("Test", test_set),
    ]:
        actual_classes = set(
            map(
                int,
                profile.loc[
                    profile["cycle_id"].isin(
                        id_set
                    ),
                    "accumulator",
                ].unique(),
            )
        )

        if actual_classes != expected_acc_classes:
            raise ValueError(
                f"{split_name} 축압기 클래스 누락: "
                f"예상={sorted(expected_acc_classes)}, "
                f"실제={sorted(actual_classes)}"
            )


def save_splits(
    splits: Mapping[str, Iterable[int]],
    path: Path | str | None = None,
) -> Path:
    """공통 분할 cycle_id를 JSON으로 저장한다."""
    output_path = Path(
        path
        if path is not None
        else SPLIT_PATH
    )

    output_path.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    payload = {
        "policy": (
            "accumulator 90/100/115/130 "
            "Stratified 70/15/15"
        ),
        "random_state": RANDOM_STATE,
        "train_ids": list(
            map(
                int,
                splits["train_ids"],
            )
        ),
        "val_ids": list(
            map(
                int,
                splits["val_ids"],
            )
        ),
        "test_ids": list(
            map(
                int,
                splits["test_ids"],
            )
        ),
    }

    output_path.write_text(
        json.dumps(
            payload,
            ensure_ascii=False,
            indent=2,
        ),
        encoding="utf-8",
    )

    return output_path


# ============================================================
# 4. 평균 특징 17개 추출
# ============================================================

def _mean_from_sensor_frame(
    sensor_df: pd.DataFrame,
    sensor: str,
    seconds: int,
) -> pd.Series:
    rate = SAMPLING_RATES[
        sensor
    ]

    n_values = (
        rate
        * int(seconds)
    )

    if sensor_df.shape[1] < n_values:
        raise ValueError(
            f"{sensor}: {seconds}초 특징에 필요한 값이 부족합니다. "
            f"필요={n_values}, 실제={sensor_df.shape[1]}"
        )

    return sensor_df.iloc[
        :,
        :n_values,
    ].mean(
        axis=1
    )


def extract_features(
    seconds: int,
    raw_dir: Path | str | None = None,
    output_path: Path | str | None = None,
) -> pd.DataFrame:
    """
    지정된 입력 시간에 대해 센서별 평균 17개만 만든다.
    """
    if seconds not in WINDOW_SECONDS:
        raise ValueError(
            f"seconds는 {WINDOW_SECONDS} 중 하나여야 합니다."
        )

    raw_path = Path(
        raw_dir
        if raw_dir is not None
        else RAW_DIR
    )

    profile = load_profile(
        raw_path
        / "profile.txt"
    )

    features = pd.DataFrame({
        "cycle_id":
            profile["cycle_id"].astype(int)
    })

    for sensor in SENSOR_NAMES:
        sensor_df = load_sensor(
            sensor,
            raw_path,
        )

        if len(sensor_df) != len(profile):
            raise ValueError(
                f"{sensor}와 profile 사이클 수 불일치"
            )

        features[
            f"{sensor}_mean"
        ] = _mean_from_sensor_frame(
            sensor_df,
            sensor,
            seconds,
        ).to_numpy()

    expected_columns = [
        "cycle_id",
        *MEAN_FEATURE_COLUMNS,
    ]

    features = features[
        expected_columns
    ]

    if output_path is not None:
        path = Path(
            output_path
        )

        path.parent.mkdir(
            parents=True,
            exist_ok=True,
        )

        features.to_parquet(
            path,
            index=False,
        )

    return features


def extract_all_features(
    raw_dir: Path | str | None = None,
    processed_dir: Path | str | None = None,
) -> dict[int, pd.DataFrame]:
    """
    센서 파일을 센서별로 한 번씩만 읽으면서
    10/20/30/60초 평균 특징 파일을 모두 생성한다.
    """
    raw_path = Path(
        raw_dir
        if raw_dir is not None
        else RAW_DIR
    )

    processed_path = Path(
        processed_dir
        if processed_dir is not None
        else PROCESSED_DIR
    )

    processed_path.mkdir(
        parents=True,
        exist_ok=True,
    )

    profile = load_profile(
        raw_path
        / "profile.txt"
    )

    outputs = {
        seconds: pd.DataFrame({
            "cycle_id":
                profile["cycle_id"].astype(int)
        })
        for seconds in WINDOW_SECONDS
    }

    for sensor in SENSOR_NAMES:
        sensor_df = load_sensor(
            sensor,
            raw_path,
        )

        if len(sensor_df) != len(profile):
            raise ValueError(
                f"{sensor}와 profile 사이클 수 불일치"
            )

        for seconds in WINDOW_SECONDS:
            outputs[
                seconds
            ][
                f"{sensor}_mean"
            ] = _mean_from_sensor_frame(
                sensor_df,
                sensor,
                seconds,
            ).to_numpy()

    expected_columns = [
        "cycle_id",
        *MEAN_FEATURE_COLUMNS,
    ]

    for seconds, df in outputs.items():
        outputs[
            seconds
        ] = df[
            expected_columns
        ]

        output_path = (
            processed_path
            / f"features_{seconds}s.parquet"
        )

        outputs[
            seconds
        ].to_parquet(
            output_path,
            index=False,
        )

    return outputs


def build_replay_test(
    test_ids: Iterable[int],
    raw_dir: Path | str | None = None,
    processed_dir: Path | str | None = None,
) -> pd.DataFrame:
    """
    Test cycle을 1초 단위로 재생할 replay_test.parquet 생성.

    정답 라벨은 넣지 않는다.
    각 센서는 해당 1초 구간의 평균값으로 맞춘다.
    """
    raw_path = Path(
        raw_dir
        if raw_dir is not None
        else RAW_DIR
    )

    processed_path = Path(
        processed_dir
        if processed_dir is not None
        else PROCESSED_DIR
    )

    processed_path.mkdir(
        parents=True,
        exist_ok=True,
    )

    test_ids = sorted(
        map(
            int,
            test_ids,
        )
    )

    n_test = len(
        test_ids
    )

    replay = pd.DataFrame({
        "cycle_id":
            np.repeat(
                test_ids,
                60,
            ),
        "second":
            np.tile(
                np.arange(
                    1,
                    61,
                ),
                n_test,
            ),
    })

    # cycle_id가 1부터 시작하고 원본 센서 행 순서와 일치한다.
    row_indices = np.asarray(
        test_ids,
        dtype=int,
    ) - 1

    for sensor in SENSOR_NAMES:
        rate = SAMPLING_RATES[
            sensor
        ]

        sensor_df = load_sensor(
            sensor,
            raw_path,
        )

        required_values = (
            60
            * rate
        )

        if sensor_df.shape[1] < required_values:
            raise ValueError(
                f"{sensor}: 60초 재생 데이터 값이 부족합니다."
            )

        values = sensor_df.iloc[
            row_indices,
            :required_values,
        ].to_numpy(
            dtype=float,
        )

        second_values = values.reshape(
            n_test,
            60,
            rate,
        ).mean(
            axis=2
        )

        replay[
            sensor
        ] = second_values.reshape(
            -1
        )

    replay_path = (
        processed_path
        / "replay_test.parquet"
    )

    replay.to_parquet(
        replay_path,
        index=False,
    )

    return replay


def save_test_labels(
    profile: pd.DataFrame,
    test_ids: Iterable[int],
    processed_dir: Path | str | None = None,
) -> pd.DataFrame:
    """Test 정답을 replay와 분리해 저장한다."""
    processed_path = Path(
        processed_dir
        if processed_dir is not None
        else PROCESSED_DIR
    )

    processed_path.mkdir(
        parents=True,
        exist_ok=True,
    )

    test_id_set = set(
        map(
            int,
            test_ids,
        )
    )

    labels = (
        profile[
            profile["cycle_id"].isin(
                test_id_set
            )
        ][
            [
                "cycle_id",
                *TARGET_ORDER,
            ]
        ]
        .sort_values(
            "cycle_id"
        )
        .reset_index(
            drop=True
        )
    )

    labels.to_parquet(
        processed_path
        / "test_labels.parquet",
        index=False,
    )

    return labels


def preprocess_all(
    raw_dir: Path | str | None = None,
    processed_dir: Path | str | None = None,
) -> dict[str, Any]:
    """
    한 번에:
    - 원본 검증
    - 공통 분할
    - 10/20/30/60초 평균 특징
    - replay_test
    - test_labels
    를 만든다.
    """
    raw_path = Path(
        raw_dir
        if raw_dir is not None
        else RAW_DIR
    )

    processed_path = Path(
        processed_dir
        if processed_dir is not None
        else PROCESSED_DIR
    )

    validate_raw_files(
        raw_path
    )

    profile = load_profile(
        raw_path
        / "profile.txt"
    )

    splits = make_splits(
        profile
    )

    split_path = save_splits(
        splits,
        processed_path
        / SPLIT_PATH.name,
    )

    features = extract_all_features(
        raw_path,
        processed_path,
    )

    replay = build_replay_test(
        splits["test_ids"],
        raw_path,
        processed_path,
    )

    test_labels = save_test_labels(
        profile,
        splits["test_ids"],
        processed_path,
    )

    return {
        "profile": profile,
        "splits": splits,
        "split_path": split_path,
        "features": features,
        "replay_test": replay,
        "test_labels": test_labels,
    }


# ============================================================
# 5. 학습용 데이터 준비
# ============================================================

def load_feature_file(
    seconds: int,
    processed_dir: Path | str | None = None,
) -> pd.DataFrame:
    if seconds not in WINDOW_SECONDS:
        raise ValueError(
            f"seconds는 {WINDOW_SECONDS} 중 하나여야 합니다."
        )

    processed_path = Path(
        processed_dir
        if processed_dir is not None
        else PROCESSED_DIR
    )

    path = (
        processed_path
        / f"features_{seconds}s.parquet"
    )

    if not path.exists():
        raise FileNotFoundError(
            f"특징 파일 없음: {path}"
        )

    features = pd.read_parquet(
        path
    )

    missing = set(
        [
            "cycle_id",
            *MEAN_FEATURE_COLUMNS,
        ]
    ) - set(
        features.columns
    )

    if missing:
        raise ValueError(
            f"{path.name}에 필요한 컬럼이 없습니다: "
            f"{sorted(missing)}"
        )

    return features


def get_xy(
    features: pd.DataFrame,
    profile: pd.DataFrame,
    ids: Iterable[int],
    target: str,
) -> tuple[pd.DataFrame, pd.Series]:
    if target not in TARGET_ORDER:
        raise ValueError(
            f"알 수 없는 타깃: {target}"
        )

    id_set = set(
        map(
            int,
            ids,
        )
    )

    feature_part = (
        features[
            features["cycle_id"].isin(
                id_set
            )
        ][
            [
                "cycle_id",
                *MEAN_FEATURE_COLUMNS,
            ]
        ]
        .copy()
    )

    label_part = (
        profile[
            profile["cycle_id"].isin(
                id_set
            )
        ][
            [
                "cycle_id",
                target,
            ]
        ]
        .copy()
    )

    merged = (
        feature_part
        .merge(
            label_part,
            on="cycle_id",
            how="inner",
            validate="one_to_one",
        )
        .sort_values(
            "cycle_id"
        )
        .reset_index(
            drop=True
        )
    )

    if len(merged) != len(id_set):
        raise ValueError(
            f"{target}: 특징/라벨 cycle_id 정렬 실패 "
            f"예상={len(id_set)}, 실제={len(merged)}"
        )

    X = merged[
        MEAN_FEATURE_COLUMNS
    ].copy()

    y = merged[
        target
    ].copy()

    return X, y


# ============================================================
# 6. RandomForest / LightGBM 비교
# ============================================================

def compare_models(
    features: pd.DataFrame,
    profile: pd.DataFrame,
    train_ids: Iterable[int],
    val_ids: Iterable[int],
    window_sec: int,
) -> pd.DataFrame:
    """
    같은 분할과 지표로 RandomForest와 LightGBM 비교.
    Test는 사용하지 않는다.
    """
    results: list[
        dict[str, Any]
    ] = []

    for target in TARGET_ORDER:
        X_train, y_train = get_xy(
            features,
            profile,
            train_ids,
            target,
        )

        X_val, y_val = get_xy(
            features,
            profile,
            val_ids,
            target,
        )

        candidates = {
            "RandomForest":
                RandomForestClassifier(
                    n_estimators=200,
                    random_state=RANDOM_STATE,
                    n_jobs=-1,
                ),
            "LightGBM":
                LGBMClassifier(
                    n_estimators=200,
                    random_state=RANDOM_STATE,
                    n_jobs=-1,
                    verbosity=-1,
                ),
        }

        for model_name, model in candidates.items():
            model.fit(
                X_train,
                y_train,
            )

            pred = model.predict(
                X_val
            )

            results.append({
                "window_sec":
                    int(window_sec),
                "target":
                    target,
                "model":
                    model_name,
                "accuracy":
                    float(
                        accuracy_score(
                            y_val,
                            pred,
                        )
                    ),
                "macro_f1":
                    float(
                        f1_score(
                            y_val,
                            pred,
                            average="macro",
                            zero_division=0,
                        )
                    ),
            })

    return pd.DataFrame(
        results
    )


# ============================================================
# 7. 입력 시간 선택
# ============================================================

def evaluate_lgbm_window(
    features: pd.DataFrame,
    profile: pd.DataFrame,
    train_ids: Iterable[int],
    val_ids: Iterable[int],
    seconds: int,
) -> pd.DataFrame:
    rows: list[
        dict[str, Any]
    ] = []

    for target in TARGET_ORDER:
        X_train, y_train = get_xy(
            features,
            profile,
            train_ids,
            target,
        )

        X_val, y_val = get_xy(
            features,
            profile,
            val_ids,
            target,
        )

        model = LGBMClassifier(
            n_estimators=200,
            random_state=RANDOM_STATE,
            n_jobs=-1,
            verbosity=-1,
        )

        model.fit(
            X_train,
            y_train,
        )

        pred = model.predict(
            X_val
        )

        rows.append({
            "window_sec":
                int(seconds),
            "target":
                target,
            "accuracy":
                float(
                    accuracy_score(
                        y_val,
                        pred,
                    )
                ),
            "macro_f1":
                float(
                    f1_score(
                        y_val,
                        pred,
                        average="macro",
                        zero_division=0,
                    )
                ),
        })

    return pd.DataFrame(
        rows
    )


def select_final_window(
    profile: pd.DataFrame,
    splits: Mapping[str, Iterable[int]],
    processed_dir: Path | str | None = None,
) -> tuple[int, pd.DataFrame]:
    """
    최신 노트북 기준 선택 규칙:
    - 20초와 60초 LightGBM Validation Macro F1 비교
    - 20초가 60초 성능의 95% 이상이면 20초
    - 아니면 60초
    """
    rows = []

    for seconds in [
        20,
        60,
    ]:
        features = load_feature_file(
            seconds,
            processed_dir,
        )

        result = evaluate_lgbm_window(
            features,
            profile,
            splits["train_ids"],
            splits["val_ids"],
            seconds,
        )

        rows.append(
            result
        )

    window_results = pd.concat(
        rows,
        ignore_index=True,
    )

    summary = (
        window_results
        .groupby(
            "window_sec"
        )[
            [
                "accuracy",
                "macro_f1",
            ]
        ]
        .mean()
    )

    f1_20 = float(
        summary.loc[
            20,
            "macro_f1",
        ]
    )

    f1_60 = float(
        summary.loc[
            60,
            "macro_f1",
        ]
    )

    performance_ratio = (
        f1_20
        / f1_60
        if f1_60 > 0
        else 0.0
    )

    final_window_sec = (
        20
        if performance_ratio >= 0.95
        else 60
    )

    return (
        final_window_sec,
        window_results,
    )


# ============================================================
# 8. LightGBM 통합모델 학습 / 저장
# ============================================================

def train_integrated_lgbm(
    profile: pd.DataFrame | None = None,
    splits: Mapping[str, Iterable[int]] | None = None,
    processed_dir: Path | str | None = None,
    model_path: Path | str | None = None,
    final_window_sec: int | None = None,
) -> tuple[dict[str, Any], pd.DataFrame]:
    """
    5개 LightGBM 분류기를 학습한 뒤
    하나의 integrated_lgbm.joblib 파일에 묶어 저장한다.

    별도의 target별 joblib 파일은 만들지 않는다.
    """
    profile = (
        profile.copy()
        if profile is not None
        else load_profile()
    )

    splits = (
        dict(splits)
        if splits is not None
        else make_splits(
            profile
        )
    )

    if final_window_sec is None:
        final_window_sec, window_results = (
            select_final_window(
                profile,
                splits,
                processed_dir,
            )
        )
    else:
        if final_window_sec not in WINDOW_SECONDS:
            raise ValueError(
                f"final_window_sec는 {WINDOW_SECONDS} 중 하나여야 합니다."
            )

        window_results = pd.DataFrame()

    features = load_feature_file(
        final_window_sec,
        processed_dir,
    )

    train_val_ids = sorted(
        set(
            map(
                int,
                splits["train_ids"],
            )
        )
        | set(
            map(
                int,
                splits["val_ids"],
            )
        )
    )

    final_models: dict[
        str,
        LGBMClassifier,
    ] = {}

    test_rows: list[
        dict[str, Any]
    ] = []

    confusion_matrices: dict[
        str,
        list[list[int]],
    ] = {}

    for target in TARGET_ORDER:
        X_train_val, y_train_val = get_xy(
            features,
            profile,
            train_val_ids,
            target,
        )

        X_test, y_test = get_xy(
            features,
            profile,
            splits["test_ids"],
            target,
        )

        model = LGBMClassifier(
            n_estimators=200,
            random_state=RANDOM_STATE,
            n_jobs=-1,
            verbosity=-1,
        )

        model.fit(
            X_train_val,
            y_train_val,
        )

        pred = model.predict(
            X_test
        )

        labels = sorted(
            map(
                int,
                profile[
                    target
                ].unique(),
            )
        )

        accuracy = float(
            accuracy_score(
                y_test,
                pred,
            )
        )

        macro_f1 = float(
            f1_score(
                y_test,
                pred,
                labels=labels,
                average="macro",
                zero_division=0,
            )
        )

        cm = confusion_matrix(
            y_test,
            pred,
            labels=labels,
        )

        final_models[
            target
        ] = model

        test_rows.append({
            "target":
                target,
            "window_sec":
                int(
                    final_window_sec
                ),
            "accuracy":
                accuracy,
            "macro_f1":
                macro_f1,
            "test_count":
                int(
                    len(
                        y_test
                    )
                ),
        })

        confusion_matrices[
            target
        ] = (
            cm
            .astype(int)
            .tolist()
        )

    test_metrics = pd.DataFrame(
        test_rows
    )

    class_labels = {
        target: sorted(
            map(
                int,
                profile[
                    target
                ].unique(),
            )
        )
        for target in TARGET_ORDER
    }

    bundle = {
        "model_type":
            "LightGBM",
        "bundle_version":
            1,
        "models":
            final_models,
        "feature_names":
            list(
                MEAN_FEATURE_COLUMNS
            ),
        "window_sec":
            int(
                final_window_sec
            ),
        "target_order":
            list(
                TARGET_ORDER
            ),
        "component_order":
            list(
                COMPONENT_ORDER
            ),
        "class_labels":
            class_labels,
        "split_policy": {
            "basis":
                "accumulator",
            "classes":
                [
                    90,
                    100,
                    115,
                    130,
                ],
            "method":
                "Stratified 70/15/15",
            "random_state":
                RANDOM_STATE,
        },
        # metadata.json을 따로 만들지 않고 모델 파일 안에 포함한다.
        "metadata": {
            "feature_count":
                len(
                    MEAN_FEATURE_COLUMNS
                ),
            "feature_names":
                list(
                    MEAN_FEATURE_COLUMNS
                ),
            "window_sec":
                int(
                    final_window_sec
                ),
            "test_metrics":
                test_metrics.to_dict(
                    orient="records"
                ),
            "confusion_matrices":
                confusion_matrices,
            "window_validation":
                (
                    window_results
                    .to_dict(
                        orient="records"
                    )
                    if not window_results.empty
                    else []
                ),
        },
    }

    output_path = Path(
        model_path
        if model_path is not None
        else MODEL_PATH
    )

    output_path.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    joblib.dump(
        bundle,
        output_path,
    )

    return (
        bundle,
        test_metrics,
    )


def load_model_bundle(
    model_path: Path | str | None = None,
) -> dict[str, Any]:
    path = Path(
        model_path
        if model_path is not None
        else MODEL_PATH
    )

    if not path.exists():
        raise FileNotFoundError(
            f"통합모델 파일 없음: {path}"
        )

    bundle = joblib.load(
        path
    )

    required_keys = {
        "models",
        "feature_names",
        "window_sec",
        "component_order",
    }

    missing = required_keys - set(
        bundle.keys()
    )

    if missing:
        raise ValueError(
            "통합모델 번들에 필요한 키가 없습니다: "
            f"{sorted(missing)}"
        )

    return bundle


# ============================================================
# 9. 평가
# ============================================================

def evaluate_bundle(
    bundle: Mapping[str, Any] | None = None,
    profile: pd.DataFrame | None = None,
    splits: Mapping[str, Iterable[int]] | None = None,
    processed_dir: Path | str | None = None,
) -> tuple[pd.DataFrame, dict[str, pd.DataFrame]]:
    bundle = (
        dict(bundle)
        if bundle is not None
        else load_model_bundle()
    )

    profile = (
        profile.copy()
        if profile is not None
        else load_profile()
    )

    splits = (
        dict(splits)
        if splits is not None
        else make_splits(
            profile
        )
    )

    seconds = int(
        bundle["window_sec"]
    )

    features = load_feature_file(
        seconds,
        processed_dir,
    )

    rows = []
    matrices: dict[
        str,
        pd.DataFrame,
    ] = {}

    target_order = bundle.get(
        "target_order",
        [
            *bundle.get(
                "component_order",
                COMPONENT_ORDER,
            ),
            "stable_flag",
        ],
    )

    for target in target_order:
        if target not in bundle[
            "models"
        ]:
            continue

        X_test, y_test = get_xy(
            features,
            profile,
            splits["test_ids"],
            target,
        )

        pred = bundle[
            "models"
        ][
            target
        ].predict(
            X_test
        )

        labels = sorted(
            map(
                int,
                profile[
                    target
                ].unique(),
            )
        )

        rows.append({
            "target":
                target,
            "window_sec":
                seconds,
            "accuracy":
                float(
                    accuracy_score(
                        y_test,
                        pred,
                    )
                ),
            "macro_f1":
                float(
                    f1_score(
                        y_test,
                        pred,
                        labels=labels,
                        average="macro",
                        zero_division=0,
                    )
                ),
            "test_count":
                int(
                    len(
                        y_test
                    )
                ),
        })

        cm = confusion_matrix(
            y_test,
            pred,
            labels=labels,
        )

        matrices[
            target
        ] = pd.DataFrame(
            cm,
            index=[
                f"actual_{label}"
                for label in labels
            ],
            columns=[
                f"pred_{label}"
                for label in labels
            ],
        )

    return (
        pd.DataFrame(
            rows
        ),
        matrices,
    )


# ============================================================
# 10. 공통 predict()
# ============================================================

def _as_feature_frame(
    feature_row: pd.DataFrame
    | pd.Series
    | Mapping[str, Any],
    feature_names: list[str],
) -> pd.DataFrame:
    if isinstance(
        feature_row,
        pd.DataFrame,
    ):
        frame = feature_row.copy()

    elif isinstance(
        feature_row,
        pd.Series,
    ):
        frame = feature_row.to_frame().T

    elif isinstance(
        feature_row,
        Mapping,
    ):
        frame = pd.DataFrame(
            [
                dict(
                    feature_row
                )
            ]
        )

    else:
        raise TypeError(
            "feature_row는 DataFrame, Series 또는 dict여야 합니다."
        )

    if len(frame) != 1:
        raise ValueError(
            "predict()에는 1개 샘플만 전달해야 합니다."
        )

    missing = set(
        feature_names
    ) - set(
        frame.columns
    )

    if missing:
        raise ValueError(
            "예측 입력에 필요한 특징이 없습니다: "
            f"{sorted(missing)}"
        )

    return frame[
        feature_names
    ].copy()


def predict(
    feature_row: pd.DataFrame
    | pd.Series
    | Mapping[str, Any],
    model_bundle: Mapping[str, Any] | None = None,
    model_path: Path | str | None = None,
) -> dict[str, Any]:
    """
    통합 LightGBM 파일 하나를 사용해 5개 결과를 예측한다.

    반환 구조는 고정:
    {
      "stable_flag": 0,
      "components": {
        "cooler": 100,
        "valve": 100,
        "pump": 0,
        "accumulator": 130
      }
    }
    """
    bundle = (
        dict(
            model_bundle
        )
        if model_bundle is not None
        else load_model_bundle(
            model_path
        )
    )

    feature_names = list(
        bundle[
            "feature_names"
        ]
    )

    X_input = _as_feature_frame(
        feature_row,
        feature_names,
    )

    models = bundle[
        "models"
    ]

    if "stable_flag" not in models:
        raise ValueError(
            "현재 통합모델에 stable_flag 모델이 없습니다. "
            "stable_flag 포함 5개 타깃으로 다시 학습해야 합니다."
        )

    stable_pred = int(
        models[
            "stable_flag"
        ].predict(
            X_input
        )[0]
    )

    component_predictions = {}

    for component in bundle.get(
        "component_order",
        COMPONENT_ORDER,
    ):
        component_predictions[
            component
        ] = int(
            models[
                component
            ].predict(
                X_input
            )[0]
        )

    return {
        "stable_flag":
            stable_pred,
        "components":
            component_predictions,
    }


def predict_json(
    feature_row: pd.DataFrame
    | pd.Series
    | Mapping[str, Any],
    model_bundle: Mapping[str, Any] | None = None,
    model_path: Path | str | None = None,
) -> str:
    return json.dumps(
        predict(
            feature_row,
            model_bundle=model_bundle,
            model_path=model_path,
        ),
        ensure_ascii=False,
        indent=2,
    )


# ============================================================
# 11. SHAP 설명
# ============================================================

def _mean_abs_shap(
    shap_values: Any,
    n_features: int,
) -> np.ndarray:
    if isinstance(
        shap_values,
        list,
    ):
        stacked = np.stack(
            [
                np.abs(
                    np.asarray(
                        values
                    )
                )
                for values
                in shap_values
            ],
            axis=0,
        )

        return stacked.mean(
            axis=(0, 1)
        )

    shap_array = np.abs(
        np.asarray(
            shap_values
        )
    )

    if shap_array.ndim == 2:
        return shap_array.mean(
            axis=0
        )

    if shap_array.ndim == 3:
        feature_axes = [
            axis
            for axis, size
            in enumerate(
                shap_array.shape
            )
            if size == n_features
        ]

        if not feature_axes:
            raise ValueError(
                "SHAP 특징 축을 찾지 못했습니다."
            )

        # 같은 크기의 축이 여러 개면 마지막 축을 특징 축으로 우선한다.
        feature_axis = feature_axes[
            -1
        ]

        reduce_axes = tuple(
            axis
            for axis
            in range(
                shap_array.ndim
            )
            if axis != feature_axis
        )

        return shap_array.mean(
            axis=reduce_axes
        )

    raise ValueError(
        f"예상하지 못한 SHAP 배열 차원: {shap_array.ndim}"
    )


def explain(
    feature_rows: pd.DataFrame,
    model_bundle: Mapping[str, Any] | None = None,
    model_path: Path | str | None = None,
    top_n: int = 5,
) -> dict[str, list[dict[str, Any]]]:
    """
    통합 LightGBM 내부 각 타깃 모델의
    평균 |SHAP| 상위 특징을 JSON 가능한 dict로 반환한다.
    """
    try:
        import shap
    except ImportError as exc:
        raise ImportError(
            "SHAP 설명을 사용하려면 shap 패키지가 필요합니다."
        ) from exc

    bundle = (
        dict(
            model_bundle
        )
        if model_bundle is not None
        else load_model_bundle(
            model_path
        )
    )

    feature_names = list(
        bundle[
            "feature_names"
        ]
    )

    missing = set(
        feature_names
    ) - set(
        feature_rows.columns
    )

    if missing:
        raise ValueError(
            f"SHAP 입력 특징 누락: {sorted(missing)}"
        )

    X = feature_rows[
        feature_names
    ].copy()

    result: dict[
        str,
        list[
            dict[
                str,
                Any,
            ]
        ],
    ] = {}

    target_order = bundle.get(
        "target_order",
        [
            *bundle.get(
                "component_order",
                COMPONENT_ORDER,
            ),
            "stable_flag",
        ],
    )

    for target in target_order:
        if target not in bundle[
            "models"
        ]:
            continue

        model = bundle[
            "models"
        ][
            target
        ]

        explainer = shap.TreeExplainer(
            model
        )

        shap_values = explainer.shap_values(
            X
        )

        mean_abs = _mean_abs_shap(
            shap_values,
            len(
                feature_names
            ),
        )

        importance = (
            pd.DataFrame({
                "feature":
                    feature_names,
                "impact":
                    mean_abs,
            })
            .sort_values(
                "impact",
                ascending=False,
            )
            .head(
                int(
                    top_n
                )
            )
        )

        result[
            target
        ] = [
            {
                "feature":
                    str(
                        row.feature
                    ),
                "impact":
                    round(
                        float(
                            row.impact
                        ),
                        6,
                    ),
            }
            for row
            in importance.itertuples(
                index=False
            )
        ]

    return result


# ============================================================
# 12. CLI용 main 함수
# ============================================================

def main_load_raw() -> None:
    counts = validate_raw_files()

    print(
        "원본 데이터 검사 통과"
    )

    print(
        "사이클 수:",
        next(
            iter(
                counts.values()
            )
        ),
    )

    print(
        "센서 수:",
        len(
            SENSOR_NAMES
        ),
    )


def main_make_splits() -> None:
    profile = load_profile()

    splits = make_splits(
        profile
    )

    path = save_splits(
        splits
    )

    print(
        "Train:",
        len(
            splits[
                "train_ids"
            ]
        ),
    )

    print(
        "Validation:",
        len(
            splits[
                "val_ids"
            ]
        ),
    )

    print(
        "Test:",
        len(
            splits[
                "test_ids"
            ]
        ),
    )

    print(
        "분할 저장:",
        path,
    )


def main_extract() -> None:
    result = preprocess_all()

    print(
        "특징 파일 생성 완료"
    )

    for seconds, df in result[
        "features"
    ].items():
        print(
            f"{seconds}초:",
            df.shape,
        )

    print(
        "replay_test:",
        result[
            "replay_test"
        ].shape,
    )

    print(
        "test_labels:",
        result[
            "test_labels"
        ].shape,
    )


def main_train() -> None:
    profile = load_profile()

    splits = make_splits(
        profile
    )

    # 기존 흐름 보존: 20초에서 RF / LightGBM 비교
    features_20 = load_feature_file(
        20
    )

    comparison = compare_models(
        features_20,
        profile,
        splits["train_ids"],
        splits["val_ids"],
        window_sec=20,
    )

    print(
        "\n=== 20초 RandomForest / LightGBM Validation 비교 ==="
    )

    print(
        comparison.to_string(
            index=False
        )
    )

    bundle, test_metrics = train_integrated_lgbm(
        profile=profile,
        splits=splits,
    )

    print(
        "\n최종 입력 시간:",
        bundle[
            "window_sec"
        ],
        "초",
    )

    print(
        "\n=== 최종 Test ==="
    )

    print(
        test_metrics.to_string(
            index=False
        )
    )

    print(
        "\n통합모델 저장 완료:"
    )

    print(
        MODEL_PATH
    )

    print(
        "\n생성 모델 파일 수: 1"
    )


def main_evaluate() -> None:
    bundle = load_model_bundle()

    results, matrices = evaluate_bundle(
        bundle
    )

    print(
        "=== 최종 Test 성능 ==="
    )

    print(
        results.to_string(
            index=False
        )
    )

    for target, matrix in matrices.items():
        print(
            f"\n=== {target} confusion matrix ==="
        )

        print(
            matrix.to_string()
        )


def main_predict() -> None:
    parser = argparse.ArgumentParser(
        description=(
            "HydroTwin 통합 LightGBM 예측"
        )
    )

    parser.add_argument(
        "--input-json",
        type=str,
        default=None,
        help=(
            "평균 특징 17개가 들어있는 JSON 파일"
        ),
    )

    parser.add_argument(
        "--cycle-id",
        type=int,
        default=None,
        help=(
            "processed 특징 파일의 cycle_id로 테스트"
        ),
    )

    args = parser.parse_args()

    bundle = load_model_bundle()

    seconds = int(
        bundle[
            "window_sec"
        ]
    )

    if args.input_json:
        path = Path(
            args.input_json
        )

        feature_row = json.loads(
            path.read_text(
                encoding="utf-8"
            )
        )

    else:
        features = load_feature_file(
            seconds
        )

        if args.cycle_id is None:
            # 인자가 없으면 첫 행으로 smoke test
            feature_row = features.iloc[
                [0]
            ]

        else:
            matched = features[
                features[
                    "cycle_id"
                ] == args.cycle_id
            ]

            if matched.empty:
                raise ValueError(
                    f"cycle_id {args.cycle_id}를 찾을 수 없습니다."
                )

            feature_row = matched.iloc[
                [0]
            ]

    print(
        predict_json(
            feature_row,
            model_bundle=bundle,
        )
    )


def main_explain() -> None:
    parser = argparse.ArgumentParser(
        description=(
            "HydroTwin 통합 LightGBM SHAP 설명"
        )
    )

    parser.add_argument(
        "--top-n",
        type=int,
        default=5,
    )

    parser.add_argument(
        "--max-rows",
        type=int,
        default=200,
    )

    args = parser.parse_args()

    bundle = load_model_bundle()

    profile = load_profile()

    splits = make_splits(
        profile
    )

    features = load_feature_file(
        int(
            bundle[
                "window_sec"
            ]
        )
    )

    test_id_set = set(
        splits[
            "test_ids"
        ]
    )

    X = (
        features[
            features[
                "cycle_id"
            ].isin(
                test_id_set
            )
        ]
        .sort_values(
            "cycle_id"
        )
        .head(
            args.max_rows
        )
    )

    shap_result = explain(
        X,
        model_bundle=bundle,
        top_n=args.top_n,
    )

    print(
        json.dumps(
            shap_result,
            ensure_ascii=False,
            indent=2,
        )
    )
