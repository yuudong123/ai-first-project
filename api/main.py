"""
api/main.py
===========
파일 1개 + 엔드포인트 2개 (health, data)

- GET /health : 서버 상태
- GET /data   : 실시간 센서값 + 그 센서값을 학습된 모델(AI)에 넣어서 나온 결과값

실행 (프로젝트 루트에서):
    uvicorn api.main:app --reload --port 8000
"""

import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Dict

import joblib
import pandas as pd
from fastapi import FastAPI
from pydantic import BaseModel

# --------------------------------------------------------------------------
# 0. 경로 설정
# --------------------------------------------------------------------------

BASE_DIR = Path(__file__).resolve().parent.parent  # ai-first-project/
DATA_DIR = BASE_DIR / "data" / "raw" / "uci_hydraulic" / "extracted"
MODEL_PATH = BASE_DIR / "models" / "model.pkl"

SENSOR_FILES = [
    "PS1", "PS2", "PS3", "PS4", "PS5", "PS6",
    "EPS1", "FS1", "FS2",
    "TS1", "TS2", "TS3", "TS4", "VS1", "CE", "CP", "SE",
]

START_TIME = time.time()

# --------------------------------------------------------------------------
# 1. 서버 시작 시: 학습된 모델 로드 + 센서 원본 데이터 로드
# --------------------------------------------------------------------------

print("모델 로드 중...")
BUNDLE = joblib.load(MODEL_PATH)
MODELS = BUNDLE["models"]              # {"cooler": clf, "valve": clf, ...}
FEATURE_NAMES = BUNDLE["feature_names"]
LABEL_MAPS = BUNDLE["label_maps"]
RISK_MAPS = BUNDLE["risk_maps"]


def load_raw_sensor_data() -> Dict[str, pd.DataFrame]:
    """센서별 원본(사이클 x 샘플) 행렬을 그대로 로드 (실시간 스트림을 흉내내기 위한 원본 보관)"""
    return {name: pd.read_csv(DATA_DIR / f"{name}.txt", sep="\t", header=None) for name in SENSOR_FILES}


RAW_DATA = load_raw_sensor_data()
TOTAL_CYCLES = len(RAW_DATA["PS1"])


def build_cycle_features(cycle_idx: int) -> pd.DataFrame:
    """특정 사이클(cycle_idx)의 센서 원본 한 행에서 mean/std/min/max 피처를 만든다.
    (train.py에서 만든 피처와 동일한 방식 -> 순서(FEATURE_NAMES)도 반드시 맞춰야 함)"""
    row_features = {}
    for name in SENSOR_FILES:
        row = RAW_DATA[name].iloc[cycle_idx]
        row_features[f"{name}_mean"] = row.mean()
        row_features[f"{name}_std"] = row.std()
        row_features[f"{name}_min"] = row.min()
        row_features[f"{name}_max"] = row.max()
    df = pd.DataFrame([row_features])
    return df[FEATURE_NAMES]  # 학습 때 컬럼 순서와 동일하게 정렬


def build_representative_sensors(cycle_idx: int) -> Dict[str, float]:
    """/data 응답에 보여줄 '현재 센서값' (각 센서 원본 행의 평균값)"""
    return {name: round(float(RAW_DATA[name].iloc[cycle_idx].mean()), 3) for name in SENSOR_FILES}


# --------------------------------------------------------------------------
# 2. 센서값 -> AI(학습된 모델) 실행 -> 결과값
# --------------------------------------------------------------------------


def run_ai_prediction(cycle_idx: int) -> Dict:
    """이 함수가 '실시간 센서를 AI로 구동해서 나온 결과값'을 만드는 부분."""
    X = build_cycle_features(cycle_idx)

    components = {}
    for component_name, clf in MODELS.items():
        pred_class = clf.predict(X)[0]                      # 원본 라벨값 (예: 3, 20, 100 ...)
        proba = clf.predict_proba(X)[0]
        confidence = float(max(proba))

        state_label = LABEL_MAPS[component_name][pred_class]  # 사람이 읽는 상태명으로 변환
        risk_level = RISK_MAPS[component_name][state_label]

        components[component_name] = {
            "raw_value": int(pred_class),
            "state_label": state_label,
            "risk_level": risk_level,
            "confidence": round(confidence, 3),
        }

    return {
        "status": "ready",
        "observed_window_sec": 20,
        "components": components,
    }


# --------------------------------------------------------------------------
# 3. 응답 스키마
# --------------------------------------------------------------------------


class HealthResponse(BaseModel):
    status: str
    uptime_sec: float
    total_cycles_loaded: int
    model_loaded: bool
    timestamp: str


class DataResponse(BaseModel):
    cycle_id: int
    elapsed_sec: int
    updated_at: str
    sensors: Dict[str, float]
    prediction: Dict


# --------------------------------------------------------------------------
# 4. FastAPI 앱 + 엔드포인트 2개
# --------------------------------------------------------------------------

app = FastAPI(title="Hydraulic Monitoring API", version="2.0.0")

_cursor = {"i": 0}  # /data 호출마다 다음 사이클로 넘어가는 커서 (실시간 스트림 흉내)


@app.get("/health", response_model=HealthResponse, summary="서버 상태 확인")
def get_health():
    return HealthResponse(
        status="ok",
        uptime_sec=round(time.time() - START_TIME, 1),
        total_cycles_loaded=TOTAL_CYCLES,
        model_loaded=MODELS is not None,
        timestamp=datetime.now(timezone.utc).isoformat(),
    )


@app.get("/data", response_model=DataResponse, summary="실시간 센서값 + AI 예측 결과")
def get_data():
    idx = _cursor["i"] % TOTAL_CYCLES
    _cursor["i"] += 1

    sensors = build_representative_sensors(idx)   # [1] 실시간 센서값
    prediction = run_ai_prediction(idx)            # [2]+[3] 센서값 -> AI 실행 -> 결과값

    return DataResponse(
        cycle_id=idx + 1,
        elapsed_sec=idx * 20,
        updated_at=datetime.now(timezone.utc).isoformat(),
        sensors=sensors,
        prediction=prediction,
    )