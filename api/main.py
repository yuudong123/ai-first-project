"""
api/main.py
===========
1차 테스트 목표: Kafka -> Raw Consumer -> 이 API -> 웹/Unity 까지
1초 간격 데이터가 끊기지 않고 흐르는지 확인하는 것 (AI 예측은 아직 없음).

엔드포인트:
    GET /health                 : 서버 상태 + 마지막 데이터 수신 후 경과 시간
    GET /api/v1/state/latest    : 최신 센서값 + 예측 상태 (통일된 API 계약)

사전 준비:
    python kafka/consumer_v5.py   (Raw Consumer, kafka/latest_raw.json 생성)

실행:
    uvicorn api.main:app --reload --port 8000
"""

import json
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Dict, Optional

from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel

BASE_DIR = Path(__file__).resolve().parent.parent
LATEST_RAW_PATH = BASE_DIR / "kafka" / "latest_raw.json"

START_TIME = time.time()
STALE_THRESHOLD_SEC = 5  # 1초 간격 스트림이므로, 5초 이상 안 들어오면 "지연"으로 간주


def read_latest_raw() -> Optional[dict]:
    if not LATEST_RAW_PATH.exists():
        return None
    with open(LATEST_RAW_PATH, "r", encoding="utf-8") as f:
        return json.load(f)


class HealthResponse(BaseModel):
    status: str
    uptime_sec: float
    consumer_connected: bool
    last_data_age_sec: Optional[float]
    timestamp: str


class PredictionBlock(BaseModel):
    status: str
    observed_window_sec: int
    components: Dict


class StateLatestResponse(BaseModel):
    event_id: int
    cycle_id: int
    elapsed_sec: int
    updated_at: str
    generated_at: Optional[str] = None
    received_at: Optional[str] = None
    sensors: Dict[str, float]
    prediction: PredictionBlock


app = FastAPI(title="Hydraulic Monitoring API", version="4.0.0")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)


@app.get("/health", response_model=HealthResponse, summary="서버 상태 확인")
def get_health():
    data = read_latest_raw()
    age = None
    connected = False
    if data:
        updated_at = datetime.fromisoformat(data["updated_at"])
        age = (datetime.now(timezone.utc) - updated_at).total_seconds()
        connected = age < STALE_THRESHOLD_SEC

    return HealthResponse(
        status="ok",
        uptime_sec=round(time.time() - START_TIME, 1),
        consumer_connected=connected,
        last_data_age_sec=round(age, 1) if age is not None else None,
        timestamp=datetime.now(timezone.utc).isoformat(),
    )


@app.get(
    "/api/v1/state/latest",
    response_model=StateLatestResponse,
    summary="최신 센서값 + 예측 상태 (웹/Unity 공통 계약)",
)
def get_state_latest():
    data = read_latest_raw()
    if data is None:
        raise HTTPException(
            status_code=503,
            detail="아직 Raw Consumer로부터 데이터를 받지 못했습니다. kafka/consumer_v5.py가 실행 중인지 확인하세요.",
        )
    return StateLatestResponse(**data)