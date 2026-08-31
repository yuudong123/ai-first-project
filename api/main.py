"""
api/main.py
===========
파일 1개 + 엔드포인트 2개 (health, data)

- GET /health : 서버 상태 + Kafka Consumer가 데이터를 잘 만들어내고 있는지 확인
- GET /data   : Kafka Consumer가 만들어 둔 최신 센서값 + AI 예측 결과 반환

사전 준비 (반드시 순서대로 실행되어 있어야 함):
    1) docker compose -f kafka/docker-compose.yml up -d
    2) python kafka/producer.py    (다른 터미널)
    3) python kafka/consumer.py    (또 다른 터미널)

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
LATEST_DATA_PATH = BASE_DIR / "kafka" / "latest_data.json"

START_TIME = time.time()
STALE_THRESHOLD_SEC = 30  # 이보다 오래된 데이터면 Consumer가 멈춘 것으로 간주


def read_latest_data() -> Optional[dict]:
    if not LATEST_DATA_PATH.exists():
        return None
    with open(LATEST_DATA_PATH, "r", encoding="utf-8") as f:
        return json.load(f)


class HealthResponse(BaseModel):
    status: str
    uptime_sec: float
    consumer_connected: bool
    last_data_age_sec: Optional[float]
    timestamp: str


class DataResponse(BaseModel):
    cycle_id: int
    elapsed_sec: int
    updated_at: str
    sensors: Dict[str, float]
    prediction: Dict


app = FastAPI(title="Hydraulic Monitoring API (Kafka)", version="3.0.0")

# 웹 대시보드(web/index.html)를 로컬 파일로 열거나 다른 포트에서 서빙해도
# fetch()로 이 API를 호출할 수 있도록 모든 출처를 허용한다 (로컬 개발용).
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)


@app.get("/health", response_model=HealthResponse, summary="서버 상태 확인")
def get_health():
    data = read_latest_data()
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


@app.get("/data", response_model=DataResponse, summary="실시간 센서값 + AI 예측 결과 (Kafka Consumer 결과)")
def get_data():
    data = read_latest_data()
    if data is None:
        raise HTTPException(
            status_code=503,
            detail="아직 Kafka Consumer로부터 데이터를 받지 못했습니다. producer/consumer가 실행 중인지 확인하세요.",
        )
    return DataResponse(**data)