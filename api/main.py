"""
api/main.py
===========
Docker 통합 실행: Kafka → 10초 추론 → 파일 → 이 API → 웹/Unity.
HYDROTWIN_RUNTIME=1이면 드리프트·재학습·시나리오 상태도 함께 제공한다.
해당 환경변수가 없으면 기존 Raw Consumer 파일 읽기 방식과 호환된다.

엔드포인트:
    GET /health                 : 서버 상태 + 마지막 데이터 수신 후 경과 시간
    GET /api/v1/state/latest    : 최신 센서값 + 예측 상태 (통일된 API 계약)

사전 준비:
    python kafka/consumer_v5.py   (Raw Consumer, kafka/latest_raw.json 생성)

실행:
    uvicorn api.main:app --reload --port 8000
"""

import json
import os
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Dict, Optional

from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, ConfigDict
from fastapi.staticfiles import StaticFiles
from src.runtime.common import read_state
from src.monitoring.sensor_bands import load_sensor_bands

BASE_DIR = Path(__file__).resolve().parent.parent
LATEST_RAW_PATH = BASE_DIR / "kafka" / "latest_raw.json"
if os.getenv('HYDROTWIN_RUNTIME') == '1':
    LATEST_RAW_PATH = BASE_DIR / 'artifacts/runtime/latest.json'

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
    model_config = ConfigDict(extra='allow')
    status: str
    observed_window_sec: int
    components: Dict


class StateLatestResponse(BaseModel):
    model_config = ConfigDict(extra='allow')
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
    if os.getenv('HYDROTWIN_RUNTIME') == '1':
        retraining = read_state('retraining.json')
        request = read_state('retrain_request.json')
        # 이전 생성기 실행의 재학습 결과를 이번 실행의 상태처럼 표시하지 않는다.
        if request.get('run_id') != data.get('run_id'):
            retraining = {'status':'idle','message':'현재 실행의 재학습 요청 대기 · 이전 기록은 파일에 보존됨'}
        data['monitoring'] = {
            'drift':read_state('monitor.json'),
            'retraining':retraining,
            'scenario':read_state('scenario.json'),
        }
    return StateLatestResponse(**data)


@app.get('/api/v1/sensors/reference-bands', summary='센서별 고정 정상 운전 기준 범위')
def get_sensor_reference_bands():
    try:
        return load_sensor_bands()
    except (OSError, ValueError, KeyError) as error:
        raise HTTPException(status_code=503,detail='정상 운전 기준 데이터를 준비하지 못했습니다.') from error


class UnityStaticFiles(StaticFiles):
    """Unity 압축 빌드에 실제 MIME 유형과 gzip 인코딩을 지정한다."""
    async def get_response(self, path, scope):
        response = await super().get_response(path, scope)
        if path.endswith('.gz') and response.status_code==200:
            response.headers['Content-Encoding'] = 'gzip'
            response.headers['Content-Type'] = (
                'application/wasm' if path.endswith('.wasm.gz') else
                'application/javascript' if path.endswith('.js.gz') else
                'application/octet-stream'
            )
        return response


# 생성된 대용량 빌드는 저장소 밖 D드라이브에서 읽기 전용으로 제공한다.
unity_build_root = Path(os.getenv('UNITY_WEBGL_PATH', str(BASE_DIR/'web')))
if (unity_build_root/'Build'/'pro-build.loader.js').is_file():
    app.mount('/Build', UnityStaticFiles(directory=unity_build_root/'Build'), name='unity-build')
    if (unity_build_root/'StreamingAssets').is_dir():
        app.mount('/StreamingAssets', StaticFiles(directory=unity_build_root/'StreamingAssets'), name='unity-assets')

app.mount('/', UnityStaticFiles(directory=BASE_DIR/'web', html=True), name='web')
