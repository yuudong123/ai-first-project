"""
api/main.py
===========
역할: AI 추론 서버가 아니라, 이미 완성된 State 데이터를 웹/Unity로 전달하는 Adapter.

이 파일이 하는 일은 딱 두 가지뿐이다.
    1) 요청이 올 때마다 State 파일을 새로 읽는다 (캐시하지 않음)
    2) 그 내용을 가공 없이 그대로 반환한다

응답 스키마를 강제하지 않는 이유:
    설비별 데이터(station-01/02/03) 형식이든 단일 형식이든 그대로 통과시켜야 하며,
    Pydantic으로 필드를 고정하면 형식이 조금만 달라져도 500 에러가 나기 때문이다.

엔드포인트:
    GET /health                 : 서버 상태 + State 파일 신선도
    GET /api/v1/state/latest    : State 파일 내용 그대로 반환

환경변수로 State 파일 경로를 덮어쓸 수 있음:
    STATE_FILE=/경로/latest_state_by_equipment.json

실행:
    uvicorn api.main:app --reload --port 8000
"""

import json
import os
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Optional

from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel

BASE_DIR = Path(__file__).resolve().parent.parent

# 우선순위대로 State 파일을 찾는다.
#   1) 환경변수 STATE_FILE 로 직접 지정한 경로
#   2) AI 파이프라인이 만드는 파일
#   3) 우리 Consumer(kafka/consumer_v5.py)가 만드는 파일
_ENV_STATE = os.environ.get("STATE_FILE")
STATE_CANDIDATES = (
    [Path(_ENV_STATE)] if _ENV_STATE else [
        BASE_DIR / "kafka" / "latest_state_by_equipment.json",
        BASE_DIR / "kafka" / "latest_raw.json",
    ]
)


def resolve_state_path() -> Path:
    """실제로 존재하는 첫 번째 후보를 고른다. 없으면 첫 후보를 그대로 돌려준다."""
    for path in STATE_CANDIDATES:
        if path.exists():
            return path
    return STATE_CANDIDATES[0]


START_TIME = time.time()
STALE_THRESHOLD_SEC = 5  # 1초 간격 스트림이므로, 5초 이상 갱신이 없으면 "지연"으로 간주


def read_state() -> Optional[Any]:
    """State 파일을 매 호출마다 새로 읽는다 (항상 최신값을 주기 위해 캐시하지 않음).

    쓰는 쪽에서 파일을 교체하는 순간에 읽으면 내용이 비었거나 깨져 보일 수 있으므로,
    그런 경우는 None으로 처리해서 호출자가 503을 주도록 한다.
    """
    path = resolve_state_path()
    if not path.exists():
        return None
    try:
        with open(path, "r", encoding="utf-8") as f:
            text = f.read().strip()
        if not text:
            return None
        return json.loads(text)
    except (json.JSONDecodeError, OSError):
        return None


def collect_updated_times(node: Any, found: list):
    """중첩된 구조 어디에 있든 prediction.updated_at 값을 모두 찾아 모은다.

    State 파일이 설비별 키 객체든, 배열이든, 단일 객체든 상관없이
    신선도를 판정할 수 있도록 재귀적으로 훑는다.
    """
    if isinstance(node, dict):
        pred = node.get("prediction")
        if isinstance(pred, dict) and isinstance(pred.get("updated_at"), str):
            found.append(pred["updated_at"])
        for value in node.values():
            collect_updated_times(value, found)
    elif isinstance(node, list):
        for item in node:
            collect_updated_times(item, found)


def newest_age_sec(state: Any) -> Optional[float]:
    """State 안에서 가장 최근 갱신 시각을 찾아, 지금으로부터 몇 초 지났는지 반환."""
    stamps: list = []
    collect_updated_times(state, stamps)
    if not stamps:
        return None

    now = datetime.now(timezone.utc)
    ages = []
    for s in stamps:
        try:
            dt = datetime.fromisoformat(s.replace("Z", "+00:00"))
            if dt.tzinfo is None:
                dt = dt.replace(tzinfo=timezone.utc)
            ages.append((now - dt).total_seconds())
        except ValueError:
            continue
    return min(ages) if ages else None


class HealthResponse(BaseModel):
    status: str
    uptime_sec: float
    state_file: str
    state_file_exists: bool
    consumer_connected: bool
    last_data_age_sec: Optional[float]
    timestamp: str


app = FastAPI(title="Hydraulic State Adapter API", version="5.0.0")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)


@app.get("/health", response_model=HealthResponse, summary="서버 상태 + State 파일 신선도")
def get_health():
    state = read_state()
    age = newest_age_sec(state) if state is not None else None

    # updated_at을 못 찾으면 파일 수정 시각으로 대체 판정
    path = resolve_state_path()
    if age is None and path.exists():
        age = time.time() - path.stat().st_mtime

    return HealthResponse(
        status="ok",
        uptime_sec=round(time.time() - START_TIME, 1),
        state_file=str(path),
        state_file_exists=path.exists(),
        consumer_connected=(age is not None and age < STALE_THRESHOLD_SEC),
        last_data_age_sec=round(age, 1) if age is not None else None,
        timestamp=datetime.now(timezone.utc).isoformat(),
    )


@app.get("/api/v1/state/latest", summary="설비별 최신 State (파일 내용 그대로 전달)")
def get_state_latest():
    state = read_state()
    if state is None:
        raise HTTPException(
            status_code=503,
            detail=(
                f"State 파일을 읽을 수 없습니다: "
                f"{', '.join(str(p) for p in STATE_CANDIDATES)} "
                "(파일이 아직 생성되지 않았거나 비어있음). "
                "Consumer가 실행 중인지 확인하세요."
            ),
        )
    return state