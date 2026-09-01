"""
kafka/consumer_v5.py  (= Raw Consumer)
========================================
V5 Virtual Factory가 실제로 발행하는 Kafka 토픽(hydraulic.sensor.raw)을 구독해서
17개 센서의 최신값을 kafka/latest_raw.json에 저장한다.

이 단계(1차 테스트)의 목적: AI 예측 없이, Kafka -> Consumer -> 파일까지
원본 센서값 파이프라인이 1초 간격으로 끊기지 않고 도는지 확인하는 것.
그래서 prediction은 항상 status="warming_up"으로 채워서 내보낸다
(웹/Unity 쪽이 "예측 없음"을 명확히 구분해서 처리할 수 있도록).

메시지마다 증가하는 event_id를 부여한다. 웹/Unity는 이 event_id가 바뀌었을 때만
화면(특히 차트)을 갱신해서, 같은 데이터를 중복으로 그리지 않도록 한다.

환경변수로 브로커/토픽을 덮어쓸 수 있음:
    KAFKA_BROKER=192.168.133.108:9092 (기본값)
    KAFKA_TOPIC=hydraulic.sensor.raw   (기본값)

실행:
    python kafka/consumer_v5.py
"""

import json
import os
import time
from datetime import datetime, timezone
from pathlib import Path

from kafka import KafkaConsumer

BROKER = os.environ.get("KAFKA_BROKER", "192.168.133.108:9092")
TOPIC = os.environ.get("KAFKA_TOPIC", "hydraulic.sensor.raw")

OUTPUT_PATH = Path(__file__).resolve().parent / "latest_raw.json"

START_TIME = time.time()
_event_counter = {"n": 0}


def save_latest(data: dict):
    """임시 파일에 쓴 뒤 원자적으로 교체 (API가 읽는 도중 파일이 깨지는 것을 방지)"""
    tmp_path = OUTPUT_PATH.with_suffix(".tmp")
    with open(tmp_path, "w", encoding="utf-8") as f:
        json.dump(data, f, ensure_ascii=False, indent=2)
    tmp_path.replace(OUTPUT_PATH)


def handle_message(payload: dict) -> dict:
    """V5가 보낸 원본 메시지를 API 계약(/api/v1/state/latest) 형태로 변환해서 저장한다."""
    sensors = payload.get("sensors", {})
    generated_at = payload.get("timestamp")  # V5가 데이터를 만든 시각

    _event_counter["n"] += 1
    received_at = datetime.now(timezone.utc).isoformat()

    result = {
        "event_id": _event_counter["n"],
        "cycle_id": 1,  # 이 단계에서는 고정값 (60초 사이클 개념 아직 미적용)
        "elapsed_sec": round(time.time() - START_TIME),
        "generated_at": generated_at,   # V5가 메시지를 만든 시각
        "received_at": received_at,     # 우리 Consumer가 받은 시각
        "updated_at": received_at,      # 하위 호환용 (기존 필드명 유지)
        "sensors": sensors,
        "prediction": {
            "status": "warming_up",
            "observed_window_sec": 0,
            "components": {},
        },
    }
    save_latest(result)
    return result


def main():
    print(f"Raw Consumer 시작 (broker={BROKER}, topic={TOPIC})")
    consumer = KafkaConsumer(
        TOPIC,
        bootstrap_servers=BROKER,
        value_deserializer=lambda v: json.loads(v.decode("utf-8")),
        auto_offset_reset="latest",
        group_id="hydraulic-raw-consumer-group",
    )

    print("메시지 대기 중...")
    for message in consumer:
        payload = message.value
        result = handle_message(payload)
        print(f"[raw_consumer] event_id={result['event_id']} 수신 -> latest_raw.json 저장")


if __name__ == "__main__":
    main()
