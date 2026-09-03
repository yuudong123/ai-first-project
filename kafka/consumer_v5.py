"""
kafka/consumer_v5.py  (= Raw Consumer)
========================================
Kafka 토픽(hydraulic.sensor.multi.raw)을 구독해서, 설비별 최신 상태를
kafka/latest_raw.json에 저장한다. FastAPI가 이 파일을 읽어 웹/Unity로 전달한다.

중요: 메시지는 설비 한 대씩 따로 온다.
    {"equipment_id": "station-01", "timestamp": ..., "sensors": {...}}
    {"equipment_id": "station-02", ...}
    {"equipment_id": "station-03", ...}
    그래서 받을 때마다 덮어쓰면 마지막 한 대만 남는다.
    이 Consumer는 설비별로 최신값을 누적 보관해서, 파일에는 항상 3대가 모두 들어있다.

저장되는 형태:
    {
      "station-01": { "equipment_id": ..., "sensors": {...}, "prediction": {...} },
      "station-02": { ... },
      "station-03": { ... },
      "_meta": { "event_id": 12, "received_at": "...", "last_updated_equipment": "station-02" }
    }

한 메시지에 설비 여러 대가 묶여 오는 형식이나, 설비 구분이 없는 예전 형식도
그대로 처리한다 (하위 호환).

환경변수로 브로커/토픽을 덮어쓸 수 있음:
    KAFKA_BROKER=192.168.133.108:9092        (기본값)
    KAFKA_TOPIC=hydraulic.sensor.multi.raw   (기본값)

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
TOPIC = os.environ.get("KAFKA_TOPIC", "hydraulic.sensor.multi.raw")

OUTPUT_PATH = Path(__file__).resolve().parent / "latest_raw.json"

START_TIME = time.time()
_event_counter = {"n": 0}


def save_latest(data: dict, max_retries: int = 8, retry_delay: float = 0.05):
    """임시 파일에 쓴 뒤 원자적으로 교체 (API가 읽는 도중 파일이 깨지는 것을 방지).

    Windows는 다른 프로세스(API 서버)가 그 순간 파일을 읽고 있으면
    교체(replace)를 PermissionError로 거부하는 경우가 있다 (Linux/Mac은 허용).
    아주 짧게(초 단위 이하) 재시도하면 대부분 다음 시도에서 통과한다.
    """
    tmp_path = OUTPUT_PATH.with_suffix(".tmp")
    with open(tmp_path, "w", encoding="utf-8") as f:
        json.dump(data, f, ensure_ascii=False, indent=2)

    for attempt in range(max_retries):
        try:
            tmp_path.replace(OUTPUT_PATH)
            return
        except PermissionError:
            if attempt == max_retries - 1:
                raise
            time.sleep(retry_delay)


def looks_like_equipment_map(payload: dict) -> bool:
    """한 메시지에 설비 여러 대가 담긴 형태인지 판별한다.

    예: {"station-01": {"sensors": {...}}, "station-02": {...}}
    """
    if not isinstance(payload, dict):
        return False
    for key, value in payload.items():
        if isinstance(value, dict) and ("sensors" in value or "prediction" in value):
            return True
    return False


# 설비별 최신 상태를 여기에 누적해 둔다.
# 메시지는 설비 한 대씩 번갈아 오기 때문에, 매번 덮어쓰면 마지막 한 대만 남는다.
_states: dict = {}


def handle_message(payload: dict) -> dict:
    """받은 메시지를 API가 읽을 파일 형태로 바꿔 저장한다.

    지원하는 입력 형태:
      1) {"equipment_id": "station-01", "timestamp": ..., "sensors": {...}}
         -> 설비 한 대씩 오는 형태. 설비별로 누적해서 3대를 모두 유지한다.
      2) {"station-01": {...}, "station-02": {...}}
         -> 이미 설비별로 묶여 온 형태. 그대로 저장한다.
      3) {"timestamp": ..., "sensors": {...}}
         -> 설비 구분이 없는 예전 형태. 하위 호환으로 감싸서 저장한다.
    """
    _event_counter["n"] += 1
    received_at = datetime.now(timezone.utc).isoformat()

    # 2) 이미 설비별로 묶여 온 경우: 원본 구조를 그대로 보존
    if looks_like_equipment_map(payload):
        result = dict(payload)
        result["_meta"] = {
            "event_id": _event_counter["n"],
            "received_at": received_at,
            "elapsed_sec": round(time.time() - START_TIME),
        }
        save_latest(result)
        return result

    # 1) 설비 한 대씩 오는 경우: 해당 설비만 갱신하고 나머지는 유지
    equipment_id = payload.get("equipment_id")
    if equipment_id:
        entry = {
            "equipment_id": equipment_id,
            "timestamp": payload.get("timestamp"),
            "sensors": payload.get("sensors", {}),
            "received_at": received_at,
        }
        # 발행 측이 예측 결과를 같이 보내면 그대로 쓰고, 없으면 대기 상태로 표시
        entry["prediction"] = payload.get("prediction") or {
            "status": "warming_up",
            "result": None,
            "window_sec": None,
            "updated_at": received_at,
        }
        _states[equipment_id] = entry

        result = dict(_states)
        result["_meta"] = {
            "event_id": _event_counter["n"],
            "received_at": received_at,
            "elapsed_sec": round(time.time() - START_TIME),
            "last_updated_equipment": equipment_id,
        }
        save_latest(result)
        return result

    # 3) 설비 구분이 없는 예전 형식
    result = {
        "event_id": _event_counter["n"],
        "cycle_id": 1,
        "elapsed_sec": round(time.time() - START_TIME),
        "generated_at": payload.get("timestamp"),
        "received_at": received_at,
        "updated_at": received_at,
        "sensors": payload.get("sensors", {}),
        "prediction": {
            "status": "warming_up",
            "observed_window_sec": 0,
            "components": {},
        },
    }
    save_latest(result)
    return result


def main():
    # 실행할 때마다 새로운 group_id를 써서, 카프카가 기억해둔 예전 커밋 위치를
    # 무시하고 항상 "지금 이 순간부터" 들어오는 메시지만 받는다.
    # (group_id를 고정해두면 재시작할 때 밀린 메시지부터 이어받게 됨)
    run_group_id = f"hydraulic-raw-consumer-{time.time_ns()}"

    print(f"Raw Consumer 시작 (broker={BROKER}, topic={TOPIC})")
    consumer = KafkaConsumer(
        TOPIC,
        bootstrap_servers=BROKER,
        value_deserializer=lambda v: json.loads(v.decode("utf-8")),
        auto_offset_reset="latest",
        group_id=run_group_id,
        enable_auto_commit=False,  # 이 그룹은 매번 새로 버려지므로 커밋 자체가 의미 없음
    )

    print("메시지 대기 중...")
    for message in consumer:
        payload = message.value
        try:
            result = handle_message(payload)
            meta = result.get("_meta")
            if meta:
                stations = sorted(k for k in result if k != "_meta")
                updated = meta.get("last_updated_equipment", "-")
                print(f"[raw_consumer] event_id={meta['event_id']} "
                      f"{updated} 갱신 | 보관 중 {len(stations)}대: {', '.join(stations)}")
            else:
                print(f"[raw_consumer] event_id={result['event_id']} 수신 -> {OUTPUT_PATH.name} 저장")
        except Exception as e:
            # 파일 저장 한 번 실패했다고 전체 스트림이 죽으면 안 됨 -> 로그만 남기고 계속 진행
            print(f"[raw_consumer] 저장 실패 (다음 메시지로 계속 진행): {e}")


if __name__ == "__main__":
    main()