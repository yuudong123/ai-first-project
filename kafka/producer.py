"""
kafka/producer.py
==================
센서 원본 데이터를 사이클 단위로 읽어서, 실시간 센서 스트림을 흉내 내며
Kafka 토픽(hydraulic.sensor.raw)에 발행(publish)한다.

사전 준비:
    docker compose -f kafka/docker-compose.yml up -d   (로컬 Kafka 브로커 실행)

실행:
    python kafka/producer.py
"""

import json
import time
from pathlib import Path

import pandas as pd
from kafka import KafkaProducer

BASE_DIR = Path(__file__).resolve().parent.parent
DATA_DIR = BASE_DIR / "data" / "raw" / "uci_hydraulic" / "extracted"

KAFKA_BROKER = "localhost:9092"
TOPIC = "hydraulic.sensor.raw"

SEND_INTERVAL_SEC = 5  # 실제 사이클은 60초지만, 테스트용으로 5초 간격으로 전송

SENSOR_FILES = [
    "PS1", "PS2", "PS3", "PS4", "PS5", "PS6",
    "EPS1", "FS1", "FS2",
    "TS1", "TS2", "TS3", "TS4", "VS1", "CE", "CP", "SE",
]


def load_raw_sensor_data():
    return {name: pd.read_csv(DATA_DIR / f"{name}.txt", sep="\t", header=None) for name in SENSOR_FILES}


def build_message(raw_data, cycle_idx: int) -> dict:
    """한 사이클의 센서 원본에서, 화면 표시용 대표값(mean)과
    모델 입력용 피처(mean/std/min/max)를 함께 만든다."""
    sensors = {}
    features = {}
    for name in SENSOR_FILES:
        row = raw_data[name].iloc[cycle_idx]
        sensors[name] = round(float(row.mean()), 3)
        features[f"{name}_mean"] = float(row.mean())
        features[f"{name}_std"] = float(row.std())
        features[f"{name}_min"] = float(row.min())
        features[f"{name}_max"] = float(row.max())

    return {
        "cycle_id": cycle_idx + 1,
        "produced_at": time.time(),
        "sensors": sensors,
        "features": features,
    }


def main():
    print("센서 원본 데이터 로드 중...")
    raw_data = load_raw_sensor_data()
    total_cycles = len(raw_data["PS1"])
    print(f"총 {total_cycles}개 사이클 로드 완료")

    producer = KafkaProducer(
        bootstrap_servers=KAFKA_BROKER,
        value_serializer=lambda v: json.dumps(v).encode("utf-8"),
    )

    print(f"Kafka Producer 시작 (topic={TOPIC}, interval={SEND_INTERVAL_SEC}s)")
    idx = 0
    try:
        while True:
            cycle_idx = idx % total_cycles
            message = build_message(raw_data, cycle_idx)
            producer.send(TOPIC, value=message)
            producer.flush()
            print(f"[producer] cycle_id={message['cycle_id']} 전송 완료")
            idx += 1
            time.sleep(SEND_INTERVAL_SEC)
    except KeyboardInterrupt:
        print("Producer 종료")
    finally:
        producer.close()


if __name__ == "__main__":
    main()
