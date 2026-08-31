"""
kafka/consumer.py
==================
Kafka 토픽(hydraulic.sensor.raw)을 구독해서 센서 데이터를 받고,
학습된 모델(models/model.pkl)로 예측을 수행한 뒤 결과를 kafka/latest_data.json에 저장한다.
FastAPI(api/main.py)는 이 파일을 읽어서 /data 응답을 만든다.

사전 준비 (반드시 순서대로):
    1) docker compose -f kafka/docker-compose.yml up -d
    2) python kafka/producer.py   (다른 터미널에서 먼저 실행해서 메시지를 흘려보내야 함)

실행:
    python kafka/consumer.py
"""

import json
import time
from datetime import datetime, timezone
from pathlib import Path

import joblib
import pandas as pd
from kafka import KafkaConsumer

BASE_DIR = Path(__file__).resolve().parent.parent
MODEL_PATH = BASE_DIR / "models" / "model.pkl"
OUTPUT_PATH = Path(__file__).resolve().parent / "latest_data.json"

KAFKA_BROKER = "localhost:9092"
TOPIC = "hydraulic.sensor.raw"


def load_model_bundle():
    print("모델 로드 중...")
    return joblib.load(MODEL_PATH)


def run_prediction(bundle, features: dict) -> dict:
    models = bundle["models"]
    feature_names = bundle["feature_names"]
    label_maps = bundle["label_maps"]
    risk_maps = bundle["risk_maps"]

    X = pd.DataFrame([features])[feature_names]

    components = {}
    for component_name, clf in models.items():
        pred_class = clf.predict(X)[0]
        proba = clf.predict_proba(X)[0]
        confidence = float(max(proba))
        state_label = label_maps[component_name][pred_class]
        risk_level = risk_maps[component_name][state_label]
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


def save_latest(data: dict):
    """임시 파일에 쓴 뒤 원자적으로 교체 (API가 읽는 도중 파일이 깨지는 것을 방지)"""
    tmp_path = OUTPUT_PATH.with_suffix(".tmp")
    with open(tmp_path, "w", encoding="utf-8") as f:
        json.dump(data, f, ensure_ascii=False, indent=2)
    tmp_path.replace(OUTPUT_PATH)


def main():
    bundle = load_model_bundle()

    consumer = KafkaConsumer(
        TOPIC,
        bootstrap_servers=KAFKA_BROKER,
        value_deserializer=lambda v: json.loads(v.decode("utf-8")),
        auto_offset_reset="latest",
        group_id="hydraulic-consumer-group",
    )

    print(f"Kafka Consumer 시작 (topic={TOPIC}), 메시지 대기 중...")
    for message in consumer:
        payload = message.value
        cycle_id = payload["cycle_id"]
        sensors = payload["sensors"]
        features = payload["features"]

        prediction = run_prediction(bundle, features)

        result = {
            "cycle_id": cycle_id,
            "elapsed_sec": (cycle_id - 1) * 20,
            "updated_at": datetime.now(timezone.utc).isoformat(),
            "sensors": sensors,
            "prediction": prediction,
        }
        save_latest(result)
        print(f"[consumer] cycle_id={cycle_id} 예측 완료 -> {OUTPUT_PATH.name} 저장")


if __name__ == "__main__":
    main()
