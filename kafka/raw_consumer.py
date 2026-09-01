import json
import os
from pathlib import Path

from kafka import KafkaConsumer


# ============================================================
# HydroTwin Raw Sensor Consumer
# ============================================================
#
# 역할
# ------------------------------------------------------------
# Kafka의 hydraulic.sensor.raw Topic에서
# 가상 공장이 생성한 실시간 Raw Sensor 데이터를 받아
# 가장 최신 데이터만 latest_raw.json에 저장한다.
#
# 절대 하지 않는 것
# ------------------------------------------------------------
# - model.pkl 사용
# - AI Prediction
# - profile.txt 정답 사용
# - Feature 생성
# - 정상/고장 판단
# - Data Drift 판단
# ============================================================


KAFKA_BROKER = os.getenv(
    "KAFKA_BROKER",
    "localhost:9092",
)

KAFKA_TOPIC = os.getenv(
    "KAFKA_TOPIC",
    "hydraulic.sensor.raw",
)


KAFKA_DIR = Path(__file__).resolve().parent

LATEST_FILE = (
    KAFKA_DIR
    / "latest_raw.json"
)

TEMP_FILE = (
    KAFKA_DIR
    / "latest_raw.json.tmp"
)


consumer = KafkaConsumer(
    KAFKA_TOPIC,

    bootstrap_servers=KAFKA_BROKER,

    auto_offset_reset="latest",

    enable_auto_commit=True,

    group_id="hydraulic-raw-consumer",

    value_deserializer=lambda value: json.loads(
        value.decode("utf-8")
    ),
)


def save_latest(data):

    # --------------------------------------------------------
    # 먼저 임시 파일에 완전히 저장한 뒤
    # latest_raw.json으로 교체한다.
    #
    # 다른 프로그램이 읽는 순간
    # JSON이 반쯤 작성된 상태가 되는 것을 방지한다.
    # --------------------------------------------------------

    with open(
        TEMP_FILE,
        "w",
        encoding="utf-8",
    ) as f:

        json.dump(
            data,
            f,
            ensure_ascii=False,
            indent=2,
        )

    os.replace(
        TEMP_FILE,
        LATEST_FILE,
    )


def main():

    print("=" * 70)
    print("HydroTwin Raw Realtime Sensor Consumer")
    print("=" * 70)

    print(
        f"Kafka Broker : "
        f"{KAFKA_BROKER}"
    )

    print(
        f"Kafka Topic  : "
        f"{KAFKA_TOPIC}"
    )

    print(
        f"Output File  : "
        f"{LATEST_FILE}"
    )

    print("=" * 70)
    print("Waiting for realtime sensor data...")
    print("=" * 70)


    try:

        for record in consumer:

            # Kafka에서 들어온 데이터를
            # 어떠한 학습/가공 없이 그대로 사용
            data = record.value

            save_latest(
                data
            )

            sensors = data.get(
                "sensors",
                {},
            )

            print(
                f"[receive] "
                f"{data.get('timestamp')} "
                f"PS1={sensors.get('PS1')} "
                f"FS1={sensors.get('FS1')} "
                f"TS1={sensors.get('TS1')} "
                f"TS2={sensors.get('TS2')} "
                f"VS1={sensors.get('VS1')}"
            )


    except KeyboardInterrupt:

        print()
        print(
            "Raw Sensor Consumer stopped."
        )


    finally:

        consumer.close()


if __name__ == "__main__":
    main()
