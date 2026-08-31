import json
import os
import time
from datetime import datetime
from pathlib import Path

import numpy as np
from kafka import KafkaProducer


# ============================================================
# HydroTwin Virtual Factory Sensor Generator
# ============================================================
#
# 목적
# ------------------------------------------------------------
# UCI Hydraulic TXT 데이터를 이용하여
# 실제 공장에서 센서 데이터가 실시간으로 발생하는 것처럼
# 17개 센서값을 1초마다 Kafka로 전송한다.
#
# Kafka Raw 데이터에 포함:
#   - timestamp
#   - sensors (17개)
#
# 포함하지 않음:
#   - cycle_id
#   - machine_id
#   - profile.txt 정답
#   - AI Prediction
#   - Feature(mean/std/min/max)
#   - confidence
#   - risk_level
#   - drift=true 같은 정답
#
# ============================================================


# ============================================================
# 경로 설정
# ============================================================

PROJECT_ROOT = Path(__file__).resolve().parents[1]

DATA_DIR = (
    PROJECT_ROOT
    / "data"
    / "raw"
    / "uci_hydraulic"
    / "extracted"
)


# ============================================================
# Kafka 설정
# ============================================================

KAFKA_BROKER = os.getenv(
    "KAFKA_BROKER",
    "localhost:9092",
)

KAFKA_TOPIC = os.getenv(
    "KAFKA_TOPIC",
    "hydraulic.sensor.raw",
)

SEND_INTERVAL_SEC = float(
    os.getenv(
        "SEND_INTERVAL_SEC",
        "1.0",
    )
)


# ============================================================
# 가상 공장 Scenario
# ============================================================
#
# normal
# ------------------------------------------------------------
# UCI 원본 센서 흐름을 그대로 실시간 재생
#
#
# summer_drift
# ------------------------------------------------------------
#
# 정상 운전
#      ↓
# 외부 기온 상승
#      ↓
# TS1~TS4 점진적 상승
#      ↓
# 높은 온도 환경 일정 시간 유지
#      ↓
# 외부 환경 정상화
#      ↓
# TS1~TS4 점진적 복귀
#      ↓
# 다시 정상 운전
#
# Drift 여부는 Kafka에 절대 전송하지 않는다.
# 실제 Sensor 값만 변화시킨다.
#
# ============================================================

SENSOR_SCENARIO = os.getenv(
    "SENSOR_SCENARIO",
    "normal",
).lower()


# 정상 운전 후 Drift가 시작되는 시간
DRIFT_START_SEC = int(
    os.getenv(
        "DRIFT_START_SEC",
        "300",
    )
)


# 온도가 서서히 상승하는 시간
DRIFT_RISE_SEC = int(
    os.getenv(
        "DRIFT_RISE_SEC",
        "300",
    )
)


# 상승한 환경을 유지하는 시간
DRIFT_HOLD_SEC = int(
    os.getenv(
        "DRIFT_HOLD_SEC",
        "600",
    )
)


# 다시 원래 환경으로 복귀하는 시간
DRIFT_RECOVERY_SEC = int(
    os.getenv(
        "DRIFT_RECOVERY_SEC",
        "300",
    )
)


# 최대 온도 상승량
DRIFT_MAX_TEMP_OFFSET = float(
    os.getenv(
        "DRIFT_MAX_TEMP_OFFSET",
        "8.0",
    )
)


# ============================================================
# UCI 17개 Sensor Sampling Rate
# ============================================================
#
# Pressure
# PS1 ~ PS6 : 100 Hz
#
# Electrical Power
# EPS1      : 100 Hz
#
# Flow
# FS1 ~ FS2 : 10 Hz
#
# Temperature
# TS1 ~ TS4 : 1 Hz
#
# Vibration
# VS1       : 1 Hz
#
# 기타
# CE / CP / SE : 1 Hz
#
# 총 17개 Sensor
#
# ============================================================

SENSOR_HZ = {
    "PS1": 100,
    "PS2": 100,
    "PS3": 100,
    "PS4": 100,
    "PS5": 100,
    "PS6": 100,

    "EPS1": 100,

    "FS1": 10,
    "FS2": 10,

    "TS1": 1,
    "TS2": 1,
    "TS3": 1,
    "TS4": 1,

    "VS1": 1,

    "CE": 1,
    "CP": 1,
    "SE": 1,
}


# ============================================================
# Kafka Producer 생성
# ============================================================

producer = KafkaProducer(
    bootstrap_servers=KAFKA_BROKER,

    value_serializer=lambda value: json.dumps(
        value,
        ensure_ascii=False,
    ).encode("utf-8"),

    acks="all",
    retries=5,
)


# ============================================================
# UCI Realtime Reader
# ============================================================
#
# UCI TXT 한 줄은 60초 동안의 Sensor 기록
#
# 예)
#
# PS1
# 100 Hz × 60초
# = 한 줄에 6000개 측정값
#
# FS1
# 10 Hz × 60초
# = 한 줄에 600개 측정값
#
# TS1
# 1 Hz × 60초
# = 한 줄에 60개 측정값
#
#
# 서로 다른 Sampling Rate를
# 1초 단위 데이터로 맞춘다.
#
# 주의:
# 여기서 평균을 사용하는 이유는
# AI Feature를 만드는 것이 아니라
# 100Hz/10Hz Sensor를 프로젝트의
# 1초 실시간 Stream으로 맞추기 위한 Resampling이다.
#
# ============================================================


class UCIRealtimeReader:

    def __init__(self):

        self.handles = {}

        for sensor in SENSOR_HZ:

            sensor_path = (
                DATA_DIR / f"{sensor}.txt"
            )

            if not sensor_path.exists():

                raise FileNotFoundError(
                    f"Sensor file not found: "
                    f"{sensor_path}"
                )

            self.handles[sensor] = open(
                sensor_path,
                "r",
                encoding="utf-8",
            )


    def rewind(self):

        for handle in self.handles.values():
            handle.seek(0)


    def close(self):

        for handle in self.handles.values():
            handle.close()


    def read_next_block(self):

        lines = {}

        # ----------------------------------------------------
        # 17개 Sensor에서 동일한 60초 구간 읽기
        # ----------------------------------------------------

        for sensor, handle in self.handles.items():

            line = handle.readline()

            # 2205개 기록을 모두 사용하면
            # 처음부터 다시 읽는다.
            if not line:

                print(
                    "[INFO] UCI 원본 데이터 마지막 도달 "
                    "→ 처음부터 계속 운전"
                )

                self.rewind()

                return self.read_next_block()

            lines[sensor] = line


        block = {}

        # ----------------------------------------------------
        # Sensor별 1초 단위 Resampling
        # ----------------------------------------------------

        for sensor, line in lines.items():

            hz = SENSOR_HZ[sensor]

            values = np.fromstring(
                line,
                sep=" ",
                dtype=float,
            )

            expected_count = 60 * hz

            if values.size < expected_count:

                raise ValueError(
                    f"{sensor}: "
                    f"expected={expected_count}, "
                    f"actual={values.size}"
                )

            values = values[:expected_count]


            # ------------------------------------------------
            # 100Hz
            # 100개 → 1초 대표 Sensor 값
            #
            # 10Hz
            # 10개 → 1초 대표 Sensor 값
            #
            # 1Hz
            # 원래 값 그대로
            # ------------------------------------------------

            one_second_values = (
                values
                .reshape(60, hz)
                .mean(axis=1)
            )

            block[sensor] = (
                one_second_values
            )


        return block


# ============================================================
# Environment Drift Offset 계산
# ============================================================
#
# summer_drift의 시간 흐름:
#
# [NORMAL]
# 0 ---------------- DRIFT_START_SEC
#
# [RISE]
# 서서히 온도 상승
#
# [HOLD]
# 높은 온도 환경 유지
#
# [RECOVERY]
# 서서히 기존 환경으로 복귀
#
# [NORMAL]
# 다시 원본 Sensor 환경
#
# ============================================================


def calculate_temperature_offset(runtime_sec):

    if SENSOR_SCENARIO == "normal":
        return 0.0


    if SENSOR_SCENARIO != "summer_drift":

        raise ValueError(
            f"Unknown SENSOR_SCENARIO: "
            f"{SENSOR_SCENARIO}"
        )


    # --------------------------------------------------------
    # 구간 계산
    # --------------------------------------------------------

    rise_start = DRIFT_START_SEC

    rise_end = (
        rise_start
        + DRIFT_RISE_SEC
    )

    hold_end = (
        rise_end
        + DRIFT_HOLD_SEC
    )

    recovery_end = (
        hold_end
        + DRIFT_RECOVERY_SEC
    )


    # --------------------------------------------------------
    # 1. 정상 구간
    # --------------------------------------------------------

    if runtime_sec < rise_start:
        return 0.0


    # --------------------------------------------------------
    # 2. 온도 상승 구간
    # --------------------------------------------------------

    if runtime_sec < rise_end:

        progress = (
            runtime_sec - rise_start
        ) / max(
            DRIFT_RISE_SEC,
            1,
        )

        return (
            DRIFT_MAX_TEMP_OFFSET
            * progress
        )


    # --------------------------------------------------------
    # 3. 높은 온도 환경 유지
    # --------------------------------------------------------

    if runtime_sec < hold_end:

        return (
            DRIFT_MAX_TEMP_OFFSET
        )


    # --------------------------------------------------------
    # 4. 환경 복귀 구간
    # --------------------------------------------------------

    if runtime_sec < recovery_end:

        progress = (
            runtime_sec - hold_end
        ) / max(
            DRIFT_RECOVERY_SEC,
            1,
        )

        return (
            DRIFT_MAX_TEMP_OFFSET
            * (1.0 - progress)
        )


    # --------------------------------------------------------
    # 5. 다시 정상 환경
    # --------------------------------------------------------

    return 0.0


# ============================================================
# 실제 Sensor 값에 환경 변화 적용
# ============================================================


def apply_environment_scenario(
    sensors,
    runtime_sec,
):

    result = dict(sensors)

    temperature_offset = (
        calculate_temperature_offset(
            runtime_sec
        )
    )


    if temperature_offset <= 0:
        return result


    # --------------------------------------------------------
    # 실제 환경에서는 온도 Sensor마다
    # 외기온의 영향을 완전히 동일하게
    # 받지 않는다고 가정
    # --------------------------------------------------------

    temperature_weights = {
        "TS1": 1.00,
        "TS2": 0.90,
        "TS3": 0.80,
        "TS4": 0.70,
    }


    for sensor, weight in (
        temperature_weights.items()
    ):

        result[sensor] = (
            result[sensor]
            + temperature_offset * weight
        )


    return result


# ============================================================
# Kafka Raw Sensor Message
# ============================================================
#
# 전송되는 최종 구조
#
# {
#     "timestamp": "...",
#     "sensors": {
#         "PS1": ...,
#         ...
#         "SE": ...
#     }
# }
#
# ============================================================


def build_message(sensor_values):

    timestamp = (
        datetime.now()
        .astimezone()
        .isoformat(
            timespec="seconds"
        )
    )


    return {

        "timestamp": timestamp,

        "sensors": {

            sensor: round(
                float(value),
                3,
            )

            for sensor, value
            in sensor_values.items()
        },
    }


# ============================================================
# 현재 가상 공장 환경 상태
#
# 이것은 Console 확인용이다.
# Kafka에는 절대 전송하지 않는다.
# ============================================================


def get_scenario_phase(runtime_sec):

    if SENSOR_SCENARIO == "normal":
        return "NORMAL"


    rise_start = (
        DRIFT_START_SEC
    )

    rise_end = (
        rise_start
        + DRIFT_RISE_SEC
    )

    hold_end = (
        rise_end
        + DRIFT_HOLD_SEC
    )

    recovery_end = (
        hold_end
        + DRIFT_RECOVERY_SEC
    )


    if runtime_sec < rise_start:
        return "NORMAL"

    if runtime_sec < rise_end:
        return "ENVIRONMENT_CHANGE"

    if runtime_sec < hold_end:
        return "HIGH_TEMP_ENVIRONMENT"

    if runtime_sec < recovery_end:
        return "RECOVERY"

    return "NORMAL"


# ============================================================
# Main
# ============================================================


def main():

    print("=" * 70)

    print(
        "HydroTwin Virtual Factory "
        "Realtime Sensor Generator"
    )

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
        f"Sensor Count : "
        f"{len(SENSOR_HZ)}"
    )

    print(
        f"Interval     : "
        f"{SEND_INTERVAL_SEC} sec"
    )

    print(
        f"Scenario     : "
        f"{SENSOR_SCENARIO}"
    )


    if SENSOR_SCENARIO == "summer_drift":

        print(
            f"Normal Time  : "
            f"{DRIFT_START_SEC} sec"
        )

        print(
            f"Drift Rise   : "
            f"{DRIFT_RISE_SEC} sec"
        )

        print(
            f"Drift Hold   : "
            f"{DRIFT_HOLD_SEC} sec"
        )

        print(
            f"Recovery     : "
            f"{DRIFT_RECOVERY_SEC} sec"
        )

        print(
            f"Max Temp +   : "
            f"{DRIFT_MAX_TEMP_OFFSET} C"
        )


    print("=" * 70)

    print(
        "Virtual Factory started."
    )

    print("=" * 70)


    reader = UCIRealtimeReader()

    runtime_sec = 0

    next_send_time = (
        time.monotonic()
    )


    try:

        while True:

            # ------------------------------------------------
            # UCI의 다음 60초 Sensor 기록
            # ------------------------------------------------

            block = (
                reader.read_next_block()
            )


            # ------------------------------------------------
            # 실제 시간이 흐르는 것처럼
            # 60초 데이터를 1초씩 순차 발생
            # ------------------------------------------------

            for second in range(60):

                sensors = {

                    sensor: values[second]

                    for sensor, values
                    in block.items()
                }


                # --------------------------------------------
                # 실제 공장 환경 변화 적용
                # --------------------------------------------

                sensors = (
                    apply_environment_scenario(
                        sensors,
                        runtime_sec,
                    )
                )


                # --------------------------------------------
                # Kafka Raw Message 생성
                # --------------------------------------------

                message = (
                    build_message(
                        sensors
                    )
                )


                # --------------------------------------------
                # Kafka 전송
                # --------------------------------------------

                producer.send(
                    KAFKA_TOPIC,
                    message,
                ).get(
                    timeout=10
                )


                # --------------------------------------------
                # Console에서만 현재 환경 상태 확인
                #
                # Kafka Message에는 phase가 들어가지 않는다.
                # --------------------------------------------

                phase = (
                    get_scenario_phase(
                        runtime_sec
                    )
                )


                print(
                    f"[{phase}] "
                    f"{message['timestamp']} "
                    f"PS1={message['sensors']['PS1']} "
                    f"FS1={message['sensors']['FS1']} "
                    f"TS1={message['sensors']['TS1']} "
                    f"TS2={message['sensors']['TS2']} "
                    f"VS1={message['sensors']['VS1']}"
                )


                runtime_sec += 1


                # --------------------------------------------
                # 실제 1초 주기 유지
                # --------------------------------------------

                next_send_time += (
                    SEND_INTERVAL_SEC
                )


                sleep_sec = (
                    next_send_time
                    - time.monotonic()
                )


                if sleep_sec > 0:

                    time.sleep(
                        sleep_sec
                    )


    except KeyboardInterrupt:

        print()

        print(
            "Virtual Factory stopped."
        )


    finally:

        producer.flush()

        producer.close()

        reader.close()


if __name__ == "__main__":
    main()
