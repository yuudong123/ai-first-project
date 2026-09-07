"""V5 시계열 모델로 설비 3대의 센서값을 생성해 Kafka에 1초마다 전송한다."""

from __future__ import annotations

import argparse
import json
import os
import sys
import time
import uuid
from pathlib import Path

os.environ.setdefault("KERAS_BACKEND", "jax")

import joblib
import keras
import numpy as np
from kafka import KafkaProducer

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))
SIMULATOR_SOURCE = PROJECT_ROOT / "src" / "simulator"
if str(SIMULATOR_SOURCE) not in sys.path:
    sys.path.insert(0, str(SIMULATOR_SOURCE))

from v5_generation_utils import CYCLE_SECONDS, SENSOR_COUNT, WINDOW_SIZE  # noqa: E402
from v5_multi_station_utils import (  # noqa: E402
    DEFAULT_SEED_RECORDS,
    EQUIPMENT_IDS,
    MixedSeedController,
    V5StationRuntime,
    assert_station_values_differ,
    create_multi_raw_message,
    current_timestamp,
    validate_multi_raw_message,
    validate_seed_records,
)
from src.runtime.common import write_state  # noqa: E402
from src.runtime.scenario import RandomSeason, ScenarioConfig  # noqa: E402


DATA_DIR = PROJECT_ROOT / "data" / "processed" / "simulator"
MODEL_DIR = PROJECT_ROOT / "models" / "simulator"
RAW_FILE = DATA_DIR / "uci_1hz_17sensors.npz"
PROFILE_FILE = (
    PROJECT_ROOT / "data" / "raw" / "uci_hydraulic" / "extracted" / "profile.txt"
)
MODEL_FILE = MODEL_DIR / "virtual_factory_generator_v5.keras"
INPUT_SCALER_FILE = MODEL_DIR / "input_scaler_v5.joblib"
OFFSET_SCALER_FILE = MODEL_DIR / "offset_scaler_v5.joblib"
BOUNDS_FILE = MODEL_DIR / "sensor_bounds_v5.npz"
METADATA_FILE = MODEL_DIR / "generator_metadata_v5.json"
DEFAULT_BROKER = os.getenv("KAFKA_BROKER", "localhost:9092")
DEFAULT_TOPIC = os.getenv("KAFKA_TOPIC", "hydraulic.sensor.multi.raw")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--broker", default=DEFAULT_BROKER)
    parser.add_argument("--topic", default=DEFAULT_TOPIC)
    parser.add_argument("--interval", type=float, default=1.0)
    parser.add_argument(
        "--seconds", type=int, default=0,
        help="실행 시간(초). 0이면 중단할 때까지 계속 실행",
    )
    parser.add_argument("--quiet", action="store_true")
    parser.add_argument("--seed", type=int, help="시나리오의 무작위 선택을 재현할 때 지정")
    parser.add_argument(
        "--dry-run", action="store_true",
        help="Kafka에 연결하지 않고 --seconds 동안 빠르게 생성",
    )
    return parser.parse_args()


def load_runtime_resources():
    with np.load(RAW_FILE, allow_pickle=False) as raw_file:
        raw_data = raw_file["data"].astype(np.float32)
        sensor_names = [str(name) for name in raw_file["sensor_names"]]
    profiles = np.loadtxt(PROFILE_FILE)
    with np.load(BOUNDS_FILE, allow_pickle=False) as bounds_file:
        sensor_min = bounds_file["sensor_min"].astype(np.float64)
        sensor_max = bounds_file["sensor_max"].astype(np.float64)
    with open(METADATA_FILE, encoding="utf-8") as metadata_file:
        metadata = json.load(metadata_file)

    if raw_data.shape != (2205, 60, SENSOR_COUNT):
        raise ValueError(f"원본 데이터 형태가 예상과 다릅니다: {raw_data.shape}")
    if profiles.shape != (raw_data.shape[0], 5):
        raise ValueError(f"상태 라벨 형태가 예상과 다릅니다: {profiles.shape}")
    if metadata.get("version") != "v5":
        raise ValueError("V5 생성 모델 메타데이터가 아닙니다.")
    if metadata.get("sensor_names") != sensor_names:
        raise ValueError("모델과 원본 데이터의 센서 순서가 다릅니다.")
    if metadata.get("window_size") != WINDOW_SIZE:
        raise ValueError("모델 입력 길이가 현재 생성 코드와 다릅니다.")
    if metadata.get("cycle_seconds") != CYCLE_SECONDS:
        raise ValueError("모델 내부 운전 패턴 길이가 현재 생성 코드와 다릅니다.")

    training_records = int(metadata.get("training_records", -1))
    validation_records = int(metadata.get("validation_records", -1))
    if training_records + validation_records != raw_data.shape[0]:
        raise ValueError("모델 메타데이터의 학습·검증 분할이 원본 데이터와 다릅니다.")
    seeds = validate_seed_records(
        raw_data, profiles, DEFAULT_SEED_RECORDS,
        sensor_min, sensor_max, training_records,
    )

    # 세 설비가 하나의 모델과 스케일러를 읽기 전용으로 공유한다.
    model = keras.models.load_model(MODEL_FILE, compile=False)
    input_scaler = joblib.load(INPUT_SCALER_FILE)
    offset_scaler = joblib.load(OFFSET_SCALER_FILE)
    return (
        raw_data, sensor_names, profiles, sensor_min, sensor_max, seeds,
        model, input_scaler, offset_scaler,
    )


def make_runtimes(
    raw_data, sensor_names, sensor_min, sensor_max,
    model, input_scaler, offset_scaler,
):
    return {
        equipment_id: V5StationRuntime(
            equipment_id=equipment_id,
            seed_record=seed_record,
            model=model,
            input_scaler=input_scaler,
            offset_scaler=offset_scaler,
            sensor_min=sensor_min,
            sensor_max=sensor_max,
            seed_window=raw_data[seed_record, :WINDOW_SIZE],
            ps4_index=sensor_names.index("PS4"),
        )
        for equipment_id, seed_record in DEFAULT_SEED_RECORDS.items()
    }


def main() -> None:
    args = parse_args()
    if args.interval <= 0:
        raise ValueError("전송 주기는 0보다 커야 합니다.")
    if args.seconds < 0:
        raise ValueError("실행 시간은 음수일 수 없습니다.")
    if args.dry_run and args.seconds == 0:
        raise ValueError("송신 없는 점검은 --seconds로 실행 시간을 제한해야 합니다.")
    if not args.dry_run and args.interval != 1.0:
        raise ValueError("실시간 생성 주기는 1초여야 합니다.")

    load_started_at = time.monotonic()
    (
        raw_data, sensor_names, profiles, sensor_min, sensor_max, seeds,
        model, input_scaler, offset_scaler,
    ) = load_runtime_resources()
    runtimes = make_runtimes(
        raw_data, sensor_names, sensor_min, sensor_max,
        model, input_scaler, offset_scaler,
    )
    scenario_config = ScenarioConfig(
        minimum_interval=int(os.getenv("DRIFT_INTERVAL_MIN_SEC", "60")),
        maximum_interval=int(os.getenv("DRIFT_INTERVAL_MAX_SEC", "1200")),
        temperature_min=float(os.getenv("TEMP_OFFSET_MIN", "-4")),
        temperature_max=float(os.getenv("TEMP_OFFSET_MAX", "4")),
        pressure_percent=float(os.getenv("PRESSURE_OFFSET_PERCENT", "10")),
        ramp_seconds=int(os.getenv("DRIFT_RAMP_SEC", "30")),
        initial_normal_seconds=int(os.getenv("INITIAL_NORMAL_SEC", "120")),
    )
    controller = MixedSeedController(
        raw_data,
        profiles,
        runtimes,
        seed=args.seed,
        initial_seconds=scenario_config.initial_normal_seconds,
        segment_seconds=int(os.getenv("OPERATING_SEGMENT_SEC", "60")),
    )
    season = RandomSeason(scenario_config, seed=args.seed)
    baseline_values = {equipment_id: [] for equipment_id in EQUIPMENT_IDS}
    baseline_means = {}
    run_id = uuid.uuid4().hex

    producer = None
    if not args.dry_run:
        producer = KafkaProducer(
            bootstrap_servers=args.broker,
            key_serializer=lambda value: value.encode("utf-8"),
            value_serializer=lambda value: json.dumps(
                value, ensure_ascii=False
            ).encode("utf-8"),
            acks="all",
            retries=5,
        )

    print("=" * 72)
    print("HydroTwin V5 설비 3대 실시간 생성기")
    print("=" * 72)
    print(f"Kafka             : {args.broker} / {args.topic}")
    print(f"모델 로딩          : 1회 ({time.monotonic() - load_started_at:.3f}초)")
    print(f"설비               : {', '.join(EQUIPMENT_IDS)}")
    print(f"초기값             : {DEFAULT_SEED_RECORDS}")
    print(f"검증된 상태         : {[seed.profile for seed in seeds]}")
    print(f"실행 시간           : {args.seconds or '제한 없음'}초")
    print("운전 시나리오       : 최초 정상 120초 → 정상 120초 / 불안정 60초 반복")
    print("초기값 전환         : 10초 완만한 전환", flush=True)

    next_tick = time.monotonic()
    elapsed_sec = 0
    try:
        while args.seconds == 0 or elapsed_sec < args.seconds:
            tick_started_at = time.monotonic()
            timestamp = current_timestamp()

            for equipment_id in controller.advance(elapsed_sec):
                choice = controller.choices[equipment_id]
                print(
                    f"운전 전환: {equipment_id}, {elapsed_sec}초, "
                    f"초기값={choice['seed_record']} (생성값의 정답 라벨 아님)",
                    flush=True,
                )

            station_values = {}
            for equipment_id in EQUIPMENT_IDS:
                values = runtimes[equipment_id].predict_next()
                if elapsed_sec < scenario_config.initial_normal_seconds:
                    baseline_values[equipment_id].append(values.copy())
                station_values[equipment_id] = values
            assert_station_values_differ(station_values)

            if elapsed_sec >= scenario_config.initial_normal_seconds and not baseline_means:
                baseline_means = {
                    equipment_id: np.mean(values, axis=0)
                    for equipment_id, values in baseline_values.items()
                }

            temperature_offset, pressure_percent = season.update(elapsed_sec)
            station_offsets = {}
            emitted_values = {}
            pending_sends = []

            for equipment_id in EQUIPMENT_IDS:
                pressure_base = baseline_means.get(equipment_id)
                if pressure_base is None:
                    pressure_base = np.mean(baseline_values[equipment_id], axis=0)
                sensor_offsets = {
                    sensor: temperature_offset
                    for sensor in sensor_names if sensor.startswith("TS")
                }
                sensor_offsets.update({
                    sensor: float(pressure_base[index] * pressure_percent / 100)
                    for index, sensor in enumerate(sensor_names)
                    if sensor.startswith("PS")
                })
                station_offsets[equipment_id] = sensor_offsets

                values = station_values[equipment_id].copy()
                for index, sensor in enumerate(sensor_names):
                    values[index] += sensor_offsets.get(sensor, 0.0)

                message = create_multi_raw_message(
                    equipment_id, timestamp, sensor_names, values,
                    run_id=run_id,
                    event_id=elapsed_sec + 1,
                    segment_id=controller.choices[equipment_id]["segment_id"],
                    reference_context=controller.choices[equipment_id]["reference_context"],
                )
                validate_multi_raw_message(message, sensor_names)
                emitted_values[equipment_id] = np.asarray(
                    list(message["sensors"].values()), dtype=np.float64
                )
                if producer is not None:
                    pending_sends.append((
                        equipment_id,
                        producer.send(args.topic, key=equipment_id, value=message),
                    ))

            # 세 메시지를 먼저 전송한 뒤 ACK를 확인해 설비별 순차 대기를 피한다.
            for equipment_id, future in pending_sends:
                try:
                    future.get(timeout=10)
                except Exception as error:
                    raise RuntimeError(f"{equipment_id} Kafka 전송 실패") from error

            assert_station_values_differ(emitted_values)

            # 주입 offset은 시뮬레이션 진단값이며 AI 입력이나 정답 라벨이 아니다.
            write_state("scenario.json", {
                "updated_at": timestamp,
                "elapsed_sec": elapsed_sec,
                "run_id": run_id,
                "equipment_ids": list(EQUIPMENT_IDS),
                "drift_event_id": season.event_id,
                "temperature_offset": temperature_offset,
                "pressure_percent": pressure_percent,
                "next_drift_in_sec": max(0, season.next_start - elapsed_sec),
                "equipment_sensor_offsets": station_offsets,
                "interval_range_sec": [
                    scenario_config.minimum_interval,
                    scenario_config.maximum_interval,
                ],
                "source": "V5 LSTM 설비 3대",
            })

            tick_elapsed = time.monotonic() - tick_started_at
            if tick_elapsed >= args.interval:
                print(
                    f"[경고] {elapsed_sec}초 데이터 생성에 {tick_elapsed:.3f}초 소요",
                    flush=True,
                )
            if not args.quiet:
                summary = " ".join(
                    f"{station}=PS1:{emitted_values[station][0]:.3f},"
                    f"TS1:{emitted_values[station][9]:.3f}"
                    for station in EQUIPMENT_IDS
                )
                print(f"[전송 {elapsed_sec:4d}초] {timestamp} {summary}", flush=True)

            elapsed_sec += 1
            if args.seconds and elapsed_sec >= args.seconds:
                break
            # 처리가 오래 걸려도 누락된 횟수만큼 몰아서 보내지 않는다.
            next_tick = max(next_tick + args.interval, time.monotonic())
            wait_seconds = next_tick - time.monotonic()
            if wait_seconds > 0 and not args.dry_run:
                time.sleep(wait_seconds)
    except KeyboardInterrupt:
        print("\n사용자가 실시간 생성을 중단했습니다.", flush=True)
    finally:
        if producer is not None:
            producer.flush()
            producer.close()

    print(f"완료한 생성 시간: {elapsed_sec}초", flush=True)


if __name__ == "__main__":
    main()
