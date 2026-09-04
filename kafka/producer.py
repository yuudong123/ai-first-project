"""학습된 V5 모델 하나로 설비 3대의 안정·불안정 초기값을 섞어 Kafka에 전송한다."""

from __future__ import annotations

import argparse
import csv
import json
import os
import resource
import sys
import time
import uuid
from pathlib import Path

import joblib
import numpy as np
import keras
from kafka import KafkaProducer

os.environ["KERAS_BACKEND"] = "jax"

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
    V5StationRuntime,
    MixedSeedController,
    assert_station_values_differ,
    clip_for_six_decimal_raw_range,
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
PROFILE_FILE = PROJECT_ROOT / "data" / "raw" / "uci_hydraulic" / "extracted" / "profile.txt"
MODEL_FILE = MODEL_DIR / "virtual_factory_generator_v5.keras"
INPUT_SCALER_FILE = MODEL_DIR / "input_scaler_v5.joblib"
OFFSET_SCALER_FILE = MODEL_DIR / "offset_scaler_v5.joblib"
BOUNDS_FILE = MODEL_DIR / "sensor_bounds_v5.npz"
METADATA_FILE = MODEL_DIR / "generator_metadata_v5.json"
DEFAULT_OUTPUT_DIR = DATA_DIR / "multi_station"
DEFAULT_BROKER = os.getenv("KAFKA_BROKER", "localhost:9092")
DEFAULT_TOPIC = os.getenv("KAFKA_MULTI_TOPIC", os.getenv("KAFKA_TOPIC", "hydraulic.sensor.multi.raw"))
TRAIN_RATIO = 0.8


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--broker", default=DEFAULT_BROKER)
    parser.add_argument("--topic", default=DEFAULT_TOPIC)
    parser.add_argument("--interval", type=float, default=1.0)
    parser.add_argument(
        "--seconds",
        type=int,
        default=0,
        help="Cycle count; zero runs until interrupted",
    )
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    parser.add_argument("--quiet", action="store_true")
    parser.add_argument("--seed", type=int, help="불안정 초기값 선택을 재현할 때 지정")
    parser.add_argument("--dry-run", action="store_true", help="Kafka에 연결·송신하지 않고 제한된 길이를 빠르게 생성")
    return parser.parse_args()


def read_rss_mb() -> float:
    with open("/proc/self/status", encoding="utf-8") as status_file:
        for line in status_file:
            if line.startswith("VmRSS:"):
                return float(line.split()[1]) / 1024.0
    return resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / 1024.0


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
        raise ValueError(f"Unexpected Raw dataset: {raw_data.shape}")
    if profiles.shape != (raw_data.shape[0], 5):
        raise ValueError(f"Unexpected UCI profile shape: {profiles.shape}")
    if metadata.get("version") != "v5":
        raise ValueError("Unexpected V5 metadata version")
    if metadata.get("sensor_names") != sensor_names:
        raise ValueError("V5 sensor names/order changed")
    if metadata.get("window_size") != WINDOW_SIZE:
        raise ValueError("V5 window size changed")
    if metadata.get("cycle_seconds") != CYCLE_SECONDS:
        raise ValueError("V5 cycle length changed")

    validation_start = int(raw_data.shape[0] * TRAIN_RATIO)
    seeds = validate_seed_records(
        raw_data,
        profiles,
        DEFAULT_SEED_RECORDS,
        sensor_min,
        sensor_max,
        validation_start,
    )

    # Intentional and auditable: the V5 Keras model is loaded exactly once.
    model = keras.models.load_model(MODEL_FILE, compile=False)
    input_scaler = joblib.load(INPUT_SCALER_FILE)
    offset_scaler = joblib.load(OFFSET_SCALER_FILE)
    return (
        raw_data,
        sensor_names,
        profiles,
        sensor_min,
        sensor_max,
        seeds,
        model,
        input_scaler,
        offset_scaler,
    )


def make_runtimes(
    raw_data,
    sensor_names,
    sensor_min,
    sensor_max,
    model,
    input_scaler,
    offset_scaler,
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


def write_seed_statistics(output_dir, raw_data, profiles, sensor_names) -> Path:
    path = output_dir / "v5_multi_station_seed_statistics.csv"
    with path.open("w", newline="", encoding="utf-8") as output_file:
        fields = [
            "equipment_id", "seed_record", "profile", "sensor",
            "mean", "std", "min", "max",
        ]
        writer = csv.DictWriter(output_file, fieldnames=fields)
        writer.writeheader()
        for equipment_id, record in DEFAULT_SEED_RECORDS.items():
            window = raw_data[record, :WINDOW_SIZE]
            profile = "/".join(f"{value:g}" for value in profiles[record])
            for index, sensor in enumerate(sensor_names):
                values = window[:, index]
                writer.writerow(
                    {
                        "equipment_id": equipment_id,
                        "seed_record": record,
                        "profile": profile,
                        "sensor": sensor,
                        "mean": f"{values.mean():.10f}",
                        "std": f"{values.std():.10f}",
                        "min": f"{values.min():.10f}",
                        "max": f"{values.max():.10f}",
                    }
                )
    return path


def write_result_artifacts(
    output_dir,
    rows,
    generated,
    phases,
    sensor_names,
    raw_sensor_min,
    raw_sensor_max,
    cycle_times,
    kafka_offsets,
    wall_seconds,
    cpu_seconds,
    rss_samples,
    interrupted,
    topic,
):
    output_dir.mkdir(parents=True, exist_ok=True)
    csv_path = output_dir / "v5_multi_station_runtime.csv"
    fields = ["elapsed_sec", "equipment_id", "timestamp", "run_id", "segment_id",
              "seed_record", "seed_stable_flag"] + sensor_names
    with csv_path.open("w", newline="", encoding="utf-8") as output_file:
        writer = csv.DictWriter(output_file, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)

    stats_path = output_dir / "v5_multi_station_sensor_statistics.csv"
    stats_rows = []
    station_checks = {}
    for equipment_id in EQUIPMENT_IDS:
        values = np.asarray(generated[equipment_id], dtype=np.float64)
        station_range = (values.min(0) >= raw_sensor_min) & (
            values.max(0) <= raw_sensor_max
        )
        first_cycle = values[:CYCLE_SECONDS].mean(0)
        last_cycle = values[-CYCLE_SECONDS:].mean(0)
        previous_cycle = (
            values[-2 * CYCLE_SECONDS : -CYCLE_SECONDS].mean(0)
            if len(values) >= 2 * CYCLE_SECONDS
            else first_cycle
        )
        station_checks[equipment_id] = {
            "message_count": int(len(values)),
            "sensor_count": SENSOR_COUNT,
            "nan_count": int(np.isnan(values).sum()),
            "inf_count": int(np.isinf(values).sum()),
            "raw_range_pass_count": int(station_range.sum()),
            "phase_zero_count": int(np.count_nonzero(np.asarray(phases[equipment_id]) == 0)),
            "max_abs_first_last_cycle_mean_change": float(
                np.abs(last_cycle - first_cycle).max()
            ),
            "max_abs_last_two_cycle_mean_change": float(
                np.abs(last_cycle - previous_cycle).max()
            ),
        }
        for index, sensor in enumerate(sensor_names):
            stats_rows.append(
                {
                    "equipment_id": equipment_id,
                    "sensor": sensor,
                    "mean": values[:, index].mean(),
                    "std": values[:, index].std(),
                    "min": values[:, index].min(),
                    "max": values[:, index].max(),
                    "first_cycle_mean": first_cycle[index],
                    "last_cycle_mean": last_cycle[index],
                    "last_minus_first_cycle_mean": last_cycle[index] - first_cycle[index],
                    "last_minus_previous_cycle_mean": last_cycle[index] - previous_cycle[index],
                    "raw_range_pass": bool(station_range[index]),
                }
            )
    with stats_path.open("w", newline="", encoding="utf-8") as output_file:
        writer = csv.DictWriter(output_file, fieldnames=list(stats_rows[0]))
        writer.writeheader()
        writer.writerows(stats_rows)

    difference_checks = {}
    for left_index, left_id in enumerate(EQUIPMENT_IDS):
        for right_id in EQUIPMENT_IDS[left_index + 1 :]:
            left = np.asarray(generated[left_id])
            right = np.asarray(generated[right_id])
            key = f"{left_id}_vs_{right_id}"
            difference_checks[key] = {
                "identical_timestamps": int(np.all(left == right, axis=1).sum()),
                "mean_absolute_sensor_difference": float(np.abs(left - right).mean()),
            }

    graph_dir = output_dir / "graphs"
    graph_dir.mkdir(parents=True, exist_ok=True)
    graph_paths = []
    try:
        import matplotlib
        matplotlib.use('Agg')
        import matplotlib.pyplot as plt
    except ImportError:
        plt = None
        print('matplotlib 미설치: 생성 CSV·검증 결과만 저장하고 선택 그래프는 생략합니다.',flush=True)
    for sensor in (("TS1", "PS1", "FS1") if plt is not None else ()):
        sensor_index = sensor_names.index(sensor)
        figure, axis = plt.subplots(figsize=(12, 5))
        for equipment_id in EQUIPMENT_IDS:
            axis.plot(
                np.arange(1, len(generated[equipment_id]) + 1),
                np.asarray(generated[equipment_id])[:, sensor_index],
                label=equipment_id,
                linewidth=1.2,
            )
        axis.set_title(f"V5 Multi-Station {sensor}")
        axis.set_xlabel("Runtime second")
        axis.set_ylabel(sensor)
        axis.grid(alpha=0.25)
        axis.legend()
        figure.tight_layout()
        graph_path = graph_dir / f"v5_multi_station_{sensor.lower()}_line.png"
        figure.savefig(graph_path, dpi=150)
        plt.close(figure)
        graph_paths.append(str(graph_path))

    completed_cycles = len(cycle_times)
    message_count = sum(len(values) for values in generated.values())
    performance = {
        "model_load_count": 1,
        "runtime_station_count": len(EQUIPMENT_IDS),
        "completed_seconds": completed_cycles,
        "message_count": message_count,
        "scheduled_messages_per_second": len(EQUIPMENT_IDS),
        "wall_seconds": wall_seconds,
        "effective_messages_per_second": (
            message_count / max(completed_cycles, 1)
        ),
        "kafka_ack_messages": len(kafka_offsets),
        "cycle_generation_and_kafka_seconds": {
            "mean": float(np.mean(cycle_times)),
            "std": float(np.std(cycle_times)),
            "min": float(np.min(cycle_times)),
            "max": float(np.max(cycle_times)),
            "over_one_second_count": int(np.count_nonzero(np.asarray(cycle_times) >= 1.0)),
        },
        "process_cpu_seconds": cpu_seconds,
        "process_cpu_percent_of_one_core": 100.0 * cpu_seconds / max(wall_seconds, 1e-9),
        "rss_mb": {
            "mean": float(np.mean(rss_samples)),
            "min": float(np.min(rss_samples)),
            "max": float(np.max(rss_samples)),
        },
        "interrupted": interrupted,
    }
    performance_path = output_dir / "v5_multi_station_performance.json"
    performance_path.write_text(
        json.dumps(performance, ensure_ascii=False, indent=2), encoding="utf-8"
    )

    validation = {
        "topic": topic,
        "equipment_ids": list(EQUIPMENT_IDS),
        "seed_records": DEFAULT_SEED_RECORDS,
        "seed_schedule": "최초 안정 120초 이후 안정 120초 / 불안정 60초 반복",
        "label_scope": "initial_seed_only_not_generated_ground_truth",
        "seed_label_counts": {equipment: {
            str(label):sum(row['equipment_id']==equipment and row['seed_stable_flag']==label for row in rows)
            for label in (0,1)} for equipment in EQUIPMENT_IDS},
        "station_checks": station_checks,
        "difference_checks": difference_checks,
        "graphs": graph_paths,
    }
    validation_path = output_dir / "v5_multi_station_validation.json"
    validation_path.write_text(
        json.dumps(validation, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    return csv_path, stats_path, performance_path, validation_path


def main() -> None:
    args = parse_args()
    if args.interval <= 0:
        raise ValueError("--interval must be greater than zero")
    if args.seconds < 0:
        raise ValueError("--seconds cannot be negative")
    if args.dry_run and args.seconds == 0:
        raise ValueError('송신 없는 점검은 --seconds로 길이를 제한해야 합니다.')
    if not args.dry_run and args.interval != 1.0:
        raise ValueError('실시간 생성 주기는 1초여야 합니다.')
    args.output_dir.mkdir(parents=True, exist_ok=True)

    load_start = time.monotonic()
    (
        raw_data,
        sensor_names,
        profiles,
        sensor_min,
        sensor_max,
        seeds,
        model,
        input_scaler,
        offset_scaler,
    ) = load_runtime_resources()
    runtimes = make_runtimes(
        raw_data,
        sensor_names,
        sensor_min,
        sensor_max,
        model,
        input_scaler,
        offset_scaler,
    )
    controller = MixedSeedController(raw_data, profiles, runtimes, seed=args.seed)
    scenario_config = ScenarioConfig(
        minimum_interval=int(os.getenv('DRIFT_INTERVAL_MIN_SEC','60')),
        maximum_interval=int(os.getenv('DRIFT_INTERVAL_MAX_SEC','1200')),
        temperature_min=float(os.getenv('TEMP_OFFSET_MIN','-4')),
        temperature_max=float(os.getenv('TEMP_OFFSET_MAX','4')),
        pressure_percent=float(os.getenv('PRESSURE_OFFSET_PERCENT','10')),
        ramp_seconds=int(os.getenv('DRIFT_RAMP_SEC','30')),
        initial_normal_seconds=int(os.getenv('INITIAL_NORMAL_SEC','120')),
    )
    season = RandomSeason(scenario_config,seed=args.seed)
    baseline_values = {equipment_id:[] for equipment_id in EQUIPMENT_IDS}
    run_id = uuid.uuid4().hex
    seed_path = write_seed_statistics(
        args.output_dir, raw_data, profiles, sensor_names
    )
    producer = None if args.dry_run else KafkaProducer(
        bootstrap_servers=args.broker,
        value_serializer=lambda value: json.dumps(
            value, ensure_ascii=False
        ).encode("utf-8"),
        acks="all",
        retries=5,
    )
    raw_sensor_values = raw_data.reshape(-1, SENSOR_COUNT)
    raw_sensor_min = raw_sensor_values.min(axis=0).astype(np.float64)
    raw_sensor_max = raw_sensor_values.max(axis=0).astype(np.float64)

    print("=" * 86)
    print("HydroTwin V5 Multi-Station Virtual Factory")
    print("=" * 86)
    print(f"Broker / Topic       : {args.broker} / {args.topic}")
    print(f"V5 model load count  : 1 ({time.monotonic() - load_start:.3f}s)")
    print(f"Runtime station count: {len(runtimes)}")
    print(f"Seed records         : {DEFAULT_SEED_RECORDS}")
    print(f"Validated profiles   : {[seed.profile for seed in seeds]}")
    print(f"Interval / seconds   : {args.interval:.3f} / {args.seconds or 'unlimited'}")
    print('생성 방식             : 최초 안정 120초 → 안정 120초 / 불안정 60초 반복 (초기값 기준)')
    print(f'Kafka 송신             : {not args.dry_run}; 초기 라벨은 생성값의 정답이 아님',flush=True)
    print(f"Seed statistics      : {seed_path}", flush=True)

    rows = []
    generated = {equipment_id: [] for equipment_id in EQUIPMENT_IDS}
    phases = {equipment_id: [] for equipment_id in EQUIPMENT_IDS}
    cycle_times = []
    kafka_offsets = []
    rss_samples = [read_rss_mb()]
    interrupted = False
    collect_artifacts = args.seconds > 0
    start_wall = time.monotonic()
    start_cpu = time.process_time()
    next_cycle = start_wall
    elapsed_sec = 0
    try:
        while args.seconds == 0 or elapsed_sec < args.seconds:
            cycle_start = time.monotonic()
            timestamp = current_timestamp()
            for equipment_id in controller.advance(elapsed_sec):
                choice = controller.choices[equipment_id]
                print(f'초기값 전환: {equipment_id}, {elapsed_sec}초, {choice} (생성값 정답 아님)',flush=True)
            station_values = {}
            for equipment_id in EQUIPMENT_IDS:
                values = runtimes[equipment_id].predict_next()
                if elapsed_sec < scenario_config.initial_normal_seconds:
                    baseline_values[equipment_id].append(values.copy())
                station_values[equipment_id] = values
                phases[equipment_id].append(runtimes[equipment_id].cycle_position)
            assert_station_values_differ(station_values)

            temperature_offset,pressure_percent = season.update(elapsed_sec)
            station_offsets = {}
            emitted_values = {}

            for equipment_id in EQUIPMENT_IDS:
                pressure_base = np.mean(baseline_values[equipment_id],axis=0)
                sensor_offsets = {
                    sensor:temperature_offset for sensor in sensor_names if sensor.startswith('TS')
                }
                sensor_offsets.update({
                    sensor:float(pressure_base[index]*pressure_percent/100)
                    for index,sensor in enumerate(sensor_names) if sensor.startswith('PS')
                })
                station_offsets[equipment_id] = sensor_offsets
                drifted = station_values[equipment_id].copy()
                for index,sensor in enumerate(sensor_names):
                    drifted[index] += sensor_offsets.get(sensor,0.0)
                values = clip_for_six_decimal_raw_range(
                    drifted, raw_sensor_min, raw_sensor_max
                )
                message = create_multi_raw_message(
                    equipment_id, timestamp, sensor_names, values,
                    run_id=run_id, event_id=elapsed_sec+1,
                    segment_id=controller.choices[equipment_id]['segment_id'],
                    reference_context=controller.choices[equipment_id]['reference_context'],
                )
                validate_multi_raw_message(message, sensor_names)
                serialized_values = np.asarray(
                    list(message["sensors"].values()), dtype=np.float64
                )
                if not (
                    (serialized_values >= raw_sensor_min)
                    & (serialized_values <= raw_sensor_max)
                ).all():
                    raise ValueError(f"{equipment_id} exceeds UCI Raw range")
                if producer is not None:
                    metadata = producer.send(args.topic, value=message).get(timeout=10)
                    if collect_artifacts:
                        kafka_offsets.append((metadata.partition, metadata.offset))
                sent_values = [message["sensors"][name] for name in sensor_names]
                station_values[equipment_id] = values
                emitted_values[equipment_id] = values
                if collect_artifacts:
                    generated[equipment_id].append(sent_values)
                    rows.append(
                        {
                            "elapsed_sec": elapsed_sec,
                            "equipment_id": equipment_id,
                            "timestamp": timestamp,
                            "run_id":run_id,
                            "segment_id":controller.choices[equipment_id]['segment_id'],
                            "seed_record":controller.choices[equipment_id]['seed_record'],
                            "seed_stable_flag":controller.choices[equipment_id]['seed_stable_flag'],
                            **dict(zip(sensor_names, sent_values)),
                        }
                    )

            assert_station_values_differ(emitted_values)

            # 주입값과 초기 라벨은 진단 기록일 뿐 Kafka 정답 라벨이나 모델 입력이 아니다.
            write_state('scenario.json',{
                'updated_at':timestamp,'elapsed_sec':elapsed_sec,'run_id':run_id,
                'equipment_ids':list(EQUIPMENT_IDS),'drift_event_id':season.event_id,
                'temperature_offset':temperature_offset,'pressure_percent':pressure_percent,
                'next_drift_in_sec':max(0,season.next_start-elapsed_sec),
                'equipment_sensor_offsets':station_offsets,
                'interval_range_sec':[scenario_config.minimum_interval,scenario_config.maximum_interval],
                'label_scope':'initial_seed_only_not_generated_ground_truth','source':'V5 LSTM 설비 3대',
            })

            cycle_elapsed = time.monotonic() - cycle_start
            if collect_artifacts:
                cycle_times.append(cycle_elapsed)
                rss_samples.append(read_rss_mb())
            if cycle_elapsed >= args.interval:
                print(
                    f"[WARN] cycle {elapsed_sec} took {cycle_elapsed:.6f}s",
                    flush=True,
                )
            if not args.quiet:
                summary = " ".join(
                    f"{station}=PS1:{station_values[station][0]:.3f},"
                    f"TS1:{station_values[station][9]:.3f}"
                    for station in EQUIPMENT_IDS
                )
                print(
                    f"[sent {elapsed_sec:4d}] {timestamp} {summary} "
                    f"cycle={cycle_elapsed:.4f}s",
                    flush=True,
                )
            elapsed_sec += 1
            if args.seconds and elapsed_sec >= args.seconds:
                break
            next_cycle += args.interval
            sleep_seconds = next_cycle - time.monotonic()
            if sleep_seconds > 0 and not args.dry_run:
                time.sleep(sleep_seconds)
    except KeyboardInterrupt:
        interrupted = True
        print("\nMulti-Station runtime stopped by operator.", flush=True)
    finally:
        if producer is not None:
            producer.flush()
            producer.close()

    wall_seconds = time.monotonic() - start_wall
    cpu_seconds = time.process_time() - start_cpu
    if not collect_artifacts:
        return
    if not rows:
        raise RuntimeError("No Multi-Station messages were produced")
    artifacts = write_result_artifacts(
        args.output_dir,
        rows,
        generated,
        phases,
        sensor_names,
        raw_sensor_min,
        raw_sensor_max,
        cycle_times,
        kafka_offsets,
        wall_seconds,
        cpu_seconds,
        rss_samples,
        interrupted,
        args.topic,
    )
    print(f"Completed cycles       : {elapsed_sec}")
    print(f"Kafka messages acked   : {len(kafka_offsets)}")
    print(f"Mean/max cycle time    : {np.mean(cycle_times):.6f}s / {np.max(cycle_times):.6f}s")
    print(f"CPU / peak RSS         : {100.0 * cpu_seconds / wall_seconds:.3f}% / {max(rss_samples):.3f} MB")
    for artifact in artifacts:
        print(f"[SAVED] {artifact}")


if __name__ == "__main__":
    main()
