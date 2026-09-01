"""Produce one 728-sample native-rate batch per second on a separate topic."""

import argparse
import json
import os
import sys
import time
from pathlib import Path

from kafka import KafkaProducer
from kafka.admin import KafkaAdminClient, NewTopic
from kafka.errors import TopicAlreadyExistsError


PROJECT_ROOT = Path(__file__).resolve().parents[1]
SIMULATOR_SOURCE = PROJECT_ROOT / "src" / "simulator"
if str(SIMULATOR_SOURCE) not in sys.path:
    sys.path.insert(0, str(SIMULATOR_SOURCE))

from native_rate_utils import (  # noqa: E402
    NATIVE_TOPIC,
    PROCESSED_DIR,
    NativeRateRuntime,
    V5BaselineRuntime,
    create_native_message,
    write_json,
)


DEFAULT_BROKER = os.getenv("KAFKA_BROKER", "localhost:9092")
LAST_MESSAGE_FILE = PROCESSED_DIR / "latest_native_message.json"


def ensure_native_topic(broker):
    admin = KafkaAdminClient(bootstrap_servers=broker, client_id="hydrotwin-native-admin")
    try:
        existing = admin.list_topics()
        if NATIVE_TOPIC in existing:
            return "already_exists"
        try:
            admin.create_topics(
                [NewTopic(name=NATIVE_TOPIC, num_partitions=1, replication_factor=1)],
                validate_only=False,
            )
            return "created"
        except TopicAlreadyExistsError:
            return "already_exists"
    finally:
        admin.close()


def parse_args():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--broker", default=DEFAULT_BROKER)
    parser.add_argument("--interval", type=float, default=1.0)
    parser.add_argument("--messages", type=int, default=0, help="0 means run forever")
    parser.add_argument("--seed-record", type=int, default=1764)
    return parser.parse_args()


def main():
    args = parse_args()
    if args.interval <= 0 or args.messages < 0:
        raise ValueError("interval must be positive and messages non-negative")
    topic_status = ensure_native_topic(args.broker)
    v5_runtime = V5BaselineRuntime(args.seed_record)
    native_runtime = NativeRateRuntime()
    producer = KafkaProducer(
        bootstrap_servers=args.broker,
        value_serializer=lambda value: json.dumps(
            value, ensure_ascii=False, separators=(",", ":")
        ).encode("utf-8"),
        acks="all",
        retries=5,
    )
    print("=" * 84)
    print("HydroTwin Native Sampling Rate Producer")
    print("=" * 84)
    print(f"Broker / Topic : {args.broker} / {NATIVE_TOPIC}")
    print(f"Topic status   : {topic_status}")
    print(f"Interval       : {args.interval:.3f}s")
    print(f"Messages       : {'unlimited' if args.messages == 0 else args.messages}")
    print("Schema         : timestamp + sensors arrays only")

    next_send = time.monotonic()
    sent = 0
    try:
        while args.messages == 0 or sent < args.messages:
            baseline_window, phase_window = v5_runtime.context()
            baseline_started = time.perf_counter()
            baseline = v5_runtime.predict_next()
            baseline_seconds = time.perf_counter() - baseline_started
            sensors, native_timings = native_runtime.generate(
                baseline, baseline_window, phase_window
            )
            message = create_native_message(sensors)
            serialized_size = len(
                json.dumps(message, ensure_ascii=False, separators=(",", ":")).encode("utf-8")
            )
            producer.send(NATIVE_TOPIC, value=message).get(timeout=10)
            write_json(LAST_MESSAGE_FILE, message)
            sent += 1
            print(
                f"[sent {sent:4d}] samples=728 size={serialized_size}B "
                f"v5={baseline_seconds * 1000:.3f}ms "
                f"native100={native_timings['native_100hz_inference_seconds'] * 1000:.3f}ms "
                f"native10={native_timings['native_10hz_inference_seconds'] * 1000:.3f}ms",
                flush=True,
            )
            next_send += args.interval
            delay = next_send - time.monotonic()
            if delay > 0 and (args.messages == 0 or sent < args.messages):
                time.sleep(delay)
    finally:
        producer.flush(timeout=10)
        producer.close(timeout=10)


if __name__ == "__main__":
    main()
