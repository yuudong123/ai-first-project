"""
test_kafka_consumer.py
========================
원격 Kafka Broker(V5 Virtual Factory)로부터 실시간 센서 데이터를
정상 수신할 수 있는지 확인하는 최소 진단 스크립트.

조건:
    - Topic: hydraulic.sensor.raw
    - Broker: 192.168.133.108:9092
    - 매 실행마다 새로운 group_id 사용 (기존 offset 영향 회피)
    - auto_offset_reset="earliest"
    - enable_auto_commit=False
    - 메시지 5건 수신 시 종료, 0건이면 RuntimeError

실행:
    python test_kafka_consumer.py
"""

import json
import uuid

from kafka import KafkaConsumer

consumer = KafkaConsumer(
    "hydraulic.sensor.raw",
    bootstrap_servers=["192.168.133.108:9092"],
    group_id="debug-" + str(uuid.uuid4()),
    auto_offset_reset="earliest",
    enable_auto_commit=False,
    value_deserializer=lambda x: json.loads(x.decode("utf-8")),
    consumer_timeout_ms=15000,
)

print("Kafka consumer connected. Waiting for messages...")

count = 0

for message in consumer:
    data = message.value
    print("timestamp:", data.get("timestamp"))
    print("sensors:", data.get("sensors"))
    print("-" * 60)

    count += 1
    if count >= 5:
        break

print("received:", count)

if count == 0:
    raise RuntimeError("No Kafka messages received")
