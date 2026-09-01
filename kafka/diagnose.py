"""
kafka/diagnose.py
==================
TCP 연결은 되는데 메시지를 못 받을 때, 어느 단계에서 막히는지 확인하는 진단 스크립트.

확인하는 것:
    1) 브로커에 실제로 접속됐는지 (bootstrap_connected)
    2) 브로커가 갖고 있는 토픽 목록 (topic 이름이 정확한지)
    3) 우리가 찾는 토픽의 파티션 정보
    4) earliest(맨 처음)부터 읽었을 때 메시지가 하나라도 있는지

실행:
    python kafka/diagnose.py
"""

import os

from kafka import KafkaConsumer, KafkaAdminClient
from kafka.errors import KafkaError

BROKER = os.environ.get("KAFKA_BROKER", "192.168.133.108:9092")
TOPIC = os.environ.get("KAFKA_TOPIC", "hydraulic.sensor.raw")


def main():
    print(f"브로커: {BROKER}")
    print(f"찾는 토픽: {TOPIC}")
    print("=" * 50)

    # 1) Admin client로 브로커 자체 정보 확인
    try:
        admin = KafkaAdminClient(bootstrap_servers=BROKER, client_id="diagnose")
        print("[1] AdminClient 접속: 성공")
        cluster = admin._client.cluster
        print(f"    브로커 목록: {[str(b) for b in cluster.brokers()]}")
        admin.close()
    except KafkaError as e:
        print(f"[1] AdminClient 접속 실패: {e}")
        print("    -> 브로커 주소/포트가 맞는지, advertised listener 설정 문제인지 확인 필요")
        return

    # 2) 전체 토픽 목록 확인
    try:
        consumer = KafkaConsumer(bootstrap_servers=BROKER, client_id="diagnose-topics")
        topics = consumer.topics()
        print(f"\n[2] 브로커에 있는 전체 토픽 ({len(topics)}개):")
        for t in sorted(topics):
            marker = " <-- 우리가 찾는 토픽" if t == TOPIC else ""
            print(f"    - {t}{marker}")
        if TOPIC not in topics:
            print(f"\n    ⚠ '{TOPIC}' 토픽이 목록에 없습니다. 토픽 이름 오타이거나 아직 안 만들어졌을 수 있습니다.")
        consumer.close()
    except KafkaError as e:
        print(f"[2] 토픽 목록 조회 실패: {e}")
        return

    if TOPIC not in topics:
        return

    # 3) 해당 토픽의 파티션 정보
    consumer = KafkaConsumer(bootstrap_servers=BROKER, client_id="diagnose-partitions")
    partitions = consumer.partitions_for_topic(TOPIC)
    print(f"\n[3] '{TOPIC}' 파티션: {partitions}")
    consumer.close()

    # 4) earliest부터 읽었을 때 메시지가 있는지 (5초만 대기)
    print(f"\n[4] earliest(맨 처음)부터 메시지 존재 여부 확인 (5초 대기)...")
    consumer = KafkaConsumer(
        TOPIC,
        bootstrap_servers=BROKER,
        auto_offset_reset="earliest",
        consumer_timeout_ms=5000,
        client_id="diagnose-earliest",
        group_id=None,  # group 없이 = 매번 새로 처음부터 읽기
    )
    count = 0
    for message in consumer:
        count += 1
        if count == 1:
            print(f"    첫 메시지 확인됨 (offset={message.offset}, partition={message.partition})")
            print(f"    내용 미리보기: {message.value[:200]}")
        if count >= 5:
            break
    consumer.close()

    print(f"\n    총 {count}개 메시지 확인됨 (5초 안에 5개까지만 셈)")
    if count == 0:
        print("    -> 토픽은 있지만 메시지가 하나도 없는 상태입니다. V5 쪽에서 실제로 발행 중인지 확인 필요.")
    else:
        print("    -> 토픽에 메시지가 있습니다. test_interval.py에서 group_id 충돌이나")
        print("       auto_offset_reset='latest' 때문에 새 메시지를 놓쳤을 가능성이 있습니다.")


if __name__ == "__main__":
    main()
