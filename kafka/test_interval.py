"""
kafka/test_interval.py
========================
V5 Virtual Factory가 정말로 1초 간격으로 메시지를 보내는지 확인하는 테스트 스크립트.
메시지를 받을 때마다 직전 메시지와의 시간 간격(초)을 출력하고,
N개를 받으면 평균/최대/최소 간격을 요약해서 보여준다.

실행:
    python kafka/test_interval.py
    python kafka/test_interval.py --count 30   (기본 20개)
"""

import argparse
import json
import os
import time

from kafka import KafkaConsumer

BROKER = os.environ.get("KAFKA_BROKER", "192.168.133.108:9092")
TOPIC = os.environ.get("KAFKA_TOPIC", "hydraulic.sensor.raw")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--count", type=int, default=20, help="측정할 메시지 개수")
    args = parser.parse_args()

    print(f"접속 중... (broker={BROKER}, topic={TOPIC})")
    consumer = KafkaConsumer(
        TOPIC,
        bootstrap_servers=BROKER,
        value_deserializer=lambda v: json.loads(v.decode("utf-8")),
        auto_offset_reset="latest",
        group_id="hydraulic-interval-test",  # 다른 컨슈머 그룹과 겹치지 않게 별도 지정
        consumer_timeout_ms=15000,  # 15초 안에 메시지가 안 오면 타임아웃
    )

    print(f"메시지 {args.count}개 수신 대기 중 (최대 15초 무응답 시 종료)...\n")

    intervals = []
    last_recv_time = None
    received = 0

    try:
        for message in consumer:
            now = time.time()
            payload = message.value
            source_ts = payload.get("timestamp", "?")

            if last_recv_time is not None:
                gap = now - last_recv_time
                intervals.append(gap)
                print(f"[{received+1:>3}] 수신 간격: {gap:5.2f}초   (원본 timestamp: {source_ts})")
            else:
                print(f"[  1] 첫 메시지 수신   (원본 timestamp: {source_ts})")

            last_recv_time = now
            received += 1

            if received >= args.count:
                break
    except Exception as e:
        print(f"\n중단됨: {e}")

    print()
    if not intervals:
        print("메시지를 2개 이상 못 받아서 간격을 계산할 수 없습니다. "
              "브로커 주소/토픽/네트워크 연결을 확인하세요.")
        return

    avg = sum(intervals) / len(intervals)
    print("=" * 40)
    print(f"수신한 메시지: {received}개")
    print(f"평균 간격: {avg:.2f}초")
    print(f"최소 간격: {min(intervals):.2f}초")
    print(f"최대 간격: {max(intervals):.2f}초")
    if 0.8 <= avg <= 1.2:
        print("판정: 1초 간격 맞음 (정상)")
    else:
        print("판정: 1초 간격과 차이가 있음 (네트워크 지연이거나 실제 발행 주기가 다를 수 있음)")


if __name__ == "__main__":
    main()
