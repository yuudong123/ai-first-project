"""설비 3대가 각각 1초 간격으로 들어오는지 확인한다."""
import argparse
import json
import os
from collections import defaultdict

from kafka import KafkaConsumer

BROKER = os.environ.get('KAFKA_BROKER','localhost:9092')
TOPIC = os.environ.get('KAFKA_TOPIC','hydraulic.sensor.multi.raw')
EQUIPMENT_IDS = ('station-01','station-02','station-03')


def main():
    parser=argparse.ArgumentParser()
    parser.add_argument('--count',type=int,default=20,help='설비별로 측정할 메시지 개수')
    args=parser.parse_args()
    consumer=KafkaConsumer(TOPIC,bootstrap_servers=BROKER,
        value_deserializer=lambda value:json.loads(value.decode('utf-8')),
        auto_offset_reset='latest',group_id=None,consumer_timeout_ms=15000)
    arrivals=defaultdict(list)
    last_timestamp={}
    print(f'설비별 메시지 {args.count}개 수신 대기: {BROKER} / {TOPIC}')
    for message in consumer:
        payload=message.value
        equipment=payload.get('equipment_id')
        if equipment not in EQUIPMENT_IDS:
            raise RuntimeError(f'허용되지 않은 설비 ID: {equipment}')
        timestamp=payload.get('timestamp')
        if equipment in last_timestamp:
            from datetime import datetime
            gap=(datetime.fromisoformat(timestamp)-datetime.fromisoformat(last_timestamp[equipment])).total_seconds()
            arrivals[equipment].append(gap)
        last_timestamp[equipment]=timestamp
        if all(len(arrivals[equipment])>=args.count-1 for equipment in EQUIPMENT_IDS):
            break
    consumer.close()
    missing=[equipment for equipment in EQUIPMENT_IDS if len(arrivals[equipment])<args.count-1]
    if missing:
        raise RuntimeError(f'15초 안에 설비별 데이터를 충분히 받지 못했습니다: {missing}')
    for equipment in EQUIPMENT_IDS:
        values=arrivals[equipment]
        average=sum(values)/len(values)
        verdict='정상' if 0.8<=average<=1.2 else '확인 필요'
        print(f'{equipment}: 평균 {average:.3f}초 / 최소 {min(values):.3f} / 최대 {max(values):.3f} / {verdict}')


if __name__=='__main__':
    main()
