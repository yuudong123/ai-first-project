"""Kafka의 설비 3대 데이터를 분리하여 10초 AI 추론 결과를 저장한다."""
import json
import math
import os
import sys
import uuid
from collections import deque
from datetime import datetime, timezone
from pathlib import Path

# `python kafka/consumer.py`로 실행해도 프로젝트 패키지를 찾도록 한다.
PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0,str(PROJECT_ROOT))

from kafka import KafkaConsumer
from src import hydrotwin_pipeline as p
from src.runtime.common import SENSORS, now, write_state
from src.runtime.inference import diagnose

EQUIPMENT_IDS = ('station-01','station-02','station-03')


class EquipmentInference:
    def __init__(self, predict, run_id=None):
        self.predict = predict
        self.run_id = run_id or uuid.uuid4().hex
        self.buffers = {key:deque(maxlen=10) for key in EQUIPMENT_IDS}
        self.states = {}
        self.event_id = 0
        self.timestamps = {}
        self.segments = dict.fromkeys(EQUIPMENT_IDS,0)
        self.source_positions = {}

    def update(self, payload, current_time=None):
        equipment = payload.get('equipment_id')
        if equipment not in self.buffers:
            raise ValueError('알 수 없는 설비 ID')
        timestamp = datetime.fromisoformat(payload['timestamp'])
        current_time = current_time or datetime.now(timezone.utc)
        if timestamp.tzinfo is None or not -2 <= (current_time-timestamp).total_seconds() <= 5:
            raise ValueError('센서 시간대 누락 또는 데이터 지연')
        sensors = payload.get('sensors',{})
        if not all(isinstance(sensors.get(s),(int,float)) and not isinstance(sensors[s],bool)
                   and math.isfinite(sensors[s]) for s in SENSORS):
            raise ValueError('17개 센서의 유효한 숫자가 필요합니다.')
        previous = self.timestamps.get(equipment)
        if previous is not None and timestamp <= previous:
            return None  # 중복·역순 메시지를 새 관측값으로 세지 않는다.
        marker = None
        if any(key in payload for key in ('run_id','event_id','segment_id')):
            if (not isinstance(payload.get('run_id'),str) or not payload['run_id'] or
                type(payload.get('event_id')) is not int or payload['event_id'] < 1 or
                type(payload.get('segment_id')) is not int or payload['segment_id'] < 0):
                raise ValueError('생성기 실행·이벤트·구간 정보가 불완전합니다.')
            marker = (payload['run_id'],payload['event_id'],payload['segment_id'])
        previous_marker = self.source_positions.get(equipment)
        if marker and previous_marker and marker[0]==previous_marker[0] and marker[1]<=previous_marker[1]:
            return None
        source_changed = (
            (marker is None) != (previous_marker is None) or
            (marker is not None and previous_marker is not None and
             (marker[0]!=previous_marker[0] or marker[2]!=previous_marker[2] or marker[1]!=previous_marker[1]+1))
        )
        if previous is not None and (source_changed or not .5 <= (timestamp-previous).total_seconds() <= 1.5):
            self.buffers[equipment].clear()
            self.segments[equipment] += 1
        self.source_positions[equipment] = marker
        self.timestamps[equipment] = timestamp
        rows = self.buffers[equipment]
        rows.append([float(sensors[s]) for s in SENSORS])
        prediction = self.predict(list(rows)) if len(rows)==10 else {
            'status':'warming_up','observed_window_sec':len(rows),'components':{}}
        sequence = self.states.get(equipment,{}).get('event_id',0)+1
        self.states[equipment] = {'equipment_id':equipment,'event_id':sequence,
            'run_id':payload.get('run_id',self.run_id),'segment_id':self.segments[equipment],
            'cycle_id':(sequence-1)//60+1,'elapsed_sec':sequence,
            'generated_at':payload['timestamp'],'updated_at':now(),'received_at':now(),
            'sensors':{s:float(sensors[s]) for s in SENSORS},'prediction':prediction}
        self.event_id += 1
        # 최상위 값은 응답 스키마용이며 웹·Unity는 반드시 설비 배열을 사용한다.
        first = self.states.get('station-01',self.states[equipment])
        return {**first,'event_id':self.event_id,'updated_at':now(),
            'source':'multi','equipment_states':[self.states[k] for k in EQUIPMENT_IDS if k in self.states],
            'expected_equipment_ids':list(EQUIPMENT_IDS)}


def main():
    broker = os.getenv('KAFKA_BROKER','localhost:9092')
    topic = os.getenv('KAFKA_TOPIC','hydraulic.sensor.multi.raw')
    bundle = p.load_model_bundle()
    if bundle['window_sec']!=10:
        raise RuntimeError('10초 분류 모델이 필요합니다.')
    version = str(p.MODEL_PATH.stat().st_mtime_ns)
    engine = EquipmentInference(lambda rows:diagnose(bundle,rows))
    consumer = KafkaConsumer(topic,bootstrap_servers=[broker],group_id=None,enable_auto_commit=False,
        auto_offset_reset='latest',allow_auto_create_topics=False,
        value_deserializer=lambda data:json.loads(data.decode('utf-8')))
    try:
        if topic not in consumer.topics():
            raise RuntimeError('설비 3대 멀티 토픽이 없습니다.')
        print(f'설비 3대 추론 시작: {broker} / {topic}',flush=True)
        for message in consumer:
            new_version = str(p.MODEL_PATH.stat().st_mtime_ns)
            if new_version != version:
                candidate = p.load_model_bundle()
                if candidate['window_sec'] != 10:
                    raise RuntimeError('교체된 모델의 입력 시간이 10초가 아닙니다.')
                bundle,version = candidate,new_version
                engine.predict = lambda rows:diagnose(bundle,rows)
            try:
                result = engine.update(message.value)
            except (KeyError,ValueError,TypeError) as error:
                print(f'입력 제외: {error}',flush=True)
                continue
            if result is not None:
                result['model_version'] = version
                result['source_broker'],result['source_topic'] = broker,topic
                for state in result['equipment_states']:
                    state['model_version'] = version
                write_state('latest.json',result)
    finally:
        consumer.close()


if __name__=='__main__':
    main()
