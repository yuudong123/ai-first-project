"""원격 멀티 설비 데이터를 분리해 추론한다. 재학습 입력과 파일을 공유하지 않는다."""
import json
import math
import os
import uuid
from collections import deque
from datetime import datetime, timezone
from kafka import KafkaConsumer, TopicPartition
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

    def update(self, payload, current_time=None):
        equipment = payload.get('equipment_id')
        if equipment not in self.buffers:
            raise ValueError('알 수 없는 설비 ID')
        timestamp = datetime.fromisoformat(payload['timestamp'])
        current_time = current_time or datetime.now(timezone.utc)
        if timestamp.tzinfo is None or not -2 <= (current_time-timestamp).total_seconds() <= 5:
            raise ValueError('원격 센서 시간대 누락 또는 데이터 지연')
        sensors = payload.get('sensors',{})
        if not all(isinstance(sensors.get(s),(int,float)) and not isinstance(sensors[s],bool)
                   and math.isfinite(sensors[s]) for s in SENSORS):
            raise ValueError('17개 센서의 유효한 숫자가 필요합니다.')
        previous = self.timestamps.get(equipment)
        if previous is not None and timestamp <= previous:
            return None  # 중복·역순 메시지를 새 관측값으로 세지 않는다.
        if previous is not None and not .5 <= (timestamp-previous).total_seconds() <= 1.5:
            self.buffers[equipment].clear()
            self.segments[equipment] += 1
        self.timestamps[equipment] = timestamp
        rows = self.buffers[equipment]
        rows.append([float(sensors[s]) for s in SENSORS])
        prediction = self.predict(list(rows)) if len(rows)==10 else {
            'status':'warming_up','observed_window_sec':len(rows),'components':{}}
        sequence = self.states.get(equipment,{}).get('event_id',0)+1
        self.states[equipment] = {'equipment_id':equipment,'event_id':sequence,
            'run_id':self.run_id,'segment_id':self.segments[equipment],
            'cycle_id':(sequence-1)//60+1,'elapsed_sec':sequence,
            'generated_at':payload['timestamp'],'updated_at':now(),'received_at':now(),
            'sensors':{s:float(sensors[s]) for s in SENSORS},'prediction':prediction}
        self.event_id += 1
        # 최상위 값은 구형 API 호환용이며 웹·Unity는 반드시 설비 배열을 사용한다.
        first = self.states.get('station-01',self.states[equipment])
        return {**first,'event_id':self.event_id,'updated_at':now(),
            'source':'remote_multi','equipment_states':[self.states[k] for k in EQUIPMENT_IDS if k in self.states],
            'expected_equipment_ids':list(EQUIPMENT_IDS)}


def main():
    broker = os.environ['KAFKA_BROKER']
    topic = os.getenv('KAFKA_TOPIC','hydraulic.sensor.multi.raw')
    bundle = p.load_model_bundle()
    if bundle['window_sec']!=10:
        raise RuntimeError('10초 분류 모델이 필요합니다.')
    # 원격 시연 중에는 시작 시점 모델을 고정해 로컬 재학습 결과와 혼합하지 않는다.
    version = str(p.MODEL_PATH.stat().st_mtime_ns)
    engine = EquipmentInference(lambda rows:diagnose(bundle,rows))
    consumer = KafkaConsumer(bootstrap_servers=[broker],group_id=None,enable_auto_commit=False,
        allow_auto_create_topics=False,value_deserializer=lambda data:json.loads(data.decode('utf-8')))
    try:
        partitions = consumer.partitions_for_topic(topic)
        if not partitions:
            raise RuntimeError('원격 멀티 설비 토픽이 없습니다.')
        consumer.assign([TopicPartition(topic,i) for i in sorted(partitions)])
        consumer.seek_to_end()  # 과거 메시지나 다른 팀 컨슈머의 커밋 위치를 사용하지 않는다.
        print(f'원격 설비별 추론 시작: {broker} / {topic}',flush=True)
        for message in consumer:
            try:
                result = engine.update(message.value)
            except (KeyError,ValueError,TypeError) as error:
                print(f'입력 제외: {error}',flush=True)
                continue
            if result is not None:
                result['model_version'] = version
                result['source_broker'],result['source_topic'] = broker,topic
                write_state('remote_latest.json',result)
    finally:
        consumer.close()


if __name__=='__main__':
    main()
