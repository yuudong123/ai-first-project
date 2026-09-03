"""학습된 V5 시계열 모델로 초당 17개 센서값을 생성한다."""
import importlib.util
import json
import os
import time
import uuid
import numpy as np
import joblib
import tensorflow as tf
from kafka import KafkaProducer
from src.runtime.common import ROOT, SENSORS, BROKER, TOPIC, now, write_state
from src.runtime.scenario import RandomSeason, ScenarioConfig
from src.runtime.seed_schedule import SeedSchedule


def main():
    # 기존 생성 모델의 입력·상태 갱신 함수를 재사용한다.
    spec = importlib.util.spec_from_file_location('v5_producer', ROOT / 'kafka/virtual_factory_producer_v5.py')
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    with np.load(ROOT / 'data/processed/simulator/uci_1hz_17sensors.npz') as data:
        records = data['data']
    profile = np.loadtxt(ROOT / 'data/raw/uci_hydraulic/extracted/profile.txt')
    path = ROOT / 'models/simulator'
    model = tf.keras.models.load_model(path / 'virtual_factory_generator_v5.keras', compile=False)
    input_scaler = joblib.load(path / 'input_scaler_v5.joblib')
    offset_scaler = joblib.load(path / 'offset_scaler_v5.joblib')
    with np.load(path / 'sensor_bounds_v5.npz') as bounds:
        sensor_min, sensor_max = bounds['sensor_min'].copy(), bounds['sensor_max'].copy()
    config = ScenarioConfig(
        minimum_interval=int(os.getenv('DRIFT_INTERVAL_MIN_SEC','60')),
        maximum_interval=int(os.getenv('DRIFT_INTERVAL_MAX_SEC','1200')),
        temperature_min=float(os.getenv('TEMP_OFFSET_MIN','-4')),
        temperature_max=float(os.getenv('TEMP_OFFSET_MAX','4')),
        pressure_percent=float(os.getenv('PRESSURE_OFFSET_PERCENT','10')),
        ramp_seconds=int(os.getenv('DRIFT_RAMP_SEC','30')),
        initial_normal_seconds=int(os.getenv('INITIAL_NORMAL_SEC','120')),
    )
    season = RandomSeason(config)
    schedule = SeedSchedule(profile, initial_seconds=config.initial_normal_seconds,
                            segment_seconds=int(os.getenv('OPERATING_SEGMENT_SEC','60')))
    runtime = None
    previous_seed = None
    producer = KafkaProducer(bootstrap_servers=BROKER, acks='all',
                            value_serializer=lambda v: json.dumps(v).encode('utf-8'))
    run_id = uuid.uuid4().hex
    elapsed = 0
    baseline = []
    deadline = time.monotonic()
    print('V5 모델 로드 완료: 기준 운전 120초 이후 안정 2구간 / 불안정 1구간', flush=True)
    while True:
        seed_id, segment_id, reference_context = schedule.select(elapsed)
        if seed_id != previous_seed:
            runtime = module.V5NormalRuntime(model, input_scaler, offset_scaler, sensor_min, sensor_max,
                records[seed_id,:30,:], SENSORS.index('PS4'))
            previous_seed = seed_id
            print(f'생성 초기값 변경: 사이클 {seed_id+1}, 초기 stable_flag={int(profile[seed_id,4])} (생성값 정답 아님)',flush=True)
        raw = runtime.predict_next()
        if elapsed < config.initial_normal_seconds:
            baseline.append(raw.copy())
        temp_offset, pressure_percent = season.update(elapsed)
        pressure_base = np.mean(baseline, axis=0)
        sensor_offsets = {s: temp_offset for s in SENSORS if s.startswith('TS')}
        sensor_offsets.update({s: float(pressure_base[i]*pressure_percent/100)
                               for i,s in enumerate(SENSORS) if s.startswith('PS')})
        sensors = {s: round(float(raw[i])+sensor_offsets.get(s,0),6) for i,s in enumerate(SENSORS)}
        timestamp = now()
        # 경계/기준 운전 문맥만 전달한다. 초기 정답 라벨과 주입 offset은 모델 입력에 넣지 않는다.
        producer.send(TOPIC, {'run_id':run_id,'event_id':elapsed+1,'timestamp':timestamp,'sensors':sensors,
                             'segment_id':segment_id,'reference_context':reference_context}).get(timeout=10)
        # 주입 설정과 초기 라벨은 로컬 진단 기록일 뿐, 생성 데이터의 정답 라벨이 아니다.
        write_state('scenario.json', {'updated_at':timestamp,'elapsed_sec':elapsed,
            'run_id':run_id,'segment_id':segment_id,'seed_cycle_id':seed_id+1,
            'seed_stable_flag':int(profile[seed_id,4]),'label_scope':'initial_seed_only_not_generated_ground_truth',
            'reference_context':reference_context,'operating_segment_sec':schedule.segment_seconds,
            'event_id':season.event_id,'temperature_offset':temp_offset,'pressure_percent':pressure_percent,
            'next_drift_in_sec':max(0,season.next_start-elapsed),'sensor_offsets':sensor_offsets,
            'interval_range_sec':[config.minimum_interval,config.maximum_interval], 'source':'V5 LSTM'})
        elapsed += 1
        deadline += 1
        time.sleep(max(0,deadline-time.monotonic()))
        if time.monotonic()-deadline > 1:
            deadline = time.monotonic()


if __name__ == '__main__':
    main()
