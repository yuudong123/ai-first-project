"""별도 Kafka 그룹에서 계절 분포 이동을 감지하고 Jenkins 재학습을 요청한다."""
import logging
from logging.handlers import RotatingFileHandler
import json
import time
import uuid
from src.monitoring.drift_detector import fit_reference, DriftReference, RollingDriftDetector, DriftConfig
from src.runtime.common import STATE_DIR, consumer, read_state, write_state, now, age
from src.runtime.jenkins import trigger
from src.runtime.seed_schedule import window_discontinuity

SEASONAL = ('PS1','PS2','PS3','PS4','PS5','PS6','TS1','TS2','TS3','TS4')


def materially_changed(first, second, reference):
    if not second:
        return True
    return any(abs(first[s]-second.get(s,0)) > (0.35 if s.startswith('TS') else max(abs(reference.sensors[s].mean)*0.015,0.01)) for s in first)


def main():
    STATE_DIR.mkdir(parents=True,exist_ok=True)
    log = logging.getLogger('telemetry')
    log.setLevel(logging.INFO)
    log.addHandler(RotatingFileHandler(STATE_DIR/'observations.jsonl',maxBytes=10_000_000,backupCount=3,encoding='utf-8'))
    run_id = None
    previous_event = None
    previous_segment = None
    baseline,detector = [],None
    previous_offsets,last_requested = {},{}
    count = 0
    last_dispatch = 0
    for message in consumer('hydrotwin-drift-v1'):
        data = message.value
        if age(data['timestamp'])>5:
            continue
        if run_id != data['run_id']:
            run_id = data['run_id']
            baseline,detector = [],None
            previous_offsets,last_requested = {},{}
            previous_event = None
            previous_segment = None
            count = 0
            saved = read_state('reference.json')
            if saved.get('run_id') == run_id:
                detector = RollingDriftDetector(DriftReference.from_dict(saved['reference']),DriftConfig())
                last_requested = read_state('monitor.json').get('last_requested_offsets',{})
        if window_discontinuity(run_id,previous_event,previous_segment,data) and detector is not None:
            detector.reset()
            previous_offsets = {}
            count = 0
        previous_event = data['event_id']
        previous_segment = data.get('segment_id',0)
        sensors = {s:data['sensors'][s] for s in SEASONAL}
        if not data.get('reference_context',True):
            # 운전 상태 변경을 계절 변화로 잘못 학습하지 않는다. 기준 운전끼리만 비교한다.
            if detector is not None:
                detector.reset()
            count,previous_offsets = 0,{}
            last = read_state('monitor.json')
            write_state('monitor.json',{**last,'run_id':run_id,'status':'condition_excluded',
                'drift_detected':False,'candidate_detected':False,'sensor_scores':{},'affected_sensors':[],
                'updated_at':now(),'message':'다른 운전 조건 구간은 계절 offset 추정 및 새 재학습 요청에서 제외'})
            log.info(json.dumps({'timestamp':data['timestamp'],'sensors':sensors,'status':'condition_excluded'},ensure_ascii=False))
            continue
        if detector is None:
            if data['event_id']<=120:
                baseline.append(sensors)
            elif len(baseline)>=60:
                reference = fit_reference(baseline,SEASONAL,min_samples=60)
                detector = RollingDriftDetector(reference,DriftConfig())
                write_state('reference.json',{'run_id':run_id,'reference':reference.to_dict()})
            else:
                write_state('monitor.json',{'status':'needs_baseline','updated_at':now(),
                    'message':'정상 기준 데이터가 부족합니다. 생성기를 재시작하세요.'})
                continue
        if detector is None:
            write_state('monitor.json',{'status':'calibrating','updated_at':now(),'observed_samples':len(baseline),'required_samples':120})
            continue
        result = detector.update(sensors,data['timestamp'])
        count += 1
        offsets = {s:float(v['mean_offset']) for s,v in result['sensor_scores'].items()}
        current_request = read_state('retraining.json')
        busy = current_request.get('status') in ('queued','running')
        if busy and age(current_request['updated_at'])>1200:
            write_state('retraining.json',{**current_request,'status':'failed','message':'재학습 작업 시간 초과','updated_at':now()})
            busy = False
        if count % 60==0 and offsets:
            # 온도/압력의 60초 평균 이동이 두 번 비슷할 때만 계절 증강을 요청한다.
            plateau = bool(previous_offsets) and not materially_changed(offsets,previous_offsets,detector.reference)
            if result['drift_detected'] and plateau and not busy and materially_changed(offsets,last_requested,detector.reference):
                selected = {s:o for s,o in offsets.items() if abs(o)>(0.4 if s.startswith('TS') else max(abs(detector.reference.sensors[s].mean)*0.01,0.01))}
                if selected:
                    request = {'request_id':uuid.uuid4().hex,'run_id':run_id,'status':'drift','sensor_offsets':selected,'created_at':now()}
                    write_state('retrain_request.json',request)
                    write_state('retraining.json',{'request_id':request['request_id'],'status':'queued','updated_at':now(),'sensor_offsets':selected})
                    last_requested = offsets.copy()
                    busy = True
            previous_offsets = offsets.copy()
        current_request = read_state('retraining.json')
        if current_request.get('status')=='queued' and not current_request.get('jenkins_queue') and time.monotonic()-last_dispatch>15:
            last_dispatch = time.monotonic()
            try:
                queue = trigger()
                # 작업이 이미 시작되었다면 running 상태를 queued로 덮어쓰지 않는다.
                latest = read_state('retraining.json')
                if latest.get('status')=='queued':
                    write_state('retraining.json',{**latest,'jenkins_queue':queue,'updated_at':now()})
            except Exception as error:
                print(f'Jenkins 요청 재시도 대기: {type(error).__name__}',flush=True)
        write_state('monitor.json',{**result,'run_id':run_id,'updated_at':now(),
            'last_requested_offsets':last_requested,'estimated_offsets':offsets})
        log.info(json.dumps({'timestamp':data['timestamp'],'sensors':sensors,'status':result['status']},ensure_ascii=False))


if __name__ == '__main__':
    main()
