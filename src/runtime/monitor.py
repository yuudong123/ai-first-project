"""설비 3대의 계절성 드리프트를 각각 감지하고 공통 재학습을 요청한다."""
import json
import logging
import time
import uuid
from logging.handlers import RotatingFileHandler

import numpy as np

from src.monitoring.drift_detector import DriftConfig, DriftReference, RollingDriftDetector, fit_reference
from src.runtime.common import STATE_DIR, age, consumer, now, read_state, write_state
from src.runtime.jenkins import trigger
from src.runtime.seed_schedule import window_discontinuity

EQUIPMENT_IDS = ('station-01','station-02','station-03')
SEASONAL = ('PS1','PS2','PS3','PS4','PS5','PS6','TS1','TS2','TS3','TS4')


def new_equipment_state():
    return {'baseline':[],'detector':None,'previous_event':None,'previous_segment':None,
            'count':0,'previous_offsets':{},'checkpoint_event':None,
            'checkpoint_plateau':False,'result':{'status':'calibrating','drift_detected':False,
            'candidate_detected':False,'sensor_scores':{},'affected_sensors':[]}}


def reference_means(states):
    return {sensor:float(np.mean([
        states[equipment]['detector'].reference.sensors[sensor].mean for equipment in EQUIPMENT_IDS
    ])) for sensor in SEASONAL}


def materially_changed(first,second,means):
    if not second:
        return True
    return any(abs(first[sensor]-second.get(sensor,0)) > (
        0.35 if sensor.startswith('TS') else max(abs(means[sensor])*0.015,0.01)
    ) for sensor in first)


def aggregate_offsets(states):
    result = {}
    for sensor in SEASONAL:
        values = []
        for equipment in EQUIPMENT_IDS:
            score = states[equipment]['result'].get('sensor_scores',{}).get(sensor)
            if score and 'mean_offset' in score:
                values.append(float(score['mean_offset']))
        if len(values)==len(EQUIPMENT_IDS):
            result[sensor] = float(np.median(values))
    return result


def public_equipment_states(states):
    return [{
        'equipment_id':equipment,
        **states[equipment]['result'],
        'baseline_samples':len(states[equipment]['baseline']),
    } for equipment in EQUIPMENT_IDS]


def save_references(run_id,states):
    if not all(states[equipment]['detector'] is not None for equipment in EQUIPMENT_IDS):
        return
    means = reference_means(states)
    write_state('reference.json',{
        'run_id':run_id,
        'equipment_ids':list(EQUIPMENT_IDS),
        'reference':{'sensors':{sensor:{'mean':value} for sensor,value in means.items()}},
        'equipment_references':{
            equipment:states[equipment]['detector'].reference.to_dict() for equipment in EQUIPMENT_IDS
        },
    })


def main():
    STATE_DIR.mkdir(parents=True,exist_ok=True)
    log = logging.getLogger('telemetry')
    log.setLevel(logging.INFO)
    if not log.handlers:
        log.addHandler(RotatingFileHandler(
            STATE_DIR/'observations.jsonl',maxBytes=10_000_000,backupCount=3,encoding='utf-8'))
    run_id = None
    states = {equipment:new_equipment_state() for equipment in EQUIPMENT_IDS}
    last_requested = {}
    processed_checkpoint = None
    last_dispatch = 0

    for message in consumer('hydrotwin-drift-v3'):
        data = message.value
        if age(data['timestamp'])>5:
            continue
        equipment = data.get('equipment_id')
        if equipment not in states:
            continue
        if data.get('run_id') != run_id:
            run_id = data['run_id']
            states = {key:new_equipment_state() for key in EQUIPMENT_IDS}
            last_requested = {}
            processed_checkpoint = None
            saved = read_state('reference.json')
            saved_references = saved.get('equipment_references',{})
            if saved.get('run_id')==run_id and set(saved_references)==set(EQUIPMENT_IDS):
                for key in EQUIPMENT_IDS:
                    states[key]['detector'] = RollingDriftDetector(
                        DriftReference.from_dict(saved_references[key]),DriftConfig())
                last_requested = read_state('monitor.json').get('last_requested_offsets',{})

        state = states[equipment]
        if window_discontinuity(run_id,state['previous_event'],state['previous_segment'],data) and state['detector']:
            state['detector'].reset()
            state['count'],state['previous_offsets'] = 0,{}
        state['previous_event'] = data['event_id']
        state['previous_segment'] = data.get('segment_id',0)
        sensors = {sensor:data['sensors'][sensor] for sensor in SEASONAL}

        if not data.get('reference_context',True):
            if state['detector']:
                state['detector'].reset()
            state['count'],state['previous_offsets'] = 0,{}
            state['checkpoint_event'] = None
            state['result'] = {'status':'condition_excluded','drift_detected':False,
                'candidate_detected':False,'sensor_scores':{},'affected_sensors':[],
                'message':'다른 운전 조건 구간은 계절 offset 추정과 재학습 요청에서 제외'}
        else:
            if state['detector'] is None:
                if data['event_id']<=120:
                    state['baseline'].append(sensors)
                elif len(state['baseline'])>=60:
                    reference = fit_reference(state['baseline'],SEASONAL,min_samples=60)
                    state['detector'] = RollingDriftDetector(reference,DriftConfig())
                    save_references(run_id,states)
                else:
                    state['result'] = {'status':'needs_baseline','drift_detected':False,
                        'candidate_detected':False,'sensor_scores':{},'affected_sensors':[],
                        'message':'설비별 정상 기준 데이터가 부족합니다. 3대 생성기를 재시작하세요.'}
            if state['detector'] is None:
                state['result'] = {'status':'calibrating','drift_detected':False,
                    'candidate_detected':False,'sensor_scores':{},'affected_sensors':[],
                    'observed_samples':len(state['baseline']),'required_samples':120}
            else:
                state['result'] = state['detector'].update(sensors,data['timestamp'])
                state['count'] += 1
                offsets = {sensor:float(value['mean_offset'])
                    for sensor,value in state['result'].get('sensor_scores',{}).items()}
                if state['count']%60==0 and offsets:
                    means = {sensor:state['detector'].reference.sensors[sensor].mean for sensor in SEASONAL}
                    state['checkpoint_plateau'] = bool(state['previous_offsets']) and not materially_changed(
                        offsets,state['previous_offsets'],means)
                    state['previous_offsets'] = offsets
                    state['checkpoint_event'] = data['event_id']

        current_request = read_state('retraining.json')
        busy = current_request.get('status') in ('queued','running')
        if busy and age(current_request['updated_at'])>1200:
            write_state('retraining.json',{**current_request,'status':'failed',
                'message':'재학습 작업 시간 초과','updated_at':now()})
            busy = False

        checkpoints = [states[key]['checkpoint_event'] for key in EQUIPMENT_IDS]
        if checkpoints[0] is not None and len(set(checkpoints))==1 and checkpoints[0]!=processed_checkpoint:
            processed_checkpoint = checkpoints[0]
            means = reference_means(states)
            offsets = aggregate_offsets(states)
            confirmed = all(states[key]['result'].get('drift_detected') and
                            states[key]['checkpoint_plateau'] for key in EQUIPMENT_IDS)
            if confirmed and offsets and not busy and materially_changed(offsets,last_requested,means):
                selected = {sensor:value for sensor,value in offsets.items() if abs(value) > (
                    0.4 if sensor.startswith('TS') else max(abs(means[sensor])*0.01,0.01))}
                if selected:
                    request = {'request_id':uuid.uuid4().hex,'run_id':run_id,'status':'drift',
                        'equipment_ids':list(EQUIPMENT_IDS),'confirmation':'all_three_equipment',
                        'sensor_offsets':selected,'created_at':now()}
                    write_state('retrain_request.json',request)
                    write_state('retraining.json',{'request_id':request['request_id'],'status':'queued',
                        'updated_at':now(),'sensor_offsets':selected,'equipment_ids':list(EQUIPMENT_IDS)})
                    last_requested = offsets.copy()

        current_request = read_state('retraining.json')
        if current_request.get('status')=='queued' and not current_request.get('jenkins_queue') and time.monotonic()-last_dispatch>15:
            last_dispatch = time.monotonic()
            try:
                queue = trigger()
                latest = read_state('retraining.json')
                if latest.get('status')=='queued':
                    write_state('retraining.json',{**latest,'jenkins_queue':queue,'updated_at':now()})
            except Exception as error:
                print(f'Jenkins 요청 재시도 대기: {type(error).__name__}',flush=True)

        equipment_results = public_equipment_states(states)
        offsets = aggregate_offsets(states)
        statuses = {item['status'] for item in equipment_results}
        overall = ('condition_excluded' if 'condition_excluded' in statuses else
                   'calibrating' if statuses & {'calibrating','needs_baseline','warming_up'} else
                   'drift' if all(item.get('drift_detected') for item in equipment_results) else
                   'suspected' if any(item.get('candidate_detected') for item in equipment_results) else 'stable')
        write_state('monitor.json',{
            'run_id':run_id,'status':overall,'updated_at':now(),
            'equipment_ids':list(EQUIPMENT_IDS),'equipment_states':equipment_results,
            'drift_detected':overall=='drift','estimated_offsets':offsets,
            'last_requested_offsets':last_requested,
            'confirmation_rule':'동일 시점에 설비 3대가 모두 드리프트와 안정된 offset을 확인해야 재학습 요청',
        })
        log.info(json.dumps({'timestamp':data['timestamp'],'equipment_id':equipment,
            'sensors':sensors,'status':state['result']['status']},ensure_ascii=False))


if __name__ == '__main__':
    main()
