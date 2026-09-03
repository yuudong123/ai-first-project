"""Kafka 센서값을 10초 평균으로 추론하고 TreeSHAP 영향 센서를 계산한다."""
from collections import deque
import numpy as np
import pandas as pd
from src import hydrotwin_pipeline as p
from src.runtime.common import SENSORS, consumer, now, write_state, age
from src.runtime.seed_schedule import window_discontinuity

NORMAL = {'pump':0,'cooler':100,'valve':100,'accumulator':130}


def diagnose(bundle, rows):
    features = pd.DataFrame([np.mean(rows,axis=0)], columns=p.MEAN_FEATURE_COLUMNS)
    result = p.predict(features, model_bundle=bundle)
    components = {}
    for component,value in result['components'].items():
        model = bundle['models'][component]
        probabilities = model.predict_proba(features)[0]
        class_index = list(model.classes_).index(value)
        # LightGBM의 pred_contrib는 TreeSHAP이다. 예측 클래스의 기여도만 사용한다.
        contributions = np.asarray(model.booster_.predict(features,pred_contrib=True)).reshape(-1,len(SENSORS)+1)
        impact = contributions[class_index if len(contributions)>1 else 0,:-1]
        indices = np.argsort(np.abs(impact))[::-1][:3]
        components[component] = {'state':int(value),'prediction':int(value),
            'raw_value':int(value),'state_label':'정상' if value==NORMAL[component] else '이상',
            'risk_level':'normal' if value==NORMAL[component] else 'warning',
            'confidence':float(probabilities[class_index]),
            'top_sensors':[{'sensor':SENSORS[i],'impact':float(impact[i])} for i in indices]}
    return {'status':'ready','observed_window_sec':10,'stable_flag':result['stable_flag'],
            'is_stable':result['stable_flag']==0,'components':components,
            'explanation_method':'TreeSHAP (모델 판단 근거이며 실제 고장 원인 확정 아님)',
            'predicted_at':now()}


def main():
    bundle = p.load_model_bundle()
    if bundle['window_sec'] != 10:
        raise RuntimeError('10초 모델이 필요합니다.')
    stamp = p.MODEL_PATH.stat().st_mtime_ns
    rows = deque(maxlen=10)
    previous_run = None
    previous_event = None
    previous_segment = None
    prediction = {'status':'warming_up','observed_window_sec':0,'components':{}}
    for message in consumer('hydrotwin-inference-v1'):
        data = message.value
        if age(data['timestamp']) > 5:
            continue
        if window_discontinuity(previous_run,previous_event,previous_segment,data):
            rows.clear()
            prediction = {'status':'warming_up','observed_window_sec':0,'components':{}}
        previous_run,previous_event = data['run_id'],data['event_id']
        previous_segment = data.get('segment_id',0)
        new_stamp = p.MODEL_PATH.stat().st_mtime_ns
        if stamp != new_stamp:
            candidate = p.load_model_bundle()
            if candidate['window_sec'] != 10:
                raise RuntimeError('교체된 모델의 입력 시간이 10초가 아닙니다.')
            bundle,stamp = candidate,new_stamp
        rows.append([data['sensors'][s] for s in SENSORS])
        if len(rows)==10:
            prediction = diagnose(bundle,list(rows))
        else:
            prediction['observed_window_sec'] = len(rows)
        write_state('latest.json', {'event_id':data['event_id'],'run_id':data['run_id'],
            'segment_id':data.get('segment_id',0),
            'cycle_id':(data['event_id']-1)//60+1,'elapsed_sec':data['event_id'],
            'updated_at':now(),'generated_at':data['timestamp'],'received_at':now(),
            'sensors':data['sensors'],'prediction':prediction,'model_version':str(stamp)})


if __name__ == '__main__':
    main()
