"""설비별 최근 10초 센서값으로 상태와 TreeSHAP 영향 센서를 계산한다."""
import numpy as np
import pandas as pd
from src import hydrotwin_pipeline as p
from src.runtime.common import SENSORS, now

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
