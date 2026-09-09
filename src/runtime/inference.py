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
        # 실시간 화면에서는 Random Forest의 전역 특징 중요도로 영향 센서를 요약한다.
        impact = np.asarray(model.feature_importances_,dtype=float)
        indices = np.argsort(np.abs(impact))[::-1][:3]
        components[component] = {'state':int(value),'prediction':int(value),
            'raw_value':int(value),'state_label':'정상' if value==NORMAL[component] else '이상',
            'risk_level':'normal' if value==NORMAL[component] else 'warning',
            'confidence':float(probabilities[class_index]),
            'top_sensors':[{'sensor':SENSORS[i],'impact':float(impact[i])} for i in indices]}
    return {'status':'ready','observed_window_sec':10,'stable_flag':result['stable_flag'],
            'is_stable':result['stable_flag']==0,'components':components,
            'explanation_method':'Random Forest feature importance (모델 영향 지표이며 실제 고장 원인 확정 아님)',
            'predicted_at':now()}
