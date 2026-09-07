"""공동 검증 사이클에서 생성값과 초기 라벨의 일치도를 검사한다. 실제 정확도 검증은 아니다."""
import json
import hashlib
import numpy as np
import joblib
import keras
import os
os.environ["KERAS_BACKEND"] = "jax"
from sklearn.metrics import accuracy_score, recall_score, f1_score
from src import hydrotwin_pipeline as p
from src.features.rolling import build_rolling_features
from src.simulator.v5_generation_utils import generate_from_seed_batch
from src.runtime.common import ROOT, write_state, now


def main():
    model_dir = ROOT/'models/simulator'
    metadata = json.loads((model_dir/'generator_metadata_v5.json').read_text(encoding='utf-8'))
    profile = p.load_profile()
    # 생성기 학습 제외 사이클과 분류기 평가 사이클의 교집합만 사용한다.
    ids = sorted(i for i in p.make_splits(profile)['test_ids'] if i > metadata['training_records'])
    if not ids:
        raise RuntimeError('공동 검증 사이클이 없습니다.')
    selected = profile.set_index('cycle_id').loc[ids]
    with np.load(p.PROCESSED_DIR/'simulator/uci_1hz_17sensors.npz') as data:
        records = data['data'][np.array(ids)-1]
    generator = keras.models.load_model(model_dir/'virtual_factory_generator_v5.keras',compile=False)
    with np.load(model_dir/'sensor_bounds_v5.npz') as bounds:
        generated = generate_from_seed_batch(generator,
            joblib.load(model_dir/'input_scaler_v5.joblib'),
            joblib.load(model_dir/'offset_scaler_v5.joblib'),records[:,:30],60,
            bounds['sensor_min'],bounds['sensor_max'],[p.SENSOR_NAMES.index('PS4')])
    bundle = p.load_model_bundle()
    features = build_rolling_features(generated,p.MEAN_FEATURE_COLUMNS)[p.MEAN_FEATURE_COLUMNS]
    results = {}
    for target in p.TARGET_ORDER:
        seed_labels = np.repeat(selected[target].to_numpy(),51)
        predicted = bundle['models'][target].predict(features)
        results[target] = {
            'seed_label_agreement':float(accuracy_score(seed_labels,predicted)),
            'seed_label_macro_f1':float(f1_score(seed_labels,predicted,average='macro',zero_division=0)),
        }
        if target=='stable_flag':
            results[target]['seed_unstable_recall'] = float(recall_score(seed_labels,predicted,zero_division=0))
            results[target]['predicted_unstable_window_count'] = int((predicted==1).sum())
    report = {'checked_at':now(),'scope':'생성값 실제 정답이 아닌 초기 사이클 라벨과의 일치도',
        'limitations':['생성기 검증 데이터는 모델 선택에 사용되어 완전 독립 테스트가 아님',
                       '사이클 라벨은 개별 10초의 실제 안정 상태를 보장하지 않음',
                       '불안정 초기 사이클 수가 매우 적고 겹치는 window는 독립 표본이 아님',
                       '60초 무드리프트 생성 검사이며 장기 생성/계절 변화 성능을 보장하지 않음'],
        'seed_cycle_ids':[int(i) for i in ids],'cycle_count':len(ids),
        'seed_stable_counts':{str(k):int(v) for k,v in selected.stable_flag.value_counts().items()},
        'model_sha256':hashlib.sha256(p.MODEL_PATH.read_bytes()).hexdigest(),
        'metrics':results}
    write_state('generated_seed_validation.json',report)
    print(json.dumps(report,ensure_ascii=False,indent=2),flush=True)


if __name__=='__main__':
    main()
