"""원본을 한 번씩 읽어 1Hz 데이터와 10초 특징을 만들고 최초 모델을 학습한다."""
import json
import argparse
import os
import shutil
import uuid
import numpy as np
import pandas as pd
from src import hydrotwin_pipeline as p
from src.runtime.common import ROOT


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--rebuild-model',action='store_true',help='기존 모델을 백업한 뒤 10초 모델을 다시 학습')
    args = parser.parse_args()
    processed = ROOT / 'data/processed'
    processed.mkdir(parents=True, exist_ok=True)
    raw_path = processed / 'simulator/uci_1hz_17sensors.npz'
    if not raw_path.exists():
        arrays = []
        for sensor in p.SENSOR_NAMES:
            print(f'전처리: {sensor}', flush=True)
            raw = np.loadtxt(p.RAW_DIR / f'{sensor}.txt', dtype=np.float32)
            arrays.append(raw.reshape(len(raw), 60, p.SAMPLING_RATES[sensor]).mean(axis=2))
        values = np.stack(arrays, axis=2)
        raw_path.parent.mkdir(parents=True, exist_ok=True)
        np.savez_compressed(raw_path, data=values, sensor_names=p.SENSOR_NAMES)
    else:
        with np.load(raw_path) as saved:
            values = saved['data']
    profile = p.load_profile()
    splits = p.make_splits(profile)
    p.validate_splits(profile, splits)
    p.save_splits(splits)
    features = pd.DataFrame(values[:, :10, :].mean(axis=1), columns=p.MEAN_FEATURE_COLUMNS)
    features.insert(0, 'cycle_id', profile.cycle_id)
    features.to_parquet(processed / 'features_10s.parquet', index=False)
    if not p.MODEL_PATH.exists() or args.rebuild_model:
        candidate = p.MODEL_PATH.with_name(f'bootstrap_{uuid.uuid4().hex}.joblib')
        bundle, metrics = p.train_integrated_lgbm(profile, splits, final_window_sec=10, model_path=candidate)
        if p.MODEL_PATH.exists():
            shutil.copy2(p.MODEL_PATH,p.MODEL_PATH.with_name(f'bootstrap_backup_{uuid.uuid4().hex}.joblib'))
        os.replace(candidate,p.MODEL_PATH)
        print(metrics.to_string(index=False), flush=True)
    else:
        bundle = p.load_model_bundle()
        if bundle['window_sec'] != 10:
            raise RuntimeError('기존 운영 모델이 10초 모델이 아닙니다. 백업 후 명시적으로 다시 학습하세요.')
        if bundle.get('metadata',{}).get('window_sampling',{}).get('policy') != 'all_within_cycle':
            raise RuntimeError('기존 모델은 첫 10초만 학습했습니다. --rebuild-model로 전체 위치 모델을 학습하세요.')
    print('10초 모델과 1Hz 데이터 준비 완료', flush=True)


if __name__ == '__main__':
    main()
