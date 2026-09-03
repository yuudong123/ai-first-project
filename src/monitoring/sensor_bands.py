"""과거 정상 운전의 1초 평균에서 고정 표시 범위를 만든다. 고장 판정 기준은 아니다."""
import json
from functools import lru_cache
from pathlib import Path
import numpy as np
from src.runtime.common import ROOT, SENSORS


def build_sensor_bands(records, profile, calibration_ids, test_ids=()):
    records, profile = np.asarray(records), np.asarray(profile)
    ids = np.array(sorted(set(map(int,calibration_ids))),dtype=int)
    if records.shape != (len(profile),60,len(SENSORS)) or profile.shape != (len(profile),5):
        raise ValueError('사이클·센서·라벨 배열 형식이 맞지 않습니다.')
    if not len(ids) or ids.min()<1 or ids.max()>len(profile) or set(ids)&set(test_ids):
        raise ValueError('기준 사이클 ID가 잘못되었거나 평가 데이터가 포함되었습니다.')
    healthy = np.all(profile[ids-1]==[100,100,0,130,0],axis=1)
    chosen = ids[healthy]
    if len(chosen)<3:
        raise ValueError('정상 부품 + 안정 상태의 기준 사이클이 3개 미만입니다.')
    values = records[chosen-1].reshape(-1,len(SENSORS))
    if not np.isfinite(values).all():
        raise ValueError('정상 기준 데이터에 유효하지 않은 값이 있습니다.')
    lower,upper = np.quantile(values,[.01,.99],axis=0)
    return {'version':'historical-healthy-1hz-q01-q99-v1',
        'source':'UCI 원본 학습·검증 사이클 중 부품 4개 정상 및 stable_flag=0',
        'sample_period_sec':1,'lower_quantile':.01,'upper_quantile':.99,
        'cycle_count':len(chosen),'sample_count':len(values),'cycle_ids':chosen.tolist(),
        'warning_consecutive_samples':3,'auto_adapt':False,
        'note':'통계적 기준 범위이며 안전 한계나 AI 고장 판정이 아님. 기준 사이클 수가 적고 운전 단계별 범위가 아님.',
        'sensors':{sensor:{'lower':float(lower[i]),'upper':float(upper[i])} for i,sensor in enumerate(SENSORS)}}


@lru_cache(maxsize=1)
def load_sensor_bands(root=ROOT):
    root = Path(root)
    with np.load(root/'data/processed/simulator/uci_1hz_17sensors.npz',allow_pickle=False) as saved:
        if list(saved['sensor_names'])!=list(SENSORS):
            raise ValueError('기준 데이터 센서 순서 불일치')
        records = saved['data']
    profile = np.loadtxt(root/'data/raw/uci_hydraulic/extracted/profile.txt')
    splits = json.loads((root/'data/processed/split_ids_accumulator_stratified.json').read_text(encoding='utf-8'))
    return build_sensor_bands(records,profile,splits['train_ids']+splits['val_ids'],splits['test_ids'])
