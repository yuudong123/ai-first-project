"""사이클 경계를 넘지 않는 모든 10초 평균 특징을 구성한다."""
import numpy as np
import pandas as pd


def build_rolling_features(values, feature_names, window_sec=10):
    values = np.asarray(values)
    if values.ndim != 3 or values.shape[1:] != (60, len(feature_names)):
        raise ValueError('입력은 (사이클 수, 60초, 센서 수) 배열이어야 합니다.')
    if window_sec != 10 or not np.isfinite(values).all():
        raise ValueError('유한한 센서값과 10초 window가 필요합니다.')
    starts = 60-window_sec+1
    means = np.stack([values[:,s:s+window_sec].mean(axis=1) for s in range(starts)],axis=1)
    result = pd.DataFrame(means.reshape(-1,len(feature_names)),columns=feature_names)
    result.insert(0,'window_start_sec',np.tile(np.arange(starts),len(values)))
    result.insert(0,'cycle_id',np.repeat(np.arange(1,len(values)+1),starts))
    return result
