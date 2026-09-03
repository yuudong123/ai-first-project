"""랜덤 계절 변화와 10초 입력 계약 회귀 검사."""
import inspect
import numpy as np
import pandas as pd
import pytest
from src.runtime.scenario import ScenarioConfig, RandomSeason
from src import hydrotwin_pipeline as p


def test_random_season_stays_bounded_and_intervals_are_valid():
    season = RandomSeason(ScenarioConfig(), seed=7)
    starts = []
    last = 0
    for second in range(15000):
        temperature,pressure = season.update(second)
        assert -4 <= temperature <= 4
        assert -10 <= pressure <= 10
        if season.event_id != last:
            starts.append(second)
            last = season.event_id
    assert starts[0]==120
    assert all(60 <= b-a <= 1200 for a,b in zip(starts,starts[1:]))
    assert len(set(b-a for a,b in zip(starts,starts[1:])))>1


def test_season_requires_safe_interval():
    with pytest.raises(ValueError):
        ScenarioConfig(minimum_interval=59)


def test_offset_keeps_original_features_and_other_sensors():
    original = pd.DataFrame({'TS1_mean':[40.],'PS1_mean':[100.],'VS1_mean':[2.]})
    changed = p.apply_sensor_offsets(original,{'TS1':4.,'PS1':-10.})
    assert changed.iloc[0].tolist()==[44.,90.,2.]
    assert original.iloc[0].tolist()==[40.,100.,2.]


def test_training_defaults_to_ten_seconds():
    assert inspect.signature(p.train_integrated_lgbm).parameters['final_window_sec'].default == 10


def test_atomic_promotion_keeps_backup(tmp_path):
    from src.model.retrain import promote_candidate
    production = tmp_path/'production.joblib'
    candidate = tmp_path/'candidate.joblib'
    production.write_bytes(b'previous-model')
    candidate.write_bytes(b'new-model')
    backup = promote_candidate(candidate,production)
    assert production.read_bytes()==b'new-model'
    assert backup.read_bytes()==b'previous-model'
    assert candidate.read_bytes()==b'new-model'


def test_native_tree_shap_and_prediction_contract():
    from src.runtime.inference import diagnose
    class Booster:
        def predict(self, features, pred_contrib=False):
            assert pred_contrib
            return np.arange(18,dtype=float).reshape(1,18)
    class Model:
        def __init__(self,value):
            self.value = value
            self.classes_ = np.array([value])
            self.booster_ = Booster()
        def predict(self, features):
            assert list(features.columns)==p.MEAN_FEATURE_COLUMNS
            return np.array([self.value])
        def predict_proba(self, features):
            return np.array([[1.0]])
    normal = {'cooler':100,'valve':100,'pump':0,'accumulator':130,'stable_flag':0}
    bundle = {'models':{k:Model(v) for k,v in normal.items()},'feature_names':p.MEAN_FEATURE_COLUMNS,'component_order':p.COMPONENT_ORDER}
    result = diagnose(bundle,np.ones((10,17)))
    assert result['is_stable']
    assert all(c['risk_level']=='normal' for c in result['components'].values())
    assert result['components']['pump']['top_sensors'][0]['sensor']=='SE'


def test_api_preserves_monitoring_and_unity_fields(tmp_path,monkeypatch):
    import json
    from fastapi.testclient import TestClient
    from api import main
    from src.runtime.common import now
    path = tmp_path/'latest.json'
    monkeypatch.setattr(main,'LATEST_RAW_PATH',path)
    client = TestClient(main.app)
    assert client.get('/api/v1/state/latest').status_code==503
    path.write_text(json.dumps({'event_id':1,'cycle_id':1,'elapsed_sec':10,'updated_at':now(),
        'sensors':dict.fromkeys(p.SENSOR_NAMES,1.),'model_version':'v-test',
        'prediction':{'status':'ready','observed_window_sec':10,'stable_flag':0,'components':{}}}))
    result = client.get('/api/v1/state/latest').json()
    assert result['prediction']['stable_flag']==0
    assert result['model_version']=='v-test'


def test_previous_run_retraining_is_not_shown_as_current(tmp_path,monkeypatch):
    import json
    from fastapi.testclient import TestClient
    from api import main
    from src.runtime.common import now
    path = tmp_path/'latest.json'
    path.write_text(json.dumps({'event_id':1,'cycle_id':1,'elapsed_sec':1,'run_id':'new-run',
        'updated_at':now(),'sensors':{},'prediction':{'status':'warming_up','observed_window_sec':1,'components':{}}}))
    monkeypatch.setattr(main,'LATEST_RAW_PATH',path)
    monkeypatch.setenv('HYDROTWIN_RUNTIME','1')
    files = {'retraining.json':{'status':'rejected'},'retrain_request.json':{'run_id':'old-run'}}
    monkeypatch.setattr(main,'read_state',lambda name: files.get(name,{}))
    result = TestClient(main.app).get('/api/v1/state/latest').json()
    assert result['monitoring']['retraining']['status']=='idle'
    assert files['retraining.json']['status']=='rejected'


def test_unity_gzip_has_correct_headers(tmp_path):
    import gzip
    from fastapi import FastAPI
    from fastapi.testclient import TestClient
    from api.main import UnityStaticFiles
    content = b'\x00asm-example'
    (tmp_path/'example.wasm.gz').write_bytes(gzip.compress(content))
    app = FastAPI()
    app.mount('/Build',UnityStaticFiles(directory=tmp_path))
    response = TestClient(app).get('/Build/example.wasm.gz')
    assert response.status_code==200
    assert response.headers['content-encoding']=='gzip'
    assert response.headers['content-type']=='application/wasm'
    assert response.content==content
