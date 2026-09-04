"""사이클 분리, 10초 전체 위치 학습, 생성 시나리오 경계 회귀 검사."""
import numpy as np
import pandas as pd
import pytest
from src.features.rolling import build_rolling_features
from src import hydrotwin_pipeline as p
from src.runtime.seed_schedule import SeedSchedule, window_discontinuity


def samples():
    values = np.broadcast_to(np.arange(60)[None,:,None], (2,60,17)).copy()
    values[1] += 1000
    return build_rolling_features(values,p.MEAN_FEATURE_COLUMNS)


def test_all_positions_without_crossing_cycle_boundary():
    features = samples()
    assert len(features) == 102
    assert features.groupby('cycle_id').size().tolist() == [51,51]
    assert features.iloc[[0,50,51,101]][p.MEAN_FEATURE_COLUMNS[0]].tolist() == [4.5,54.5,1004.5,1054.5]
    assert features.window_start_sec.tolist() == list(range(51))*2


def test_cycle_selection_and_labels_never_enter_features():
    profile = pd.DataFrame({'cycle_id':[1,2],'stable_flag':[0,1]})
    X_train,y_train = p.get_xy(samples(),profile,[1],'stable_flag')
    X_test,y_test = p.get_xy(samples(),profile,[2],'stable_flag')
    assert X_train.shape == X_test.shape == (51,17)
    assert list(X_train.columns) == p.MEAN_FEATURE_COLUMNS
    assert set(y_train) == {0} and set(y_test) == {1}
    assert X_train.to_numpy().max() < X_test.to_numpy().min()


def test_duplicate_window_is_rejected():
    features = samples()
    with pytest.raises(ValueError,match='중복'):
        p.get_xy(pd.concat([features,features.iloc[:1]]),
                 pd.DataFrame({'cycle_id':[1,2],'stable_flag':[0,1]}),[1],'stable_flag')


def test_invalid_array_is_rejected():
    with pytest.raises(ValueError):
        build_rolling_features(np.zeros((2,59,17)),p.MEAN_FEATURE_COLUMNS)


def test_schedule_reference_and_two_to_one_seed_ratio():
    profile = np.array([[100,100,0,130,0],[20,80,2,90,1],[3,73,1,100,1]])
    schedule = SeedSchedule(profile,seed=7)
    assert schedule.select(0) == (0,0,True)
    assert schedule.select(119) == (0,0,True)
    assert schedule.select(120) == (0,0,True)
    assert schedule.select(180) == (0,0,True)
    seed,segment,reference = schedule.select(240)
    assert profile[seed,4] == 1 and segment == 1 and not reference
    assert schedule.select(299) == (seed,segment,reference)
    assert schedule.select(300) == (0,2,True)
    flags = [profile[schedule.select(t)[0],4] for t in range(480,480+180*10,60)]
    assert flags.count(0) == 20 and flags.count(1) == 10


@pytest.mark.parametrize('change', [{'segment_id':1},{'run_id':'new'},{'event_id':12}])
def test_window_resets_at_segment_run_or_message_gap(change):
    data = {'run_id':'run','event_id':11,'segment_id':0}
    assert not window_discontinuity('run',10,0,data)
    assert window_discontinuity('run',10,0,{**data,**change})


def test_legacy_message_and_first_message():
    data = {'run_id':'run','event_id':11}
    assert not window_discontinuity('run',10,0,data)
    assert window_discontinuity(None,None,None,data)


def test_monitor_excludes_changed_operating_context(tmp_path,monkeypatch):
    from types import SimpleNamespace
    from src.runtime import monitor
    from src.runtime.common import now
    states,history,updates,resets = {},[],[],[]
    class Detector:
        def __init__(self,reference,config):
            self.reference = reference
        def reset(self):
            resets.append(True)
        def update(self,sensors,timestamp):
            updates.append(sensors)
            return {'sensor_scores':{},'status':'warming_up','drift_detected':False}
    def save(name,value):
        states[name] = value
        if name=='monitor.json':
            history.append(value.copy())
    messages = []
    for second in range(1,244):
        excluded = 182 <= second <= 241
        for equipment in monitor.EQUIPMENT_IDS:
            messages.append(SimpleNamespace(value={'equipment_id':equipment,
                'event_id':second,'run_id':'test','timestamp':now(),
                'segment_id':1 if excluded else (2 if second>=242 else 0),
                'reference_context':not excluded,'sensors':dict.fromkeys(monitor.SEASONAL,10.)}))
    monkeypatch.setattr(monitor,'STATE_DIR',tmp_path)
    monkeypatch.setattr(monitor,'consumer',lambda group:iter(messages))
    monkeypatch.setattr(monitor,'read_state',lambda name:states.get(name,{}))
    monkeypatch.setattr(monitor,'write_state',save)
    monkeypatch.setattr(monitor,'RollingDriftDetector',Detector)
    monkeypatch.setattr(monitor,'trigger',lambda:pytest.fail('제외 구간은 새 학습을 요청하면 안 됨'))
    monitor.main()
    assert len(updates)==63*3
    excluded_states = [s for s in history if s['status']=='condition_excluded']
    assert excluded_states
    assert all(not s['drift_detected'] for s in excluded_states)
    assert set(history[-1]['equipment_ids'])==set(monitor.EQUIPMENT_IDS)
    assert resets and 'retrain_request.json' not in states


def test_monitor_requests_retraining_only_after_all_three_confirm(tmp_path,monkeypatch):
    from types import SimpleNamespace
    from src.runtime import monitor
    from src.runtime.common import now
    files={}
    class Detector:
        def __init__(self,reference,config): self.reference=reference
        def reset(self): pass
        def update(self,sensors,timestamp):
            scores={sensor:{'mean_offset':1.0,'affected':True} for sensor in monitor.SEASONAL}
            return {'status':'drift','drift_detected':True,'candidate_detected':True,
                    'sensor_scores':scores,'affected_sensors':list(monitor.SEASONAL)}
    messages=[]
    for second in range(1,241):
        for equipment in monitor.EQUIPMENT_IDS:
            messages.append(SimpleNamespace(value={'equipment_id':equipment,'event_id':second,
                'run_id':'three-run','timestamp':now(),'segment_id':0,'reference_context':True,
                'sensors':dict.fromkeys(monitor.SEASONAL,10.)}))
    monkeypatch.setattr(monitor,'STATE_DIR',tmp_path)
    monkeypatch.setattr(monitor,'consumer',lambda group:iter(messages))
    monkeypatch.setattr(monitor,'read_state',lambda name:files.get(name,{}))
    monkeypatch.setattr(monitor,'write_state',lambda name,value:files.__setitem__(name,value))
    monkeypatch.setattr(monitor,'RollingDriftDetector',Detector)
    monkeypatch.setattr(monitor,'trigger',lambda:17)
    monitor.main()
    request=files['retrain_request.json']
    assert request['run_id']=='three-run'
    assert set(request['equipment_ids'])==set(monitor.EQUIPMENT_IDS)
    assert request['confirmation']=='all_three_equipment'
    assert files['retraining.json']['jenkins_queue']==17
