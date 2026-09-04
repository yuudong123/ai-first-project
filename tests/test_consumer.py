"""설비별 버퍼, 중복/지연 처리 및 웹 API의 설비 배열 검사."""
from datetime import datetime,timezone,timedelta
import json
import pytest
import importlib.util
from pathlib import Path

CONSUMER_PATH=Path(__file__).resolve().parents[1]/'kafka/consumer.py'
SPEC=importlib.util.spec_from_file_location('hydrotwin_consumer',CONSUMER_PATH)
CONSUMER=importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(CONSUMER)
EquipmentInference,EQUIPMENT_IDS=CONSUMER.EquipmentInference,CONSUMER.EQUIPMENT_IDS
from src.runtime.common import SENSORS


def message(equipment,second,value):
    stamp=datetime(2026,9,3,tzinfo=timezone.utc)+timedelta(seconds=second)
    return {'equipment_id':equipment,'timestamp':stamp.isoformat(),'sensors':dict.fromkeys(SENSORS,value)},stamp


def engine():
    return EquipmentInference(lambda rows:{'status':'ready','observed_window_sec':10,'components':{},
        'mean':sum(r[0] for r in rows)/len(rows)},'test-run')


def test_interleaved_equipment_uses_independent_ten_sample_windows():
    runtime=engine()
    for second in range(10):
        for i,equipment in enumerate(EQUIPMENT_IDS):
            payload,stamp=message(equipment,second,(i+1)*100+second)
            result=runtime.update(payload,stamp)
            assert result['equipment_states'][i]['prediction']['status']==('ready' if second==9 else 'warming_up')
    assert [s['prediction']['mean'] for s in result['equipment_states']]==[104.5,204.5,304.5]
    assert [s['event_id'] for s in result['equipment_states']]==[10,10,10]
    assert result['event_id']==30


def test_duplicate_and_out_of_order_do_not_increase_count():
    runtime=engine();payload,stamp=message('station-01',5,1)
    runtime.update(payload,stamp)
    assert runtime.update(payload,stamp) is None
    older,old_stamp=message('station-01',4,2)
    assert runtime.update(older,old_stamp) is None
    assert len(runtime.buffers['station-01'])==1


def test_gap_resets_only_affected_equipment():
    runtime=engine()
    for equipment in EQUIPMENT_IDS:
        for second in range(10):
            payload,stamp=message(equipment,second,1);runtime.update(payload,stamp)
    payload,stamp=message('station-02',12,2);runtime.update(payload,stamp)
    assert [len(runtime.buffers[e]) for e in EQUIPMENT_IDS]==[10,1,10]
    assert runtime.states['station-02']['segment_id']==1


def test_unknown_missing_or_stale_data_is_rejected():
    runtime=engine();payload,stamp=message('station-01',1,1)
    with pytest.raises(ValueError):runtime.update({**payload,'equipment_id':'unknown'},stamp)
    with pytest.raises(ValueError):runtime.update({**payload,'sensors':{}},stamp)
    with pytest.raises(ValueError):runtime.update(payload,stamp+timedelta(seconds=6))
    assert not runtime.states


def test_api_preserves_ids_and_marks_only_stale_station(tmp_path,monkeypatch):
    from api import main
    from fastapi.testclient import TestClient
    current=datetime.now(timezone.utc)
    runtime=engine()
    for i,equipment in enumerate(EQUIPMENT_IDS):
        stamp=current-timedelta(seconds=20 if i==1 else 0)
        payload={'equipment_id':equipment,'timestamp':stamp.isoformat(),'sensors':dict.fromkeys(SENSORS,i+1)}
        result=runtime.update(payload,stamp)
    path=tmp_path/'latest.json';path.write_text(json.dumps(result),encoding='utf-8')
    monkeypatch.setattr(main,'LATEST_RAW_PATH',path)
    data=TestClient(main.app).get('/api/v1/state/latest').json()
    assert [s['equipment_id'] for s in data['equipment_states']]==list(EQUIPMENT_IDS)
    assert [s['sensors']['PS1'] for s in data['equipment_states']]==[1,2,3]
    assert [s['prediction']['status'] for s in data['equipment_states']]==['warming_up','stale','warming_up']
    assert data['monitoring']['retraining']['status']=='idle'


@pytest.mark.parametrize('new_run,new_event,new_segment',[('a',11,1),('b',1,0),('a',12,0)])
def test_source_transition_restart_or_event_gap_resets_only_its_equipment(new_run,new_event,new_segment):
    runtime=engine()
    for second in range(10):
        for equipment in EQUIPMENT_IDS:
            payload,stamp=message(equipment,second,10)
            runtime.update({**payload,'run_id':'a','event_id':second+1,'segment_id':0},stamp)
    payload,stamp=message('station-02',10,100)
    runtime.update({**payload,'run_id':new_run,'event_id':new_event,'segment_id':new_segment},stamp)
    assert [len(runtime.buffers[e]) for e in EQUIPMENT_IDS]==[10,1,10]
    assert runtime.states['station-02']['prediction']['status']=='warming_up'
    for second in range(11,20):
        payload,stamp=message('station-02',second,100)
        runtime.update({**payload,'run_id':new_run,'event_id':new_event+second-10,'segment_id':new_segment},stamp)
    assert runtime.states['station-02']['prediction']['mean']==100


def test_source_duplicate_is_ignored_even_with_new_timestamp():
    runtime=engine()
    for second in range(2):
        payload,stamp=message('station-01',second,1)
        runtime.update({**payload,'run_id':'a','event_id':1,'segment_id':0},stamp)
    assert len(runtime.buffers['station-01'])==1


def test_source_run_id_is_preserved_for_retraining_contract():
    runtime=engine()
    payload,stamp=message('station-01',0,1)
    result=runtime.update({**payload,'run_id':'producer-run','event_id':1,'segment_id':0},stamp)
    assert result['equipment_states'][0]['run_id']=='producer-run'
