"""Jenkins 및 수동 실행용 실시간 연결 점검."""
import time
import json
from urllib.request import urlopen


def fetch():
    with urlopen('http://localhost:8000/api/v1/state/latest',timeout=5) as response:
        return json.load(response)


def main():
    first = fetch()
    time.sleep(3)
    data = fetch()
    assert data['event_id'] != first['event_id'], '1초 스트림이 정지했습니다.'
    # 초기 시계열 전환 직후에는 새 구간의 10개 값이 모일 때까지 기다린다.
    deadline = time.monotonic()+15
    while any(s['prediction']['status']=='warming_up' for s in data.get('equipment_states',[])) and time.monotonic()<deadline:
        time.sleep(1)
        data = fetch()
    states = data.get('equipment_states',[])
    assert {s['equipment_id'] for s in states}=={'station-01','station-02','station-03'}, '설비 3대 누락'
    for state in states:
        assert len(state['sensors'])==17, '센서 개수 불일치'
        assert state['prediction']['status']=='ready', '설비별 10초 추론이 아직 준비되지 않았거나 지연됐습니다.'
        assert len(state['prediction']['components'])==4, '부품 예측 누락'
        assert all(c['top_sensors'] for c in state['prediction']['components'].values()), '설명 누락'
    print('Kafka → 10초 추론 → TreeSHAP → 웹 API 연결 검사 통과',flush=True)


if __name__ == '__main__':
    main()
