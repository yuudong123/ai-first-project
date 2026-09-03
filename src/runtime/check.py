"""Jenkins 및 수동 실행용 실시간 연결 점검."""
import time
import requests


def main():
    first = requests.get('http://api:8000/api/v1/state/latest',timeout=5)
    first.raise_for_status()
    time.sleep(3)
    response = requests.get('http://api:8000/api/v1/state/latest',timeout=5)
    response.raise_for_status()
    data = response.json()
    assert data['event_id'] != first.json()['event_id'], '1초 스트림이 정지했습니다.'
    # 초기 시계열 전환 직후에는 새 구간의 10개 값이 모일 때까지 기다린다.
    deadline = time.monotonic()+15
    while data['prediction']['status']=='warming_up' and time.monotonic()<deadline:
        time.sleep(1)
        response = requests.get('http://api:8000/api/v1/state/latest',timeout=5)
        response.raise_for_status()
        data = response.json()
    assert len(data['sensors'])==17, '센서 개수 불일치'
    assert data['prediction']['status']=='ready', '10초 추론이 아직 준비되지 않았습니다.'
    assert len(data['prediction']['components'])==4, '부품 예측 누락'
    assert all(c['top_sensors'] for c in data['prediction']['components'].values()), '설명 누락'
    print('Kafka → 10초 추론 → TreeSHAP → 웹 API 연결 검사 통과',flush=True)


if __name__ == '__main__':
    main()
