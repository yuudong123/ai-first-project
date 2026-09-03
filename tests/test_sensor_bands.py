"""고정 기준 범위는 검증된 정상 원본만 사용하고 평가 사이클을 제외한다."""
import numpy as np
import pytest
from src.monitoring.sensor_bands import build_sensor_bands
from src.runtime.common import SENSORS


def fixture():
    data = np.broadcast_to(np.arange(60)[None,:,None],(5,60,17)).astype(float).copy()
    profile = np.tile([100,100,0,130,0],(5,1))
    profile[3,4] = 1
    data[3:] = 100000
    return data,profile


def test_reference_uses_only_healthy_stable_non_test_cycles():
    data,profile = fixture()
    result = build_sensor_bands(data,profile,[1,2,3,4],[5])
    assert result['cycle_ids']==[1,2,3]
    assert result['sample_count']==180 and not result['auto_adapt']
    assert list(result['sensors'])==list(SENSORS)
    expected = np.quantile(data[:3,:,0].reshape(-1),[.01,.99])
    assert list(result['sensors']['PS1'].values())==list(expected)


def test_test_overlap_and_insufficient_reference_rejected():
    data,profile = fixture()
    with pytest.raises(ValueError):
        build_sensor_bands(data,profile,[1,2,3,5],[5])
    with pytest.raises(ValueError):
        build_sensor_bands(data,profile,[1,2,4],[5])


def test_api_missing_reference_fails_without_fake_limits(monkeypatch):
    from api import main
    from fastapi.testclient import TestClient
    def missing():
        raise FileNotFoundError()
    monkeypatch.setattr(main,'load_sensor_bands',missing)
    assert TestClient(main.app).get('/api/v1/sensors/reference-bands').status_code==503


def test_api_returns_all_seventeen_limits(monkeypatch):
    from api import main
    from fastapi.testclient import TestClient
    data,profile = fixture()
    monkeypatch.setattr(main,'load_sensor_bands',lambda:build_sensor_bands(data,profile,[1,2,3,4],[5]))
    response = TestClient(main.app).get('/api/v1/sensors/reference-bands')
    assert response.status_code==200
    assert len(response.json()['sensors'])==17
