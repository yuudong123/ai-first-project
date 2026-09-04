# HydroTwin

UCI 유압 시스템 데이터와 V5 시계열 생성 모델을 이용해 **설비 3대**의 센서값을 생성하고,
각 설비의 최근 10초 평균으로 부품 상태와 안정 여부를 판별하는 디지털 트윈 프로젝트입니다.

## 고정 실행 계약

- 설비: `station-01`, `station-02`, `station-03` 세 대
- 센서: 설비마다 17개, 1초마다 한 번 갱신
- Kafka Topic: `hydraulic.sensor.multi.raw`
- AI 입력: 설비별로 분리한 최근 10초 센서 평균 17개
- AI 출력: 냉각기, 밸브, 펌프, 축압기, `stable_flag`
- 화면: 하나의 API 응답에 `equipment_states` 세 개를 담아 웹과 Unity에 전달
- MLOps: 설비별 드리프트를 따로 감지하고 세 대가 함께 확인한 계절 변화로 공통 모델을 재학습

단일 설비 실행 모드와 단일 설비 데이터 대체 기능은 지원하지 않습니다. 세 대 중 하나라도
준비되지 않으면 API가 준비 대기 상태를 반환합니다.

## 데이터 흐름

```text
kafka/producer.py
  → 설비 3대의 서로 다른 센서값 + 공통 계절 변화
  → Kafka hydraulic.sensor.multi.raw
  ├─ kafka/consumer.py
  │    → 설비별 독립 10초 버퍼 → LightGBM/TreeSHAP
  │    → artifacts/runtime/latest.json
  │    → FastAPI → 웹 UI + Unity WebGL
  └─ src/runtime/monitor.py
       → 설비별 기준 통계와 드리프트 감지
       → 3대 공통 확인 → Jenkins → 재학습·성능 검증·모델 교체
```

## 실행

Docker Desktop을 실행한 뒤 프로젝트 루트에서:

```powershell
Set-ExecutionPolicy -Scope Process -ExecutionPolicy Bypass
.\start.ps1
```

- 웹·Unity: `http://localhost:8000`
- API 상태: `http://localhost:8000/health`
- 최신 3설비 상태: `http://localhost:8000/api/v1/state/latest`
- Jenkins: `http://localhost:8080`

확인:

```powershell
docker compose ps
docker exec hydrotwin-monitor python -m src.runtime.check
```

중지할 때는 볼륨을 보존합니다.

```powershell
docker compose stop
```

`docker compose down -v`는 Kafka와 Jenkins 볼륨을 삭제하므로 일반 중지에 사용하지 않습니다.

## 필수 로컬 자산

Git에서 제외되는 다음 자산이 필요합니다.

- `data/raw/uci_hydraulic/extracted/`
- `data/processed/simulator/uci_1hz_17sensors.npz`
- `models/predict/integrated_lgbm.joblib`
- Unity WebGL 빌드의 `Build/pro-build.*`

설정과 자세한 검증 방법은 [Docker 실행 가이드](docs/local-runtime.md)를 참고하세요.

## 프로젝트 범위

이 데이터는 실제 고장 시점까지의 run-to-failure 데이터가 아닙니다. 따라서 잔여수명 예측이
아니라 최근 센서 구간에 대한 부품 상태 판별을 목표로 합니다. TreeSHAP 결과는 모델 판단에
영향을 준 센서이며 물리적인 고장 원인을 확정하는 증거는 아닙니다.

원본 데이터: [UCI Condition Monitoring of Hydraulic Systems](https://archive.ics.uci.edu/dataset/447/condition+monitoring+of+hydraulic+systems), CC BY 4.0
