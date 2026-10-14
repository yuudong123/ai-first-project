# HydroTwin

> 유압 설비 3대의 센서값과 부품 상태를 웹과 Unity에서 확인할 수 있도록 만든 프로젝트입니다.

## 📌 프로젝트 개요

| 항목 | 내용 |
|---|---|
| 프로젝트명 | HydroTwin |
| 개발 형태·기간 | 4인 팀, 2026-08-28 ~ 2026-09-15 |
| 데이터 | UCI Condition Monitoring of Hydraulic Systems |
| 시연 입력 | V5 생성 모델 기반 설비 3대 × 센서 17개, 1초 갱신 |
| AI | LightGBM 부품·안정 상태 판별, 예측에 영향을 준 센서 분석(TreeSHAP) |
| 서비스 | Kafka · FastAPI · Web · Unity WebGL · Docker Compose · Jenkins |
| 개인 담당 | 조현재 — 팀장, Unity 3D·WebGL, API 규격 협의, 전체 통합, 발표·데모 |

## 담당한 작업

- 팀장을 맡아 일정을 관리하고, 모델·백엔드·웹·Unity를 연결하는 데 필요한 데이터 형식을 팀원들과 정했습니다.
- Unity에서 설비를 구성하고 부품 상태에 따라 색상이 바뀌도록 만들었습니다. 설비와 센서를 선택해 상태를 확인하는 기능도 담당했습니다.
- Unity에서 FastAPI를 1초마다 호출해 최신 센서값과 예측 결과를 받아오도록 연결했습니다.
- Unity를 WebGL로 빌드해 웹에 연결하고, 전체 시연 점검과 발표를 맡았습니다.

모델 학습·평가, Kafka·API, CI/CD는 각 팀원이 맡았습니다. 전체 역할 분담은 [팀 일정](docs/team-milestones.md)과 [상세 작업 목록](docs/team-task-board.md)에 정리했습니다.

## 시스템 구조

```text
UCI 데이터 → 전처리·모델 학습 → 상태 판별 모델
V5 시뮬레이터 → Kafka → 설비별 10초 버퍼 → LightGBM·TreeSHAP
                                            ↓
                                  최신 상태 파일 → FastAPI
                                                     ├─ Web 관제
                                                     └─ Unity WebGL
```

학습·재학습·실시간 추론은 **센서별 최근 10초 평균 17개**로 입력을 맞춥니다. 세 설비는 센서 생성 상태·예측용 데이터·센서 변화의 비교 기준을 각각 유지하고 하나의 API 응답으로 전달합니다.

## 구현하면서 고려한 점

| 고려한 점 | 구현 방법 | 관련 문서·코드 |
|---|---|---|
| 설비마다 서로 다른 센서값과 상태 유지 | 설비 ID를 Kafka 메시지의 구분값으로 쓰고, 센서 생성 상태와 예측용 데이터를 설비별로 따로 관리 | [다중 설비 시험](tests/test_v5_multi_station.py), [Consumer](kafka/consumer.py) |
| 데이터가 부족하거나 끊겼을 때의 화면 표시 | 3대가 모두 준비돼야 정상 API 응답, 초기 10초는 데이터 수집 중(`warming_up`), 수신이 끊긴 설비는 갱신 중단(`stale`)으로 표시 | [실행 계약](docs/local-runtime.md), [진입점 시험](tests/test_server_entry.py) |
| Unity에서 센서 처리까지 맡지 않도록 구성 | Kafka 소비·추론은 Backend에 두고 Unity는 공통 상태 API만 조회 | [팀 작업 명세](docs/team-task-board.md), [웹·API 서버](web/serve.py) |
| 한 설비의 상태 변화를 계절 변화로 오인하지 않도록 처리 | 설비별 기준을 유지하고 최근 60초의 3대에서 공통으로 나타나는 변화와 변화량이 안정됐는지 확인한 뒤 재학습 요청 | [모니터 코드](src/runtime/monitor.py), [운영 설명](docs/local-runtime.md) |
| 재학습 후 성능이 떨어지는 모델의 적용 방지 | 기존 환경과 계절 변화 환경에서 부품별 성능을 비교해 교체 여부 결정 | [승격 조건 시험](tests/test_retrain_gate.py) |

## 테스트와 참고 자료

- [테스트 코드](tests/)에는 설비별 상태가 섞이지 않는지, 10초 입력 조건을 지키는지, 필요한 예측값이 모두 있는지, 성능이 떨어진 모델의 교체를 막는지 확인하는 항목이 있습니다.
- 실행 후 `python -m src.runtime.check`로 설비 3대의 연결 상태를 확인할 수 있습니다. 자세한 방법은 [실행 가이드](docs/local-runtime.md)를 참고하세요.
- **발표 자료:** [최종 발표자료](docs/HydroTwin%20프로젝트%20최종%20발표자료%20%28수정%20완성본%29.pdf)를 제공합니다. Unity 프로젝트 원본은 로컬에서 관리하며 Git에는 포함하지 않았습니다.
- **데이터 한계:** 시뮬레이터가 시작할 때 사용한 라벨은 생성된 센서값의 정답으로 쓰지 않습니다. 실제 설비에서의 정확도나 고장 시점·잔여수명은 평가하지 않았습니다.

---


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

`dev` 브랜치의 GitHub push는 Jenkins의 `/github-webhook/`으로 전달한다. Jenkins는 새 커밋을
별도 작업공간에서 받아 후보 이미지를 만들고 테스트한 뒤, 성공한 경우에만 개발 서버의
컨테이너 네 개를 교체한다. 웹훅 전달 실패에 대비해 같은 저장소를 1분마다 확인하는 Poll SCM도
함께 사용한다. 설정 방법은 [Docker 실행 가이드](docs/local-runtime.md)의
`GitHub 웹훅 자동 배포`를 참고한다.

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
