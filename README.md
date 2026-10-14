# HydroTwin

> 설비 3대의 센서·AI 진단을 Web과 Unity로 연결하고, 설비별 상태와 공통 환경 변화를 구분하는 디지털 트윈 프로젝트입니다.

## 📌 프로젝트 개요

| 항목 | 내용 |
|---|---|
| 프로젝트명 | HydroTwin |
| 개발 형태·기간 | 4인 팀, 2026-08-28 ~ 2026-09-15 |
| 데이터 | UCI Condition Monitoring of Hydraulic Systems |
| 시연 입력 | V5 생성 모델 기반 설비 3대 × 센서 17개, 1초 갱신 |
| AI | LightGBM 부품·안정 상태 판별, TreeSHAP |
| 서비스 | Kafka · FastAPI · Web · Unity WebGL · Docker Compose · Jenkins |
| 개인 담당 | 조현재 — 팀장, Unity 3D·WebGL, API 규격 협의, 전체 통합, 발표·데모 |

## 🙋 개인 기여

- **팀장·통합:** 전체 일정과 모듈 의존성을 정리하고, AI·Backend·Web·Unity 사이의 데이터 규격을 협의했습니다.
- **Unity 시각화:** 설비 구조와 부품 상태 색상, 설비·센서 선택 및 상태 확인 상호작용을 담당했습니다.
- **API 연동:** Unity가 Kafka에 직접 연결하지 않고 FastAPI 최신 상태를 1초마다 조회하도록 연결 구조를 정했습니다.
- **WebGL·시연:** Unity WebGL과 웹의 통합, 전체 시연 점검과 발표를 담당했습니다.

개인 역할과 마일스톤의 근거는 [확정 팀 마일스톤](docs/team-milestones.md), 세부 작업 배정은 [팀 상세 작업 명세](docs/team-task-board.md)입니다. 모델 학습·평가, Kafka·API, CI/CD는 팀의 각 담당 영역과 구분합니다.

## 🏗️ 시스템 구조와 연동 원칙

```text
UCI 데이터 → 전처리·모델 학습 → 상태 판별 모델
V5 시뮬레이터 → Kafka → 설비별 10초 버퍼 → LightGBM·TreeSHAP
                                            ↓
                                  최신 상태 파일 → FastAPI
                                                     ├─ Web 관제
                                                     └─ Unity WebGL
```

학습·재학습·실시간 추론은 **센서별 최근 10초 평균 17개**로 입력을 맞춥니다. 세 설비는 생성 상태·추론 버퍼·드리프트 기준을 각각 유지하고 하나의 API 응답으로 전달합니다.

## 💡 해결한 설계 과제

| 과제 | 해결 방식 | 확인할 근거 |
|---|---|---|
| 세 설비의 데이터가 섞이거나 같은 값으로 보이는 문제 방지 | 고정 설비 ID, Kafka key, 설비별 독립 생성 상태·추론 버퍼 사용 | [다중 설비 시험](tests/test_v5_multi_station.py), [Consumer](kafka/consumer.py) |
| 준비되지 않은 설비를 정상처럼 표시하는 문제 방지 | 3대가 모두 준비돼야 정상 API 응답, 초기 10초 `warming_up`, 수신 중단 시 설비별 `stale` | [실행 계약](docs/local-runtime.md), [진입점 시험](tests/test_server_entry.py) |
| Unity와 데이터 처리의 결합도 증가 | Kafka 소비·추론은 Backend에 두고 Unity는 공통 상태 API만 조회 | [팀 작업 명세](docs/team-task-board.md), [웹·API 서버](web/serve.py) |
| 개별 설비 운전 전환을 공통 계절 변화로 오인 | 설비별 기준을 유지하고 최근 60초의 3대 공통 변화·안정 offset을 확인한 뒤 재학습 요청 | [모니터 코드](src/runtime/monitor.py), [운영 설명](docs/local-runtime.md) |
| 재학습 모델이 기존 성능을 훼손할 위험 | 원래 환경·계절 변화 환경별 성능과 타깃별 비회귀 조건으로 후보 승격 검사 | [승격 조건 시험](tests/test_retrain_gate.py) |

## 🧪 검증 방법과 포트폴리오 범위

- **코드로 확인 가능한 항목:** 설비 상태 독립성, 10초 입력 조건, 필수 예측 타깃, 모델 승격 거부 조건을 [tests](tests/)에서 확인할 수 있습니다.
- **실행 확인:** 아래 실행 절차와 `python -m src.runtime.check`로 현재 환경의 3대 연결 상태를 확인합니다. 구체적인 점검 조건은 [실행 가이드](docs/local-runtime.md)에 있습니다.
- **시각화 자료:** [최종 발표자료](docs/HydroTwin%20프로젝트%20최종%20발표자료%20%28수정%20완성본%29.pdf)를 제공합니다. Unity 프로젝트 원본은 로컬 관리 대상이므로 저장소만으로 Editor 작업 전체를 재현하는 데는 제약이 있습니다.
- **성과 구분:** 이번 README 편집은 저장소 코드·문서 확인 기준입니다. 테스트 전체 통과 수, 장시간 실서비스 성능, 현재 Unity 화면 동작을 새로 실측한 결과로 주장하지 않습니다.
- **해석 한계:** 생성 센서의 초기 라벨을 실제 정답으로 간주하지 않으며, 실제 설비 현장 정확도·고장 시점·잔여수명을 입증한 프로젝트로 표현하지 않습니다.

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
