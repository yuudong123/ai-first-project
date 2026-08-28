# 팀 상세 작업 명세

- 기준일: 2026-08-28
- 프로젝트: HydroTwin
- 일정 기준: [확정 팀 마일스톤](team-milestones.md)

## 1. 문서 사용 원칙

- `team-milestones.md`는 **누가 언제 무엇을 끝낼지** 정한 유일한 일정 원본이다.
- 이 문서는 **어떤 파일과 기능을 만들어야 완료인지** 설명한다.
- 일정이나 담당자가 달라지면 마일스톤을 먼저 수정하고 이 문서를 맞춘다.
- Unity Scene과 Prefab은 병합 충돌을 피하기 위해 조현재만 수정한다.
- 실제 입력이 아직 없으면 아래 JSON 계약을 사용한 mock 데이터로 먼저 개발한다.

## 2. 전체 시스템 흐름

```text
[오프라인 학습]
UCI 원본 TXT
→ 전처리·특징 추출
→ Train/Validation/Test 분리
→ RandomForest·LightGBM 학습·평가
→ 모델·라벨 매핑·SHAP 결과 저장

[실시간 시연]
학습에 쓰지 않은 Test 사이클
→ Kafka Producer가 1초 단위로 재생
→ hydraulic.telemetry.v1
→ 20초 버퍼·AI 추론
→ hydraulic.prediction.v1
→ FastAPI 최신 상태 API
→ 웹 대시보드 + Unity WebGL

[운영 자동화]
Git Push
→ Jenkins
→ pytest
→ Docker 이미지 빌드
→ 서비스 실행
→ /health 확인
```

Unity는 Kafka에 직접 연결하지 않는다. Kafka를 소비하는 백엔드가 최신 상태를 FastAPI로 제공하고 Unity는 1초 간격으로 API를 조회한다.

## 3. 담당자와 파일 소유권

| 담당자 | 담당 영역 | 주 소유 경로 |
|---|---|---|
| 조현재 | Unity 3D·WebGL, 인터페이스 결정, 전체 통합 | `unity/`, `docs/contracts/` |
| 홍유나 | 데이터·EDA·특징 추출, 모델·평가·SHAP | `src/data/`, `src/features/`, `src/model/`, `docs/data/`, `docs/model/` |
| 박민 | Kafka 응용 코드, FastAPI, 웹 UI | `src/streaming/`, `src/api/`, `web/`, `docs/api/` |
| 신종건 | Docker·Kafka 실행환경, pytest 통합, Jenkins·MLflow | `infra/`, `tests/integration/`, `docker-compose.yml`, `Jenkinsfile` |

공용 설정 파일을 수정할 때는 담당 브랜치에서 PR을 만들고 관련 담당자의 확인을 받는다. 데이터 원본, 생성 데이터, 모델 파일, MLflow 실행 결과와 Unity 빌드는 Git에 올리지 않는다.

## 4. 조현재 — Unity·인터페이스·최종 통합

### 입력받는 결과물

- 홍유나: 부품 상태 코드와 라벨 매핑, 주요 영향 센서 이름
- 박민: `GET /health`, `GET /api/v1/state/latest`
- 신종건: Docker Compose 실행 주소와 Jenkins 결과

### 구현 작업

1. Unity 6.3 LTS에서 Web 빌드가 가능한지 Cube로 먼저 검증한다.
2. 유압 탱크, 펌프, 밸브, 냉각기, 축압기의 저폴리 3D 구조를 만든다.
3. `normal`, `caution`, `warning`, `danger` 상태별 색상 규칙을 만든다.
4. mock JSON으로 부품 상태 변경과 부품 클릭 UI를 구현한다.
5. 펌프 누설, 밸브 지연, 냉각 효율 저하, 축압기 압력 저하를 구분해 표현한다.
6. FastAPI 최신 상태를 1초 간격으로 조회하는 Unity C# 클라이언트를 만든다.
7. 부품 클릭 시 상태, 신뢰도, 상위 영향 센서와 점검 메시지를 표시한다.
8. WebGL로 빌드하고 박민의 웹 페이지에 삽입한다.
9. 전체 파이프라인을 연결하고 시연 실패에 대비한 mock 모드를 유지한다.

### 완료 기준

- WebGL 화면이 Chrome 또는 Edge에서 열린다.
- API 응답에 따라 네 부품의 색상과 문구가 바뀐다.
- API가 중단되어도 화면이 멈추지 않고 연결 오류를 표시한다.
- 실제 API와 mock 모드를 설정 하나로 전환할 수 있다.
- 발표 PC에서 전체화면 시연과 백업 영상 재생이 가능하다.

### 발표 범위

- 전체 아키텍처와 데이터 흐름
- Unity 디지털 트윈과 WebGL 시연
- 전체 통합 과정과 프로젝트 한계

## 5. 홍유나 — 데이터·모델·SHAP

### 입력 데이터

- `data/raw/uci_hydraulic/extracted/*.txt`
- 정답 파일 `profile.txt`

### 데이터 작업

1. 센서 17개 파일과 `profile.txt`가 각각 2,205개 사이클인지 검사한다.
2. 센서명, 물리량, 단위, 샘플링 속도와 라벨 의미를 데이터 카드에 기록한다.
3. 모든 데이터에 `cycle_id`를 부여한다.
4. 학습·검증·테스트 사이클 목록을 고정하고 중복이 없는지 검사한다.
5. 평균, 표준편차, 최솟값, 최댓값, 범위, 기울기와 RMS 특징을 만든다.
6. 10초, 20초, 30초, 60초 특징 파일의 컬럼 순서를 동일하게 맞춘다.
7. Test 사이클을 1초 단위로 재생할 `replay_test.parquet`을 만든다.
8. 정답은 재생 파일에 포함하지 않고 `test_labels.parquet`에 따로 저장한다.
9. 부품별 클래스 분포와 대표 센서 그래프를 저장한다.

### 모델 작업

1. 냉각기, 밸브, 펌프, 축압기 모델을 각각 학습한다.
2. RandomForest를 기준 모델로 사용한다.
3. LightGBM과 동일한 분할·평가지표로 비교한다.
4. Accuracy, Macro F1과 부품별 혼동행렬을 계산한다.
5. 10초, 20초, 30초, 60초 조기판별 성능을 비교한다.
6. 모든 선택이 끝난 뒤 Test 데이터로 최종 성능을 한 번 평가한다.
7. 모델 입력 컬럼, 라벨 매핑, 학습 시각과 성능을 `metadata.json`에 저장한다.
8. SHAP으로 부품별 상위 영향 센서와 특징을 정리한다.
9. 네 부품의 상태·확률을 반환하는 공통 `predict()` 함수를 제공한다.

### 결과물

```text
src/data/load_raw.py
src/data/make_splits.py
src/features/extract.py
src/model/train.py
src/model/evaluate.py
src/model/predict.py
src/model/explain.py
docs/data/data-card.md
docs/model/model-card.md
reports/eda/*.png
reports/model/*.png
```

다음 생성 파일은 로컬에만 두고 Git에는 올리지 않는다.

```text
data/processed/features_10s.parquet
data/processed/features_20s.parquet
data/processed/features_30s.parquet
data/processed/features_60s.parquet
data/processed/replay_test.parquet
data/processed/test_labels.parquet
artifacts/models/*.joblib
artifacts/models/metadata.json
```

### 완료 기준

- 한 명령으로 전처리와 특징 파일을 다시 만들 수 있다.
- 학습·검증·테스트의 `cycle_id` 중복이 0개다.
- 네 부품 모두 모델 성능표와 혼동행렬이 있다.
- `predict()`가 상태 코드, 라벨과 확률을 반환한다.
- SHAP 결과가 API에 전달 가능한 JSON 형태로 정리된다.

### 발표 범위

- 데이터 구조·센서·라벨·전처리
- 모델 선택과 성능 비교
- SHAP 기반 영향 센서와 한계

## 6. 박민 — Kafka 응용·FastAPI·웹 UI

Kafka 브로커와 Docker 환경은 신종건이 담당하고, 박민은 해당 환경을 사용하는 Producer·Consumer와 API 코드를 담당한다.

### 입력받는 결과물

- 홍유나: `replay_test.parquet`, `predict()`와 모델 메타데이터
- 조현재: 아래 API 응답 계약과 Unity WebGL 빌드 경로
- 신종건: Kafka 접속 주소, Topic과 Docker Compose 실행 방법

### 구현 작업

1. mock JSON을 전송·출력하는 최소 Kafka Producer·Consumer를 만든다.
2. Producer가 Test 사이클을 1초 간격으로 재생하도록 바꾼다.
3. 메시지 Key는 `machine_id`로 지정해 한 설비의 순서를 유지한다.
4. `hydraulic.telemetry.v1` Consumer가 최근 20초 데이터를 메모리에 유지하게 한다.
5. 20초가 쌓이면 홍유나의 `predict()`를 호출하고 이후 1초마다 다시 예측한다.
6. 결과를 `hydraulic.prediction.v1` Topic으로 전송한다.
7. FastAPI가 최신 예측을 읽어 `/health`와 `/api/v1/state/latest`로 제공하게 한다.
8. Chart.js로 센서 차트, 부품 상태 카드와 마지막 갱신 시각을 표시한다.
9. 조현재의 Unity WebGL 빌드를 웹 페이지에 삽입한다.
10. Kafka·모델이 준비되기 전에도 mock 모드로 웹과 API를 실행할 수 있게 한다.

### 결과물

```text
src/streaming/replay_producer.py
src/streaming/inference_consumer.py
src/streaming/schemas.py
src/api/app.py
src/api/state_store.py
web/index.html
web/app.js
web/styles.css
docs/api/api-contract.md
```

### 완료 기준

- 사이클 하나가 60개의 1초 이벤트로 순서대로 전송된다.
- 첫 예측은 20초 이후 생성되고 이후 1초마다 갱신된다.
- `/health`가 HTTP 200을 반환한다.
- `/api/v1/state/latest`가 아래 JSON 계약을 지킨다.
- API 미응답과 데이터 대기 상태가 웹에 구분되어 표시된다.
- 웹 차트와 Unity가 동일한 `cycle_id`와 상태를 보여준다.

### 발표 범위

- Kafka Producer·Consumer와 실시간 처리 흐름
- FastAPI와 웹 대시보드
- AI·Unity 사이의 인터페이스

## 7. 신종건 — Docker·테스트·Jenkins·MLflow

### 입력받는 결과물

- 홍유나: 전처리·학습 실행 명령과 모델 평가 기준
- 박민: Kafka·FastAPI·웹 실행 명령과 `/health`
- 조현재: 최종 서비스 구성과 배포 확인 기준

### 구현 작업

1. 단일 Kafka 브로커를 KRaft 방식으로 실행하는 Compose 설정을 만든다.
2. `hydraulic.telemetry.v1`, `hydraulic.prediction.v1` Topic 준비 방법을 문서화한다.
3. FastAPI용 Dockerfile과 전체 `docker-compose.yml`을 만든다.
4. 데이터 전처리 함수와 FastAPI 응답 스키마의 pytest를 작성한다.
5. Jenkinsfile에 Checkout → Test → Docker Build → Deploy → Health Check 단계를 작성한다.
6. Jenkins가 저장소의 실제 `master` 브랜치를 사용하도록 설정한다.
7. MLflow에 파라미터, Macro F1, 모델 파일과 학습 시각을 기록한다.
8. 보류 데이터를 신규 라벨 배치처럼 투입하는 재학습 데모를 만든다.
9. 후보 모델이 기존 모델의 기준을 통과할 때만 승격하도록 검사한다.
10. 새 서비스의 Health Check가 실패하면 기존 버전을 유지하는 절차를 문서화한다.

### 결과물

```text
infra/kafka/
infra/mlflow/
tests/integration/
Dockerfile
docker-compose.yml
Jenkinsfile
docs/operations/runbook.md
```

### 완료 기준

- `docker compose up`으로 Kafka와 FastAPI가 실행된다.
- pytest가 전처리와 API 계약 오류를 잡는다.
- Jenkins에서 Test, Build와 Health Check가 성공한다.
- MLflow에서 최소 두 모델의 지표를 비교할 수 있다.
- 재학습은 라벨이 있는 데이터에서만 실행된다.
- 새 모델 성능이 기준보다 낮으면 기존 모델이 유지된다.

### 발표 범위

- Docker·Compose 실행환경
- pytest와 Jenkins CI/CD
- MLflow·드리프트·조건부 재학습

## 8. Kafka와 API 계약

### Topic

| Topic | 생산자 | 소비자 | 용도 |
|---|---|---|---|
| `hydraulic.telemetry.v1` | Test 데이터 재생 Producer | 추론 Consumer | 1초 단위 센서 이벤트 |
| `hydraulic.prediction.v1` | 추론 Consumer | FastAPI 상태 저장기 | 부품별 예측과 영향 센서 |

교육용 단일 설비 시연은 Topic당 Partition 1개로 시작한다. 데이터 흐름이 확인된 뒤에만 Partition 수를 늘린다.

### 센서 이벤트 예시

```json
{
  "schema_version": 1,
  "event_id": "cycle-1731-sec-12",
  "machine_id": "hydraulic-rig-01",
  "cycle_id": 1731,
  "elapsed_sec": 12,
  "occurred_at": "2026-08-28T14:10:12+09:00",
  "sensors": {
    "PS1": {"mean": 151.2, "std": 2.1, "min": 147.4, "max": 155.7},
    "FS1": {"mean": 7.8, "std": 0.2, "min": 7.4, "max": 8.1},
    "TS1": {"value": 38.2},
    "VS1": {"value": 0.61}
  }
}
```

실제 메시지에는 필요한 센서가 모두 들어가지만 정답 라벨은 포함하지 않는다.

### 최신 상태 API 예시

Endpoint: `GET /api/v1/state/latest`

```json
{
  "machine_id": "hydraulic-rig-01",
  "cycle_id": 1731,
  "elapsed_sec": 20,
  "observed_window_sec": 20,
  "updated_at": "2026-08-28T14:10:20+09:00",
  "components": {
    "pump": {
      "state_code": 2,
      "state_label": "severe_leakage",
      "risk_level": "danger",
      "confidence": 0.91
    },
    "valve": {
      "state_code": 100,
      "state_label": "normal",
      "risk_level": "normal",
      "confidence": 0.96
    },
    "cooler": {
      "state_code": 20,
      "state_label": "reduced_efficiency",
      "risk_level": "warning",
      "confidence": 0.88
    },
    "accumulator": {
      "state_code": 115,
      "state_label": "slightly_reduced",
      "risk_level": "caution",
      "confidence": 0.82
    }
  },
  "top_factors": [
    {"feature": "PS1_mean", "impact": 0.31},
    {"feature": "FS1_std", "impact": 0.19}
  ]
}
```

## 9. Git 브랜치

| 담당 | 브랜치 |
|---|---|
| 조현재 Unity | `feat/unity-digital-twin` |
| 홍유나 데이터·AI | `feat/data-model` |
| 박민 스트리밍·API·웹 | `feat/streaming-api-web` |
| 신종건 인프라·MLOps | `infra/mlops` |

공통 규칙:

1. `master`에 직접 작업하지 않는다.
2. 작업 시작 전에 최신 `master`를 반영한다.
3. 한 커밋에는 한 목적만 담는다.
4. PR에는 실행 명령, 결과 화면과 남은 문제를 기록한다.
5. 담당자는 자기 영역의 최소 테스트와 발표용 캡처를 함께 만든다.

## 10. 통합 순서

```text
1. 조현재·박민: API JSON 계약 확정
2. 각자 mock 데이터로 Unity·API·Kafka·모델 독립 실행
3. 홍유나 특징 데이터 → 실제 모델 연결
4. 홍유나 Test 재생 파일 → 박민 Producer 연결
5. 박민 Kafka 추론 → FastAPI 연결
6. FastAPI → 조현재 Unity 연결
7. 박민 웹에 Unity WebGL 삽입
8. 신종건 Docker Compose·Jenkins·MLflow 연결
9. 전체 시연 고정 후 기능 변경 중단
```

## 11. MVP에서 제외할 기술

- MongoDB: 현재 규모에는 Parquet·JSONL과 메모리 상태 저장으로 충분하다.
- PySpark·Hadoop: 약 530MB 데이터에 분산 처리 환경은 과도하다.
- TensorFlow·CNN·YOLO: 현재 데이터와 모델 목적에 맞지 않는다.
- Unity에서 Kafka 직접 연결: WebGL 호환성과 보안 문제 때문에 FastAPI를 경유한다.
- 실제 RUL 예측: 이 데이터는 run-to-failure 데이터가 아니다.
