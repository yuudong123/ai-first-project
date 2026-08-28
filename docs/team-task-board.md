# 4인 팀 역할 및 작업 명세

- 기준일: 2026-08-28
- 프로젝트: HydroTwin
- 원칙: 각 담당자가 자신의 폴더와 결과물을 끝까지 책임지고, 팀장은 Unity와 전체 통합을 담당한다.

> 실명과 날짜별 확정 배정은 [확정 팀 마일스톤](team-milestones.md)을 최우선 기준으로 한다. 이 문서의 팀원 A·B·C 표기는 세부 작업을 설명하기 위한 작업 묶음이며, 실제 담당자는 각각 데이터·AI `홍유나`, API·웹 `박민`, 인프라·MLOps `신종건`이다. 팀장 `조현재`는 Unity와 전체 통합을 담당한다.

## 1. 이번 프로젝트에 실제로 사용할 강의 기술

`강의 질문 도우미`에서 확인한 학습 내용 중 프로젝트에 직접 필요한 것만 선택한다.

| 강의 기술 | 프로젝트 적용 | 이유 |
|---|---|---|
| Git·GitHub | 사용 | 브랜치, PR, 코드·문서 버전 관리 |
| Kafka Producer·Topic·Consumer Group | 사용 | 테스트 센서 데이터의 실시간 재생과 예측 전달 |
| FastAPI | 사용 | Unity·웹에 현재 설비 상태 제공 |
| RandomForest 등 전통 ML | 사용 | 2,205개 사이클과 저사양 CPU에 적합 |
| Docker·Jenkins | 사용 | 테스트, 이미지 빌드, 배포 자동화 |
| 데이터 드리프트·재학습 | 축소 구현 | 보류 데이터를 신규 배치처럼 넣어 운영 과정을 시연 |
| MLflow | 시간이 되면 사용 | 실험 지표와 모델 버전 관리 |
| MongoDB | MVP에서 제외 | 현재 규모에는 Parquet·JSONL로 충분하고 DB 운영 부담이 큼 |
| PySpark·Hadoop | MVP에서 제외 | 약 530MB 데이터에 분산 처리는 과도함 |
| TensorFlow·CNN·YOLO | 제외 | 현재 과제는 표 형식 멀티센서 시계열 분류이며 영상·음성 모델이 아님 |

기술 수를 늘리는 것이 목표가 아니다. 아래 한 흐름이 실제로 작동하는 것이 목표다.

```text
Test 센서 데이터 재생
→ Kafka
→ 20초 버퍼와 AI 추론
→ 예측 Topic
→ FastAPI
→ 웹 대시보드·Unity WebGL
```

## 2. 역할 요약

| 담당 | 주 업무 | 최종 결과물 |
|---|---|---|
| 팀장 | Unity 3D·WebGL, 시스템 계약, 통합, CI/CD | 작동하는 디지털 트윈과 전체 시연 |
| 팀원 A | 데이터 파싱·EDA·특징 데이터 생성 | 학습용 특징과 Kafka 재생용 Test 데이터 |
| 팀원 B | 모델 학습·평가·SHAP·드리프트 | 모델 파일, 성능표, 설명 결과 |
| 팀원 C | Kafka·FastAPI·웹 대시보드 | 실시간 파이프라인과 상태 API·웹 화면 |

## 3. 파일 소유권

동시에 같은 파일을 고치지 않도록 처음부터 담당 폴더를 나눈다.

```text
ai-first-project/
├─ src/
│  ├─ data/              # 팀원 A
│  ├─ features/          # 팀원 A, 공통 특징 함수
│  ├─ model/             # 팀원 B
│  ├─ streaming/         # 팀원 C
│  └─ api/               # 팀원 C
├─ web/                  # 팀원 C
├─ unity/                # 팀장만 수정
├─ infra/                # 팀장
├─ tests/
│  ├─ data/              # 팀원 A
│  ├─ model/             # 팀원 B
│  └─ streaming_api/     # 팀원 C
├─ docs/
│  ├─ data/              # 팀원 A
│  ├─ model/             # 팀원 B
│  ├─ api/               # 팀원 C
│  └─ presentation/      # 전원 작성, 팀원 C가 병합
├─ docker-compose.yml    # 팀장
└─ Jenkinsfile           # 팀장
```

Unity의 Scene과 Prefab은 병합 충돌이 크므로 팀장 외에는 직접 수정하지 않는다. 수정 요청은 이미지, 모델 파일 또는 문서로 전달한다.

## 4. 팀장 작업 명세 — Unity·통합·MLOps

### 입력받는 것

- 팀원 A: Kafka 재생용 Test 데이터와 센서 스키마
- 팀원 B: 모델 파일, 라벨 매핑, 성능·SHAP 결과
- 팀원 C: `/health`, `/api/v1/state/latest` API

### 순서대로 할 일

1. 저장소 기본 폴더, 브랜치 규칙, JSON 계약을 먼저 만든다.
2. Unity 6.3 LTS에서 빈 Cube가 WebGL로 빌드되는지 확인한다.
3. 유압 탱크, 펌프, 밸브, 냉각기, 축압기를 저폴리 오브젝트로 구성한다.
4. mock JSON을 읽어 정상·주의·경고·위험 색상을 바꾸는 기능을 만든다.
5. 펌프 누설, 밸브 지연, 냉각 효율 저하, 축압기 압력 저하 애니메이션을 만든다.
6. mock 입력을 실제 FastAPI 주소로 교체하고 1초 간격으로 상태를 조회한다.
7. WebGL 빌드를 팀원 C의 웹 화면에 삽입한다.
8. 각 서비스의 Docker 실행 방식을 `docker-compose.yml`로 묶는다.
9. Jenkins에 테스트 → Docker 빌드 → 실행 → `/health` 확인 단계를 작성한다.
10. 전체 시연, 장애 시 수동 복구, 백업 영상을 준비한다.

### 완료 기준

- 브라우저에서 Unity WebGL이 열린다.
- Kafka 재생 시 네 부품의 색상과 상태 문구가 실제 예측 결과에 따라 바뀐다.
- `docker compose up` 이후 API와 웹이 실행된다.
- Git Push 후 Jenkins 테스트와 Docker 빌드가 성공한다.
- Unity 부분은 모델이 없어도 mock JSON으로 독립 시연할 수 있다.

### 발표 담당

- 전체 아키텍처
- Unity 디지털 트윈
- Git·Docker·Jenkins·MLOps
- 기술 질의응답

## 5. 팀원 A 작업 명세 — 데이터·EDA·특징 추출

### 입력

- `data/raw/uci_hydraulic/extracted/*.txt`
- 정답 파일 `profile.txt`

### 순서대로 할 일

1. 센서 파일 17개와 `profile.txt`의 행 수가 모두 2,205개인지 검사한다.
2. 센서명, 단위, 샘플링 속도, 부품 라벨 의미를 표로 정리한다.
3. `cycle_id`를 추가하고 학습·검증·테스트 사이클 목록을 고정 저장한다.
4. 학습·검증·테스트의 `cycle_id`가 서로 겹치지 않는지 검사한다.
5. 센서별 평균, 표준편차, 최솟값, 최댓값, 범위, 기울기, RMS를 계산한다.
6. 10초, 20초, 30초, 60초 구간별 특징 파일을 만든다.
7. Test 사이클을 1초 단위로 정리한 Kafka 재생 파일을 만든다.
8. 정답은 재생 센서 파일에 넣지 않고 `test_labels.parquet`에 따로 보관한다.
9. 부품별 클래스 분포와 대표 센서 그래프를 PNG로 저장한다.
10. 재실행 방법과 컬럼 설명을 `docs/data/`에 기록한다.

### 담당 파일 예시

```text
src/data/load_raw.py
src/data/make_splits.py
src/features/extract.py
data/processed/features_10s.parquet
data/processed/features_20s.parquet
data/processed/features_30s.parquet
data/processed/features_60s.parquet
data/processed/replay_test.parquet
data/processed/test_labels.parquet
docs/data/data-card.md
reports/eda/*.png
```

`data/processed/` 결과는 Git에 올리지 않고 생성 스크립트와 문서만 올린다.

### 완료 기준

- 한 명령으로 특징 파일을 다시 만들 수 있다.
- 학습·검증·테스트 사이클 중복이 0개다.
- 생성된 데이터에 `cycle_id`가 있고 결측치가 없다.
- 10/20/30/60초 특징의 컬럼 순서가 동일하다.
- Kafka 재생 파일에는 정답 라벨이 포함되지 않는다.

### 발표 담당

- 데이터 출처·센서·라벨
- 데이터 분포와 전처리
- 데이터 누수 방지 방법

## 6. 팀원 B 작업 명세 — AI 모델·평가·설명

### 입력받는 것

- 팀원 A의 구간별 특징 파일
- 고정된 학습·검증·테스트 `cycle_id`

### 순서대로 할 일

1. 냉각기, 밸브, 펌프, 축압기 모델을 각각 만든다.
2. 가장 먼저 RandomForest 기준 모델을 학습한다.
3. 시간이 되면 LightGBM 또는 HistGradientBoosting과 비교한다.
4. Accuracy만 쓰지 않고 Macro F1과 부품별 혼동행렬을 계산한다.
5. 10초, 20초, 30초, 60초 입력의 성능을 같은 표에 정리한다.
6. Test 데이터는 모든 선택이 끝난 뒤 한 번만 최종 평가한다.
7. 모델 입력 컬럼, 라벨 매핑, 학습 시각, 성능을 `metadata.json`에 저장한다.
8. SHAP으로 부품별 상위 영향 센서와 특징을 정리한다.
9. `predict()` 함수가 네 부품의 상태와 확률을 반환하도록 만든다.
10. 고도화 단계에서 기준 데이터와 최근 데이터의 특징 분포 차이를 검사한다.

### 담당 파일 예시

```text
src/model/train.py
src/model/evaluate.py
src/model/predict.py
src/model/explain.py
src/model/drift.py
artifacts/models/*.joblib
artifacts/metrics/metrics.json
artifacts/metrics/confusion_matrix_*.png
artifacts/models/metadata.json
docs/model/model-card.md
```

`artifacts/`는 Git에 올리지 않고 학습 명령과 결과 요약만 올린다.

### 완료 기준

- 동일한 입력으로 다시 학습할 때 결과를 재현할 수 있다.
- 네 부품 모두 `predict()` 결과가 나온다.
- 10/20/30/60초 성능 비교표가 있다.
- 각 부품의 혼동행렬과 대표 오분류 사례가 있다.
- 예측 결과에 확률과 상위 영향 특징이 포함된다.
- 새 모델은 기존 모델보다 평가 기준이 나쁠 경우 배포하지 않는다.

### 발표 담당

- 모델 선택 이유
- 성능표·혼동행렬
- 조기판별 성능과 한계
- SHAP 기반 원인 후보 설명

## 7. 팀원 C 작업 명세 — Kafka·FastAPI·웹 UI

### 입력받는 것

- 팀원 A: `replay_test.parquet`과 센서 스키마
- 팀원 B: `predict()` 함수와 모델 메타데이터
- 팀장: API·Kafka JSON 계약

### 순서대로 할 일

1. 단일 Kafka 브로커를 실행하고 테스트 Topic을 만든다.
2. 숫자 5개를 보내고 출력하는 최소 Producer·Consumer부터 성공시킨다.
3. Producer가 Test 사이클을 1초 간격으로 재생하도록 바꾼다.
4. 메시지 Key는 `machine_id`로 설정해 한 설비의 순서를 유지한다.
5. `hydraulic.telemetry.v1` Consumer가 최근 20초 메시지를 메모리에 유지하게 한다.
6. 팀원 B의 `predict()`를 호출해 네 부품의 상태를 계산한다.
7. 예측 결과를 `hydraulic.prediction.v1` Topic에 보낸다.
8. FastAPI가 최신 예측을 읽고 `/health`와 `/api/v1/state/latest`로 제공하게 한다.
9. 웹에 센서 선 그래프, 부품 상태 카드, 경고 내역을 표시한다.
10. 중복 `event_id`, Kafka 중단, API 미응답 상황을 테스트한다.

### 담당 파일 예시

```text
src/streaming/producer.py
src/streaming/inference_worker.py
src/streaming/schemas.py
src/api/app.py
src/api/state_store.py
web/index.html
web/app.js
web/styles.css
docs/api/openapi-notes.md
tests/streaming_api/*.py
```

### 완료 기준

- `cycle_id` 하나를 선택하면 60개의 1초 메시지가 순서대로 전송된다.
- Consumer를 다시 실행해도 동일한 `event_id`를 중복 처리하지 않는다.
- 20초가 쌓인 뒤 첫 예측이 생성되고 이후 1초마다 갱신된다.
- `/health`가 200을 반환한다.
- `/api/v1/state/latest`가 팀장이 정한 JSON 형식을 지킨다.
- 웹에서 현재 센서, 네 부품 상태, 마지막 갱신 시각을 확인할 수 있다.

### 발표 담당

- Kafka Producer·Topic·Consumer 흐름
- FastAPI 역할
- 웹 대시보드와 장애 처리
- 전체 PPT 파일 취합과 시연 순서 정리

## 8. 팀 사이의 고정 계약

아래 형식을 먼저 확정하면 실제 코드가 완성되기 전에도 각자 mock 데이터로 개발할 수 있다.

### Kafka 센서 이벤트

Topic: `hydraulic.telemetry.v1`

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

실제 구현에서는 모든 센서가 들어가지만 구조는 변경하지 않는다. 정답 라벨은 이 메시지에 넣지 않는다.

### Unity·웹용 최신 상태 응답

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

## 9. 오늘 각자 해야 할 일

### 팀장

- Unity 설치와 WebGL Cube 빌드 테스트
- 위 JSON 두 개를 팀원에게 공유하고 필드명 확정
- Git 브랜치 네 개 생성 또는 작업 규칙 공지
- 기본 폴더와 빈 실행 파일 골격 준비

### 팀원 A

- `description.txt`, `documentation.txt`, `profile.txt` 읽기
- 부품별 라벨 분포 표 만들기
- 센서 17개 파일의 행·열 크기 확인
- 첫 번째 사이클 그래프 2개 그리기

### 팀원 B

- `profile.txt`의 네 타깃 의미 정리
- 60초 센서 평균만 사용한 가장 단순한 RandomForest 기준 모델 실행
- 네 부품 중 우선 펌프 하나의 혼동행렬 만들기

### 팀원 C

- Kafka 단일 브로커 실행
- `hydraulic.telemetry.v1` Topic 생성
- 임시 JSON Producer와 출력 Consumer 작성
- FastAPI `/health` 엔드포인트 작성

## 10. Git 작업 규칙

| 담당 | 브랜치 |
|---|---|
| 팀장 Unity | `feat/unity-digital-twin` |
| 팀원 A | `feat/data-pipeline` |
| 팀원 B | `feat/model-training` |
| 팀원 C | `feat/streaming-api` |
| 팀장 인프라 | `infra/mlops` |

공통 규칙:

1. `master`에 직접 작업하지 않는다.
2. 작업 시작 전에 `master`의 최신 변경을 받는다.
3. 한 커밋에는 한 목적만 담는다.
4. 데이터 원본, 모델 파일, Unity `Library`와 WebGL Build는 Push하지 않는다.
5. PR 설명에는 실행 명령, 실행 결과, 남은 문제를 적는다.
6. 각 담당자는 자기 코드의 최소 테스트를 함께 작성한다.
7. 다른 담당자의 폴더를 수정해야 하면 먼저 담당자와 API·파일 형식을 합의한다.

## 11. 통합 순서

```text
1차: 전원 mock 데이터로 각자 기능 실행
2차: A의 실제 데이터 → B의 실제 모델
3차: A의 Test 재생 파일 → C의 Kafka
4차: C의 실시간 입력 → B의 predict 함수
5차: C의 FastAPI → 팀장의 Unity
6차: 웹에 Unity WebGL 삽입
7차: Docker Compose와 Jenkins로 전체 검증
```

통합 전까지 서로 기다리지 않는다. 입력이 아직 없으면 이 문서의 JSON 예시를 사용하고, 모델이 없으면 고정된 mock 예측을 반환한다.
