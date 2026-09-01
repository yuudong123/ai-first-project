# HydroTwin

멀티센서 데이터를 이용해 유압 시스템의 부품 상태를 조기에 판별하고, 이상 부품과 판단 근거를 3D 디지털 트윈에서 보여주는 프로젝트입니다.

## 정확한 프로젝트 범위

- 예측 대상: 냉각기, 밸브, 펌프, 축압기의 상태 및 고장 심각도
- 입력 데이터: 압력, 유량, 온도, 진동, 모터 전력 등 17개 센서 계열
- 서비스 출력: 부품별 상태·위험도, 주요 영향 센서, 대응 권고
- 시각화: 웹 대시보드 + Unity WebGL 유압 시스템 디지털 트윈
- 운영 자동화: Git, 테스트, Docker, Jenkins, MLflow 기반 재학습·배포 데모

> 이 데이터는 실제 고장 시점까지의 run-to-failure 데이터가 아닙니다. 따라서 잔여수명(RUL) 예측이 아니라, 60초 운전 사이클의 앞부분을 이용한 **부품 상태 조기판별**을 목표로 합니다.

## 문서

- [프로젝트 기획 및 조사](docs/project-research.md)
- [확정 팀 마일스톤](docs/team-milestones.md)
- [팀 상세 작업 명세](docs/team-task-board.md)

## 데이터

- 원본: [UCI Condition Monitoring of Hydraulic Systems](https://archive.ics.uci.edu/dataset/447/condition+monitoring+of+hydraulic+systems)
- 라이선스: CC BY 4.0

====================================================================================================
# AI First Project — 유압 시스템 모니터링 API

UCI 유압 시스템 상태 모니터링 데이터셋을 기반으로, 센서값을 학습된 모델(AI)에 넣어
pump / valve / cooler / accumulator 상태를 예측하는 FastAPI 서버입니다.

## 프로젝트 구조

```
ai-first-project/
├── data/raw/uci_hydraulic/extracted/   # 센서 원본 데이터 + profile.txt(라벨)
├── models/
│   ├── train.py                         # 모델 학습 스크립트
│   └── model.pkl                        # 학습된 모델 (train.py 실행 후 생성)
├── api/
│   └── main.py                          # FastAPI 앱 (health, data 엔드포인트)
├── docs/                                 # 프로젝트 문서
└── unity/                                # (Unity 연동 관련)
```

## 사전 준비물

- Python 3.10 이상 (Microsoft Store용 python.exe가 아닌 python.org 정식 설치본이어야 함)
- 확인: `python --version` 또는 `& "C:\Users\<사용자명>\AppData\Local\Python\bin\python.exe" --version`

## 실행 방법

### 1. 프로젝트 루트로 이동

```powershell
cd C:\ai-first-project
```

### 2. 가상환경 생성 (최초 1회만)

```powershell
python -m venv .venv
```

### 3. 가상환경 활성화

```powershell
.\.venv\Scripts\Activate.ps1
```

실행 정책 에러가 나면 아래 명령어 실행 후 다시 활성화:

```powershell
Set-ExecutionPolicy -Scope Process -ExecutionPolicy Bypass
```

활성화되면 프롬프트 앞에 `(.venv)`가 붙습니다.

```
(.venv) PS C:\ai-first-project>
```

### 4. 필요한 패키지 설치 (최초 1회만)

```powershell
pip install fastapi uvicorn[standard] pandas scikit-learn joblib
```

### 5. 모델 학습

```powershell
python models/train.py
```

`profile.txt` 라벨 기준으로 cooler / valve / pump / accumulator 4개 모델을 학습하고
`models/model.pkl`을 생성합니다. 약 15~20초 소요됩니다.

**참고 정확도** (테스트셋 기준)

| 컴포넌트 | 정확도 |
|---|---|
| cooler | 100% |
| valve | 98.2% |
| pump | 99.8% |
| accumulator | 97.1% |

### 6. 서버 실행

```powershell
uvicorn api.main:app --reload --port 8000
```

17개 센서 파일(사이클당 최대 6000개 값)을 메모리에 로드하고 모델을 불러오기 때문에
서버 기동에 15~20초 정도 걸립니다. 아래 로그가 뜨면 정상 기동된 것입니다.

```
모델 로드 중...
INFO:     Application startup complete.
INFO:     Uvicorn running on http://127.0.0.1:8000
```

## API 엔드포인트

### `GET /health` — 서버 상태 확인

```json
{
  "status": "ok",
  "uptime_sec": 30.1,
  "total_cycles_loaded": 2205,
  "model_loaded": true,
  "timestamp": "2026-08-28T07:44:08.085160+00:00"
}
```

### `GET /data` — 실시간 센서값 + AI 예측 결과

호출할 때마다 다음 사이클로 순환하며, 그 사이클의 센서값을 모델에 넣어 예측한 결과를 반환합니다.

```json
{
  "cycle_id": 1,
  "elapsed_sec": 0,
  "updated_at": "2026-08-28T07:44:26.500648+00:00",
  "sensors": {
    "PS1": 160.673, "PS2": 109.467, "PS3": 1.991, "PS4": 0,
    "PS5": 9.842, "PS6": 9.728, "EPS1": 2538.929,
    "FS1": 6.71, "FS2": 10.305,
    "TS1": 35.622, "TS2": 40.979, "TS3": 38.471, "TS4": 31.745,
    "VS1": 0.577, "CE": 39.601, "CP": 1.863, "SE": 59.157
  },
  "prediction": {
    "status": "ready",
    "observed_window_sec": 20,
    "components": {
      "cooler": { "raw_value": 3, "state_label": "close_to_total_failure", "risk_level": "danger", "confidence": 0.74 },
      "valve": { "raw_value": 100, "state_label": "optimal", "risk_level": "normal", "confidence": 0.955 },
      "pump": { "raw_value": 0, "state_label": "no_leakage", "risk_level": "normal", "confidence": 0.975 },
      "accumulator": { "raw_value": 130, "state_label": "optimal_pressure", "risk_level": "normal", "confidence": 0.84 }
    }
  }
}
```

## 자주 발생하는 문제

### `python --version` 쳤을 때 `Python`만 뜨고 버전이 안 나옴

Microsoft Store용 가짜 python.exe가 PATH 우선순위에 잡힌 경우입니다.

```powershell
where.exe python
```

경로에 `WindowsApps`가 포함되어 있으면 이 문제입니다. python.org에서 정식 설치본을 받아
"Add python.exe to PATH" 체크 후 설치하거나, 진짜 python 경로를 직접 지정해서 venv를 만듭니다.

```powershell
& "C:\Users\<사용자명>\AppData\Local\Python\bin\python.exe" -m venv .venv
```

### `.venv\Scripts\Activate.ps1`가 인식되지 않음

PowerShell에서는 앞에 `.\`를 붙여야 합니다.

```powershell
.\.venv\Scripts\Activate.ps1
```

### 스크립트 실행이 차단된다는 에러

PowerShell 보안 정책 때문입니다. 현재 세션만 풀어주면 됩니다.

```powershell
Set-ExecutionPolicy -Scope Process -ExecutionPolicy Bypass
```

### `/data` 응답에서 numpy 타입 직렬화 오류(`Unable to serialize unknown type`)가 날 경우

모델 예측값(`numpy.int64` 등)을 응답에 그대로 넣으면 발생합니다. `api/main.py`에서
`int(pred_class)`처럼 파이썬 기본 타입으로 변환해서 반환해야 합니다 (이미 반영되어 있음).

## 다음 단계 (선택)

- 프론트엔드 대시보드 연결 (실시간 그래프, 상태 표시등)
- `/data` 호출 주기를 실제 60초 사이클에 맞춰 백그라운드 스트리밍으로 전환
- 모델 재학습 자동화 (새 데이터 반영 시 `train.py` 스케줄링)

==================================================================================================
# AI First Project — 유압 시스템 실시간 모니터링

Kafka로 들어오는 유압 시스템 센서값을 실시간으로 받아 웹 대시보드와 Unity 3D 디지털
트윈에 동시에 반영하는 프로젝트입니다.

## 아키텍처

```
V5 Virtual Factory (실제 센서 데이터, 1초 간격)
        │  Kafka topic: hydraulic.sensor.raw
        ▼
kafka/consumer_v5.py  (Raw Consumer)
        │  kafka/latest_raw.json
        ▼
api/main.py  (FastAPI, GET /api/v1/state/latest)
        │  1초 폴링
        ▼
web/index.html  ──────────────┐
   (센서 텔레메트리, AI 진단,   │  같은 JSON을
    Chart.js 추이 차트)         │  SendMessage로 전달
        │                       ▼
        └──────────────▶ Unity WebGL (3D 디지털 트윈)
```

로컬 테스트용으로 Kafka 브로커·센서 데이터를 자체 시뮬레이션하는 별도 경로
(`kafka/producer.py` + `kafka/consumer.py` + Docker)도 있습니다. 실제 V5 연동과
로컬 시뮬레이션 중 선택해서 쓸 수 있습니다 (아래 "실행 방법" 참고).

## 프로젝트 구조

```
ai-first-project/
├── data/raw/uci_hydraulic/extracted/   # 센서 원본 데이터 + profile.txt(라벨) — 모델 학습용
├── models/
│   ├── train.py                         # 모델 학습 스크립트
│   └── model.pkl                        # 학습된 모델 (train.py 실행 후 생성, git 제외)
├── api/
│   └── main.py                          # FastAPI 앱 — /health, /api/v1/state/latest
├── kafka/
│   ├── consumer_v5.py                   # [실제 연동] V5 브로커 구독 → latest_raw.json 저장
│   ├── producer.py                      # [로컬 테스트] 정적 데이터로 가짜 스트림 생성
│   ├── consumer.py                      # [로컬 테스트] AI 예측까지 포함한 컨슈머
│   ├── docker-compose.yml               # [로컬 테스트용] Kafka 브로커
│   ├── diagnose.py                      # 브로커 연결 진단 스크립트
│   ├── test_interval.py                 # 메시지 수신 간격(1초 여부) 측정 스크립트
│   └── latest_raw.json                  # Consumer가 저장하는 최신 데이터 (git 제외, 자동 생성)
├── web/
│   ├── index.html                       # 대시보드 (센서 텔레메트리 / AI 진단 / Unity 3D / Chart.js)
│   ├── chart.umd.js                     # Chart.js 로컬 번들 (CDN 차단 환경 대응)
│   ├── serve.py                         # gzip 헤더 처리하는 로컬 정적 서버 (Unity WebGL용)
│   └── Build/                           # Unity WebGL 빌드 산출물 (git 제외 — 아래 안내 참고)
├── test_kafka_consumer.py               # 최소 재현 Kafka 연결 진단 스크립트
├── docs/                                 # 프로젝트 문서
└── unity/                                # Unity 원본 프로젝트
```

## 사전 준비물

- Python 3.10 이상 (Microsoft Store용 python.exe가 아닌 python.org 정식 설치본이어야 함)
  확인: `python --version` (버전 번호가 안 나오면 `where.exe python`으로 원인 확인, 아래 문제 해결 참고)
- Docker Desktop (로컬 테스트 모드에서만 필요. 실제 V5 브로커에 붙는 경우엔 불필요)

## 실행 방법

### 1. 가상환경 준비 (최초 1회)

```powershell
cd C:\ai-first-project
python -m venv .venv
.\.venv\Scripts\Activate.ps1
```

### 2. 패키지 설치 (최초 1회)

```powershell
pip install fastapi uvicorn[standard] pandas scikit-learn joblib kafka-python
```

### 3-A. 실제 V5 데이터로 연동하는 경우

```powershell
python kafka/consumer_v5.py
```

기본 브로커 `192.168.133.108:9092`, 토픽 `hydraulic.sensor.raw`를 구독합니다.
다른 브로커/토픽을 쓰려면 환경변수로 덮어쓸 수 있습니다.

```powershell
$env:KAFKA_BROKER = "다른주소:9092"
$env:KAFKA_TOPIC = "다른토픽"
python kafka/consumer_v5.py
```

### 3-B. 로컬 시뮬레이션으로 테스트하는 경우

```powershell
docker compose -f kafka/docker-compose.yml up -d
python kafka/producer.py      # 터미널 1
python kafka/consumer.py      # 터미널 2 (모델 예측까지 포함된 버전)
```

### 4. API 서버 실행

```powershell
uvicorn api.main:app --reload --port 8000
```

### 5. 웹 대시보드 실행

Unity WebGL은 `.wasm`/`.data` 파일을 `fetch()`로 불러오기 때문에 `index.html`을
더블클릭(`file://`)해서 열면 안 되고, 반드시 로컬 서버로 열어야 합니다.

```powershell
cd web
python serve.py
```

브라우저에서 `http://localhost:5500/index.html` 접속. (기본 `http.server` 대신
`serve.py`를 쓰는 이유는 아래 "Unity 빌드 파일" 항목 참고.)

## Unity 빌드 파일 안내 (중요)

`web/Build/` 폴더(Unity WebGL 산출물 — `pro-build.loader.js`, `pro-build.data.gz`,
`pro-build.framework.js.gz`, `pro-build.wasm.gz`)는 용량이 크고(총 18MB+) 재생성
가능한 산출물이라 **git에 포함되어 있지 않습니다** (`.gitignore` 처리됨).

이 브랜치를 새로 받은 사람은 웹 대시보드를 열어도 Unity 3D 화면이 안 뜨는 게
정상입니다. 아래 방법 중 하나로 빌드 파일을 받아서 `web/Build/` 폴더에 넣어야 합니다.

- [ ] TODO: Unity 빌드 파일 공유 링크(구글 드라이브 등)를 여기에 추가
- 또는 Unity 담당자에게 직접 요청해서 `web/Build/` 폴더 통째로 전달받기
- 또는 `unity/` 원본 프로젝트를 Unity 에디터로 열어서 WebGL로 직접 빌드
  (빌드 설정에서 Compression Format을 **Gzip**으로 맞춰야 아래 `serve.py` 설정과 호환됩니다)

Unity 없이 웹 대시보드(센서 텔레메트리, AI 진단, Chart.js 차트)만 확인하고 싶다면
`web/Build/` 폴더가 없어도 나머지 화면은 정상 작동합니다 — Unity 패널에만
"Unity 빌드 파일을 못 찾음" 메시지가 뜨고 나머지는 그대로 동작합니다.

## API 엔드포인트

### `GET /health` — 서버 상태 확인

```json
{
  "status": "ok",
  "uptime_sec": 30.1,
  "consumer_connected": true,
  "last_data_age_sec": 0.8,
  "timestamp": "2026-09-01T01:19:00.697984+00:00"
}
```

`consumer_connected`가 `false`면 Consumer(`consumer_v5.py` 또는 `consumer.py`)가
꺼져있거나 5초 이상 새 데이터가 안 들어온 상태입니다.

### `GET /api/v1/state/latest` — 최신 센서값 + 예측 상태

웹/Unity가 공통으로 쓰는 계약(contract)입니다. `event_id`는 메시지마다 증가하며,
웹/Unity는 이 값이 바뀌었을 때만 화면(특히 차트)을 갱신해서 중복 반영을 막습니다.

```json
{
  "event_id": 53,
  "cycle_id": 1,
  "elapsed_sec": 1040,
  "updated_at": "2026-09-01T01:19:00.324515+00:00",
  "generated_at": "2026-08-31T09:00:02Z",
  "received_at": "2026-09-01T01:19:00.324515+00:00",
  "sensors": {
    "PS1": 160.1, "PS2": 108.5, "PS3": 1.9, "PS4": 0.0,
    "PS5": 9.5, "PS6": 9.4, "EPS1": 2500.3,
    "FS1": 6.7, "FS2": 10.1,
    "TS1": 36.2, "TS2": 41.5, "TS3": 38.9, "TS4": 32.1,
    "VS1": 0.58, "CE": 40.2, "CP": 1.9, "SE": 59.5
  },
  "prediction": {
    "status": "warming_up",
    "observed_window_sec": 0,
    "components": {}
  }
}
```

`prediction.status`가 `"warming_up"`이면 아직 AI 모델이 안 붙은 원본 센서값
단계라는 뜻입니다 (`components`가 비어있음). 모델이 연동되면 `status: "ready"`와
함께 `components`에 pump/valve/cooler/accumulator 각각의 상태가 채워집니다.

## 모델 학습 (선택 — AI 예측을 붙이고 싶을 때)

```powershell
python models/train.py
```

`profile.txt` 라벨 기준으로 cooler / valve / pump / accumulator 4개 모델을 학습하고
`models/model.pkl`을 생성합니다. 약 15~20초 소요됩니다.

**참고 정확도** (테스트셋 기준)

| 컴포넌트 | 정확도 |
|---|---|
| cooler | 100% |
| valve | 98.2% |
| pump | 99.8% |
| accumulator | 97.1% |

## 자주 발생하는 문제

### `python --version` 쳤을 때 `Python`만 뜨고 버전이 안 나옴

Microsoft Store용 가짜 python.exe가 PATH 우선순위에 잡힌 경우입니다.

```powershell
where.exe python
```

경로에 `WindowsApps`가 포함되어 있으면 이 문제입니다. python.org에서 정식 설치본을 받아
"Add python.exe to PATH" 체크 후 설치하거나, 진짜 python 경로를 직접 지정해서 venv를 만듭니다.

```powershell
& "C:\Users\<사용자명>\AppData\Local\Python\bin\python.exe" -m venv .venv
```

`python3` 명령어도 Store용 가짜로 연결될 수 있습니다 — Windows에서는 항상 `python`을
쓰세요 (venv 활성화 상태라면 `python`이 venv 안의 진짜 파이썬을 정확히 가리킵니다).

### `.venv\Scripts\Activate.ps1`가 인식되지 않음 / 스크립트 실행이 차단됨

```powershell
Set-ExecutionPolicy -Scope Process -ExecutionPolicy Bypass
.\.venv\Scripts\Activate.ps1
```

### Kafka: TCP 연결(`Test-NetConnection`)은 되는데 메시지를 못 받음

브로커의 `advertised.listeners` 설정이 `localhost`로 되어 있어서, 최초 접속(bootstrap)은
성공하지만 실제 메시지 fetch 시 브로커가 광고한 주소(`localhost`)로 재접속을 시도하다
실패하는 경우입니다. `python kafka/diagnose.py`로 브로커 목록에 찍히는 host가
실제 접근 가능한 IP인지 확인하세요. `localhost`로 나오면 브로커 운영자에게
`advertised.listeners`를 외부 IP로 바꿔달라고 요청해야 합니다.

### `serve.py`로 실행 시 `PermissionError: [WinError 10013]`

포트가 이미 다른 프로세스에 점유되어 있거나 Windows가 예약해둔 경우입니다.

```powershell
netstat -ano | findstr :5500
taskkill /PID <위에서_나온_PID> /F
```

또는 다른 포트로 실행: `python serve.py 5501`

### Unity WebGL: "gzip-compressed... web server hosting the content was misconfigured"

Python 기본 `http.server`는 `.gz` 파일에 `Content-Encoding: gzip` 헤더를 안 붙여줘서
브라우저가 압축을 못 풉니다. `web/serve.py`를 쓰면 이 헤더가 자동으로 붙습니다
(`python -m http.server` 대신 `python serve.py` 사용).

### Unity WebGL: 서버를 고쳤는데도 여전히 같은 에러

브라우저가 예전(잘못된 헤더로 받았던) 데이터를 IndexedDB에 캐시해둔 상태일 수 있습니다.
개발자도구 → Application → Storage → "Clear site data" 또는 IndexedDB에서
`UnityCache` 삭제 후 강력 새로고침(`Ctrl+Shift+R`).

### `/api/v1/state/latest`가 503을 반환함

Consumer(`consumer_v5.py` 또는 `consumer.py`)가 아직 한 번도 메시지를 못 받았거나
꺼져있는 상태입니다. Consumer 터미널에 메시지 수신 로그가 찍히고 있는지 먼저 확인하세요.

## 다음 단계

- AI 모델을 실시간 스트림에 연결 (20초 버퍼 → 피처 추출 → `model.pkl` 예측,
  `prediction.status`를 `"warming_up"` → `"ready"`로 전환)
- SHAP 기반 원인 분석 결과를 `prediction`에 추가
- Kafka 중단 시 웹/Unity가 "데이터 지연" 상태를 얼마나 정확히/빠르게 잡아내는지 테스트 보강
- CI/CD 파이프라인의 형식적인 테스트(`assert True`)를 실제 테스트로 교체