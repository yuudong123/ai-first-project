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