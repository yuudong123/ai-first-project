# 설비 3대 Docker 실행 가이드

## 실행 원칙

이 프로젝트는 모든 환경에서 `station-01`, `station-02`, `station-03` 세 대만 사용한다.
단일 설비 생성·추론·화면 호환 경로는 없다. 세 설비는 같은 V5 모델을 공유하지만 생성 상태,
10초 추론 버퍼, 최신 상태와 드리프트 기준은 서로 분리한다.

## 최초 실행

Docker Desktop을 실행하고 프로젝트 루트에서 다음을 실행한다.

```powershell
Set-ExecutionPolicy -Scope Process -ExecutionPolicy Bypass
.\start.ps1
```

`start.ps1`은 다음 순서로 동작한다.

1. 애플리케이션과 Jenkins 이미지를 빌드한다.
2. 원본 데이터를 1Hz로 전처리하고 10초 모델을 준비한다.
3. 로컬 Kafka를 시작한다.
4. `kafka/producer.py`가 매초 설비 3대의 메시지 세 개를 보낸다.
5. `kafka/consumer.py`가 설비별 10개를 모아 각각 추론한다.
6. 드리프트 감지기, Jenkins, API와 웹을 함께 실행한다.

기존 이미지를 그대로 쓸 때만 다음을 사용한다.

```powershell
.\start.ps1 -SkipBuild
```

## 고정 데이터 계약

- Topic: `hydraulic.sensor.multi.raw`
- 설비 ID: `station-01`, `station-02`, `station-03`
- 메시지: `equipment_id`, `timestamp`, 센서 17개, `run_id`, `event_id`, `segment_id`, `reference_context`
- API: `GET /api/v1/state/latest`
- 상태 파일: `artifacts/runtime/latest.json`

API는 `equipment_states`에 세 설비가 모두 있어야 정상 응답한다. 설비가 누락됐을 때 다른
설비의 값을 복제하지 않는다. 각 설비는 초기 10초 동안 `warming_up`이며 5초 이상 메시지가
멈추면 해당 설비만 `stale`이 된다.

## 생성 시나리오와 드리프트

- 최초 120초: 설비마다 서로 다른 정상·안정 초기값
- 이후: 안정 초기값 120초, 불안정 초기값 60초 반복
- 온도: 세 설비에 공통으로 -4~+4°C 범위의 계절 offset
- 압력: 세 설비의 각 기준 압력에서 -10~+10% 범위
- 다음 드리프트: 이전 시작 후 무작위 60~1,200초
- 이동 시간: 기본 30초

초기 사이클 라벨과 주입 offset은 진단 기록일 뿐 생성값의 정답으로 사용하지 않는다.
부품·안정 상태는 각 설비의 실제 생성값을 10초 모델에 넣은 결과만 화면에 표시한다.

드리프트 감지기는 설비별 초기 기준과 이동 구간을 따로 유지한다. 같은 검사 시점에 세 설비가
모두 드리프트를 확인하고 offset이 안정됐을 때만 중앙값 offset으로 공통 분류 모델의 재학습을
요청한다. 불안정 초기값 구간은 계절 변화 학습에서 제외한다.

## 설정

`.env`의 주요 항목:

```dotenv
JENKINS_ADMIN_USER=admin
JENKINS_ADMIN_PASSWORD=직접_설정한_비밀번호
TEMP_OFFSET_MIN=-4
TEMP_OFFSET_MAX=4
PRESSURE_OFFSET_PERCENT=10
DRIFT_INTERVAL_MIN_SEC=60
DRIFT_INTERVAL_MAX_SEC=1200
DRIFT_RAMP_SEC=30
INITIAL_NORMAL_SEC=120
OPERATING_SEGMENT_SEC=60
UNITY_WEBGL_HOST_PATH=D:/ai-first-project/artifacts/unity/ai-labels/pro-build
WEB_BIND_ADDRESS=127.0.0.1
WEB_PORT=8000
KAFKA_BIND_ADDRESS=127.0.0.1
KAFKA_ADVERTISED_HOST=localhost
```

`.env`가 없으면 `start.ps1`이 비밀번호와 기본 시나리오 설정을 생성한다. 다른 PC가 Kafka를
받아야 하는 개발 서버에서는 두 Kafka 값을 서버 내부망 IP로 바꾸고 Windows 방화벽의 TCP
9092를 내부망에만 허용한다.

## GitHub 웹훅 자동 배포

자동 배포 대상은 `dev` 브랜치 하나다. GitHub push가 들어오면 Jenkins가 다음 순서로 처리한다.

1. 비공개 GitHub 저장소에서 해당 `dev` 커밋과 `Jenkinsfile`을 받는다.
2. `hydrotwin-app:candidate-빌드번호` 후보 이미지를 만든다.
3. 후보 이미지에서 전체 Python 회귀 테스트를 실행한다.
4. 테스트가 성공했을 때만 개발 서버의 `C:/ai-first-project`를 fast-forward한다.
5. `producer`, `inference`, `monitor`, `api`를 새 이미지로 교체한다.
6. 설비 3대 연결 검사가 실패하면 직전 코드와 이미지로 자동 복구한다.

비공개 저장소를 읽을 수 있도록 GitHub fine-grained personal access token을 만들고 저장소
`Contents: Read-only` 권한만 준다. 개발 서버의 `.env`에는 다음 값을 직접 추가한다. 실제
토큰은 Git에 커밋하지 않는다.

```dotenv
GITHUB_USERNAME=GitHub_사용자명
GITHUB_TOKEN=발급한_토큰
GIT_REPOSITORY_URL=https://github.com/yuudong123/ai-first-project.git
GIT_DEPLOY_BRANCH=dev
GIT_CREDENTIAL_ID=hydrotwin-github
PROJECT_HOST_DIR=C:/ai-first-project
DATA_HOST_DIR=C:/ai-first-project/data
MODEL_HOST_DIR=C:/ai-first-project/models
STATE_HOST_DIR=C:/ai-first-project/artifacts/runtime
```

설정을 넣은 다음 Jenkins 이미지와 컨테이너를 한 번 갱신한다.

```powershell
docker compose build jenkins
docker compose up -d --force-recreate jenkins
```

GitHub 저장소의 `Settings → Webhooks → Add webhook`에서 다음과 같이 등록한다.

- Payload URL: `https://외부에서-접근-가능한-Jenkins주소/github-webhook/`
- Content type: `application/json`
- Event: `Just the push event`
- Active: 체크

GitHub 서버는 `localhost`나 `192.168.x.x`로 접속할 수 없다. Jenkins 자체 포트를 인터넷에
직접 개방하지 말고, 인증과 HTTPS가 적용된 리버스 프록시 또는 터널을 통해 `/github-webhook/`
경로만 도달하게 한다. Jenkins 화면의 `GitHub Hook Log`에서 응답을 확인하고 최초 1회는
`hydrotwin-local` 작업을 수동 실행해 비공개 저장소 인증과 배포 경로를 검증한다.

자동 배포 중 개발 서버 작업 폴더에 커밋하지 않은 파일이 있으면 Jenkins는 이를 덮어쓰지
않고 배포를 실패 처리한다. 개발 서버 폴더는 배포 전용으로 유지한다.

## 검증

```powershell
docker compose ps
docker compose logs --tail 100 producer inference monitor api jenkins
docker exec hydrotwin-monitor python -m src.runtime.check
```

연결 검사는 다음을 모두 확인한다.

- 설비 ID 세 개가 정확히 존재함
- 각 설비에 센서 17개가 존재함
- 설비마다 최근 10초 추론이 `ready`임
- 부품 네 개와 영향 센서가 존재함

웹은 `http://localhost:8000`, Jenkins는 `http://localhost:8080`에서 확인한다.

## 모델 교체

재학습 후보는 기존 라벨 데이터에 감지된 계절 offset을 합성해 학습한다. 실시간 AI 예측을
정답 라벨로 사용하지 않는다. 원본 환경과 계절 환경의 성능 기준을 모두 통과한 경우에만 운영
모델을 교체한다. 추론 컨슈머는 모델 파일 변경을 감지해 새 모델을 읽고 세 설비 모두의 새
모델 버전 반영을 확인한다.

## Unity 빌드

Unity는 Kafka에 직접 접속하지 않고 API의 설비 3대 응답을 받는다. 최신 WebGL 빌드에는
`_App.ApplyWebState(string)` 진입점이 있어야 한다. 빌드 경로를 바꾼 뒤에는 API 컨테이너를
다시 만들어 연결 경로를 갱신한다.

```powershell
docker compose up -d --force-recreate api
```

## 중지와 재시작

```powershell
docker compose stop
.\start.ps1 -SkipBuild
```

일상적인 중지에 `docker compose down -v`를 사용하지 않는다. `-v`는 Kafka와 Jenkins 볼륨을
삭제한다.
