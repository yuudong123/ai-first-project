# 개발서버 Docker 배포: 3대 수신·추론·웹 전용

## 범위와 기존 서비스 보호

`compose.server.yml`은 **단독 실행**한다. 루트 `docker-compose.yml` 또는 `infra/compose.remote.yml`과 합치지 않는다.
프로젝트 이름은 `hydrotwin-multi`이며, 고정 컨테이너 이름이나 기존 Docker 볼륨을 사용하지 않는다.
추론과 웹 API 두 서비스만 실행한다. 기존 Kafka, 생성기, Jenkins, 모니터를 생성·중지·교체하지 않는다.
`infra/start-remote.ps1`은 로컬 테스트 전환용이므로 공용 서버에서 실행하지 않는다.

- 입력: 기존 Kafka의 `hydraulic.sensor.multi.raw` 토픽, 1초마다 설비별 메시지.
- `equipment_id`: `station-01`, `station-02`, `station-03`.
- `timestamp`: 시간대가 포함된 ISO 8601 문자열. 서버·생성기 시간을 동기화한다.
- `sensors`: PS1~PS6, EPS1, FS1~FS2, TS1~TS4, VS1, CE, CP, SE의 유한 숫자 17개.
- 설비별 최근 10개 연속 관측의 평균 → 부품 4개 및 stable_flag → TreeSHAP → 웹·Unity.
- Kafka는 읽기만 하며 그룹 가입·오프셋 커밋·토픽 생성을 하지 않는다. 시작 시 과거 메시지를 건너뛴다.
- 수신 시 5초 초과 지연이나 2초 초과 미래 데이터는 제외한다. 관측 간격이 0.5~1.5초를 벗어나면 해당 설비만 10개를 다시 수집한다.

**멀티 설비 자동 재학습은 이 배포 범위에 없다.** 모델은 시작 시 고정된다. 기존 로컬 Jenkinsfile은 monitor 컨테이너에 의존하므로 이 배포에 그대로 적용하지 않는다.
기존 서버의 감지·재학습 흐름은 그대로 두고, 향후 설비별 감지 입력·라벨 정책·검증·모델 교체를 별도로 연결해야 한다.

## Git 외에 필요한 자산

모델·데이터·Unity 빌드는 Git 및 이미지에 포함하지 않는다. 신뢰할 수 있는 팀 자산을 서버에 별도로 복사하고 아래 구조를 유지한다.
joblib은 로드 시 코드를 실행할 수 있으므로 출처 불명 모델을 사용하지 않는다.

```text
/srv/hydrotwin/assets/
  models/predict/integrated_lgbm.joblib
  data/processed/simulator/uci_1hz_17sensors.npz
  data/processed/split_ids_accumulator_stratified.json
  data/raw/uci_hydraulic/extracted/profile.txt
  unity/pro-build/Build/pro-build.loader.js
  unity/pro-build/Build/pro-build.data.gz
  unity/pro-build/Build/pro-build.framework.js.gz
  unity/pro-build/Build/pro-build.wasm.gz
  unity/pro-build/StreamingAssets/  (빌드에 있는 경우 함께 복사)
```

분류 모델은 10초 평균 17개 입력, 5개 타깃을 포함해야 한다. 서버는 시작 시 실제 추론·TreeSHAP 호환성과 자산 누락을 확인한다.
정상 밴드 표시에는 위 데이터 파일 3개가 필요하며 전체 센서 원본 파일은 필요 없다.
밴드는 정상·안정 5개 기준 사이클의 통계 범위이지 안전 한계나 고장 정답이 아니다.
Unity는 현재 웹과 동일한 `pro-build` 파일명 및 `equipment_states` 지원 빌드를 사용한다.

## 서버 설정

저장소 루트에서 `infra/server.env.example`을 `.env.server`로 복사해 수정한다. `.env.server`는 Git에서 제외된다.

- `MODEL_HOST_DIR`, `DATA_HOST_DIR`, `UNITY_WEBGL_HOST_PATH`: 서버에 실제로 존재하는 절대 경로. Windows Docker Desktop에서는 `D:/...` 형식도 가능하다.
- `KAFKA_BROKER`: 기존 브로커 접속 주소. Kafka가 반환하는 advertised.listeners 주소도 컨테이너에서 접근 가능해야 한다. 컨테이너 안의 localhost는 호스트가 아니다.
- `WEB_BIND_ADDRESS`: 기본은 `127.0.0.1`. 팀 내부망에서 직접 접속하려면 서버의 내부망 IP로 설정하고 방화벽 접근 대상을 제한한다. 외부 공개는 인증·TLS를 제공하는 프록시를 별도 구성한다.
- `WEB_PORT`: 기존 서버와 충돌하지 않는 포트. 예: 8001.
- `HYDROTWIN_IMAGE_TAG`: 배포 커밋 번호 등 고유 태그. 이전 이미지와 자산을 보존하면 되돌릴 수 있다.

Compose v2가 필요하다. 기존 서비스를 실행하는 디렉터리를 덮어쓰지 말고 별도 배포 체크아웃을 사용하는 것이 좋다.

## 빌드·실행·검증

아래 명령은 저장소 루트에서 실행한다. 개발서버에서 실행하기 전 운영 담당자가 경로, 포트, Kafka 주소를 확인한다.

```sh
docker compose --env-file .env.server -f compose.server.yml config --quiet
docker compose --env-file .env.server -f compose.server.yml build
docker compose --env-file .env.server -f compose.server.yml up -d --no-build
docker compose --env-file .env.server -f compose.server.yml ps
docker compose --env-file .env.server -f compose.server.yml logs --tail 100 inference api
```

이미지 안에 코드가 들어가므로 Git pull만으로 실행 코드가 바뀌지 않는다. 코드 변경 시 새 태그로 다시 빌드·실행한다.
모델과 표시용 데이터·Unity는 읽기 전용이며, 최신 수신 상태만 이 프로젝트 전용 runtime 볼륨에 저장한다.

새 메시지가 10초 이상 연속 수신된 후:

```sh
docker compose --env-file .env.server -f compose.server.yml exec inference python -m src.runtime.check
```

웹에서 설비 3대를 각각 선택해 센서값·AI 진단·차트·Unity 이름표를 확인한다.
`/api/v1/state/latest`의 `equipment_states`에 3개 ID와 각각의 prediction.status=ready가 있어야 한다.
`/api/v1/sensors/reference-bands`가 성공해야 정상 밴드가 표시된다.
`/health`는 웹 프로세스 생존 검사다. health가 정상이어도 Kafka 연결이나 세 설비의 예측 준비를 보장하지 않으므로 위 연결 검사를 반드시 함께 한다.

## 중지·복구

```sh
docker compose --env-file .env.server -f compose.server.yml stop
```

이 명령은 이 프로젝트의 두 서비스만 중지한다. 기존 Kafka·생성기·Jenkins는 유지된다.
롤백은 이전 배포 체크아웃과 해당 이미지 태그·자산 경로를 준비하고 `up -d --no-build`로 수행한다.
모델을 교체했다면 inference 재생성이 필요하다. runtime 볼륨을 공유하는 추론 복제본을 여러 개 실행하지 않는다.
데이터가 끊기면 과거 파일이 남아 있더라도 API는 해당 설비를 stale로 표시한다.

## 검증 기록 (2026-09-03)

- Python 회귀 테스트 49개, JavaScript 테스트 6개 통과.
- `infra/Dockerfile.server`로 새 이미지 빌드 성공. 생성용 TensorFlow 없이 분류 추론 실행 확인.
- 로컬 Docker에 별도 프로젝트 `hydrotwin-deploy-check`, 웹 포트 18000으로 배포 설정 실행.
- 실제 원격 Kafka에서 설비 3대를 수신하고 각각 10초 추론·TreeSHAP·API 연결 검사 통과.
- 세 설비의 서로 다른 PS1 값과 ready 상태 확인. 정상 밴드·웹·Unity 4개 빌드 파일 200 응답, 압축 헤더와 wasm MIME 확인.
- 검증 후 테스트 컨테이너만 중지. 실제 개발서버 배포는 수행하지 않았으며 서버 자산·네트워크·포트는 운영 담당자의 확인이 필요하다.
