# 확정 팀 마일스톤

- 팀장: **조현재**
- 팀원: **홍유나, 박민, 신종건**
- 프로젝트 기간: 2026-08-28 ~ 2026-09-15
- 이 문서는 팀이 정한 현재 배정을 그대로 기록한다.
- 세부 구현 범위와 완료 기준: [팀 상세 작업 명세](team-task-board.md)

## 담당 영역

| 이름 | 주 담당 |
|---|---|
| 조현재 | 전체 일정, Unity 3D·WebGL, API 규격 협의, 전체 통합, 발표·데모 |
| 홍유나 | UCI 데이터 분석·전처리·EDA·특징 추출, 모델 학습·평가·SHAP |
| 박민 | FastAPI, 예측·상태 API, 웹 UI·Chart.js, AI·Unity 연결 지원 |
| 신종건 | Docker·Compose, pytest, Jenkins CI/CD, MLflow·재학습 데모 |

## 날짜별 마일스톤

| 기간 | 핵심 목표 | 조현재 | 홍유나 | 박민 | 신종건 |
|---|---|---|---|---|---|
| 8/28(금) | 프로젝트 구조 확정 | 전체 일정·Unity 구조 설계, API 규격 협의 | UCI 데이터 구조·센서·라벨 분석 | FastAPI 구조 설계, API 규격 협의 | Docker·Jenkins·pytest 구조 설계 |
| 8/31(월)~9/2(수) | 1차 기능 개발 | Unity 설비 3D 기본 구성, 상태별 색상 구현 | 데이터 전처리·EDA·특징 추출·학습 데이터 구성 | FastAPI 기본 API, 웹 기본 화면 구현 | Dockerfile·Compose·pytest 기본 환경 구성 |
| 9/3(목)~9/4(금) | 개별 기능 1차 완성 | 임시 데이터 기반 상태 변화·부품 클릭 UI | RandomForest 1차 모델 학습·평가 | 예측 API·상태 API·Chart.js 기본 차트 | FastAPI Docker 실행·전처리/API 테스트 |
| 9/7(월)~9/10(목) | 1차 시스템 통합 | FastAPI↔Unity 연결, 실제 예측 결과 반영 | LightGBM 비교·최종 모델 선정 및 저장 | 실제 AI 모델 FastAPI 연결, Unity API 지원 | Jenkins CI/CD 구성, 자동 테스트·빌드 연결, MLflow 실험 기록·모델 버전 관리·재학습 데모 |
| 9/11(금) | 전체 통합·기능 동결 | Unity WebGL 빌드, 전체 서비스 최종 통합 | 최종 모델 성능 검증·발표용 결과 정리 | Web·API↔Unity WebGL 통합 확인 | Docker Compose·Jenkins 배포환경 최종 점검 |
| 9/14(월) | 발표 준비 | 전체 데모 점검·발표 흐름·리허설 총괄 | AI 결과·성능·SHAP 발표자료 준비 | API·웹 기능 발표자료·데모 점검 | MLOps·CI/CD 발표자료·실행환경 점검 |
| 9/15(화) | 최종 발표 | 발표·Unity 데모 | AI 파트 발표·질의응답 | API·웹 파트 발표·질의응답 | MLOps 파트 발표·질의응답 |

## Kafka 요구사항 보완안

강사가 요구한 Kafka 실시간 스트리밍은 원래 마일스톤 표에 명시되어 있지 않다. 아래는 기존 담당을 크게 바꾸지 않는 작업 배분안이며 팀 확인 후 확정한다.

| Kafka 작업 | 권장 담당 | 연결되는 기존 업무 |
|---|---|---|
| Test 센서 재생용 데이터 생성 | 홍유나 | 데이터 전처리·특징 추출 |
| Kafka 브로커·Topic·Docker Compose 구성 | 신종건 | Docker·Compose·Jenkins |
| Producer·Consumer와 추론 API 연결 | 박민 | FastAPI·예측 API |
| Kafka 결과의 Unity 표현·전체 통합 | 조현재 | Unity·전체 통합 |

Kafka는 8/31~9/4에 각자 최소 기능을 만들고, 9/7~9/8 시스템 통합 단계에서 실제 모델·Unity와 연결하는 일정이 적절하다.
