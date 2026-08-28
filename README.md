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
