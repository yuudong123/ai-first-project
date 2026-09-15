# HydroTwin repository instructions

## Unity

- Unity project root: `D:\ai-first-project\unity` (local-only and excluded from Git).
- Unity version: read `D:\ai-first-project\unity\ProjectSettings\ProjectVersion.txt`; do not silently upgrade or downgrade it.
- Runtime scene: use only `D:\ai-first-project\unity\Assets\Scenes\Main.unity`.
- Do not edit `SampleScene.unity` or add it to Build Settings.
- Prefer Unity Editor scripts and `tools/unity-cli.ps1` over hand-editing Unity YAML scene or prefab files.
- Generate or refresh the owned HydroTwin scene roots with `powershell -ExecutionPolicy Bypass -File tools/unity-cli.ps1 build-scene`.
- After Unity changes, run `powershell -ExecutionPolicy Bypass -File tools/unity-cli.ps1 validate`.
- After scene hierarchy changes, also run `powershell -ExecutionPolicy Bypass -File tools/unity-cli.ps1 dump-scene` and inspect `D:\ai-first-project\artifacts\unity\scene-hierarchy.json`.
- Run EditMode tests with `powershell -ExecutionPolicy Bypass -File tools/unity-cli.ps1 test` when tests exist.
- WebGL builds go under `D:\ai-first-project\artifacts\unity\` or another explicitly selected local build directory; do not commit generated builds.
- The Unity Editor must not have this project open while a batch-mode CLI command runs.

## HydroTwin runtime contract

- Unity reads FastAPI; Unity does not connect directly to Kafka.
- MVP data endpoint: `GET /api/v1/state/latest`.
- Poll interval: one second.
- The current sensor panel must include all 17 IDs without omission: `PS1`, `PS2`, `PS3`, `PS4`, `PS5`, `PS6`, `EPS1`, `FS1`, `FS2`, `TS1`, `TS2`, `TS3`, `TS4`, `VS1`, `CE`, `CP`, `SE`.
- AI component results cover `pump`, `valve`, `cooler`, and `accumulator`.
- 학습·재학습·실시간 추론은 센서별 10초 평균 17개로 통일한다. 10초가 모이기 전에는 센서값은 계속 갱신하고 예측 상태를 warming_up으로 표시한다.
- 통합 모델 학습·평가는 사이클별 모든 10초 위치(1초 간격 51개)를 사용한다. 사이클 단위 분할을 유지하고 구간을 무작위 분할하지 않는다. stable_flag는 사이클 라벨이며 생성 초기 라벨을 생성값의 실제 정답으로 간주하지 않는다.
