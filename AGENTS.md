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
- Before the 20-second inference window is ready, keep sensor values updating and show prediction status as warming up.
