# Unity CLI 작업 환경

HydroTwin Unity 프로젝트는 Unity Editor의 공식 배치 모드와 `-executeMethod`를 사용해 터미널에서 검증·테스트·씬 확인·WebGL 빌드를 실행한다.

## 환경

- Unity 프로젝트: `D:\ai-first-project\unity` (로컬 전용, Git 제외)
- 프로젝트 버전: `6000.3.23f1`
- 기본 Editor: 프로젝트 버전에 맞는 설치 경로를 Windows 레지스트리에서 자동 탐색
- Editor 경로를 직접 지정할 때: `HYDROTWIN_UNITY_EDITOR`
- 프로젝트 경로를 직접 지정할 때: `HYDROTWIN_UNITY_PROJECT` 또는 `-ProjectPath`
- 실행 씬: `Assets/Scenes/Main.unity` 하나
- CLI 스크립트: `tools/unity-cli.ps1`
- 로그와 결과: `D:\ai-first-project\artifacts\unity\`

## 명령

저장소 루트 `C:\ai-first-project`에서 실행한다.

```powershell
# 폴더, Player Settings, Main 씬 Build Settings 구성
powershell -ExecutionPolicy Bypass -File tools/unity-cli.ps1 setup

# Main 씬의 3D 플레이스홀더, 센서 17개 UI, AI 상태 UI 생성
powershell -ExecutionPolicy Bypass -File tools/unity-cli.ps1 build-scene

# 컴파일, Main 씬 등록, WebGL 모듈, 센서 ID 17개 검증
powershell -ExecutionPolicy Bypass -File tools/unity-cli.ps1 validate

# Main 씬 오브젝트·컴포넌트 계층을 JSON으로 출력
powershell -ExecutionPolicy Bypass -File tools/unity-cli.ps1 dump-scene

# Unity EditMode 테스트 실행
powershell -ExecutionPolicy Bypass -File tools/unity-cli.ps1 test

# D:\ai-first-project\artifacts\unity\webgl에 WebGL 빌드
powershell -ExecutionPolicy Bypass -File tools/unity-cli.ps1 build-webgl

# 다른 출력 경로에 WebGL 빌드
powershell -ExecutionPolicy Bypass -File tools/unity-cli.ps1 build-webgl -OutputPath D:\temp\hydrotwin-webgl

# Unity 프로젝트 경로를 일시적으로 직접 지정
powershell -ExecutionPolicy Bypass -File tools/unity-cli.ps1 validate -ProjectPath D:\ai-first-project\unity
```

## 결과 파일

| 작업 | 결과 |
|---|---|
| `validate` | `D:\ai-first-project\artifacts\unity\validation.json` |
| `dump-scene` | `D:\ai-first-project\artifacts\unity\scene-hierarchy.json` |
| `test` | `D:\ai-first-project\artifacts\unity\editmode-results.xml` |
| `build-webgl` | `D:\ai-first-project\artifacts\unity\webgl\`, `webgl-build.json` |
| 모든 작업 | `D:\ai-first-project\artifacts\unity\logs\*.log` |

## 브라우저에서 로컬 테스트

WebGL 빌드는 파일을 직접 더블클릭하지 않고 HTTP 서버를 통해 연다. 저장소 루트에서 다음 명령을 실행하면 압축 헤더를 지원하는 로컬 서버가 시작되고 기본 브라우저가 자동으로 열린다.

```powershell
powershell -ExecutionPolicy Bypass -File tools/serve-unity-webgl.ps1
```

- 주소: `http://127.0.0.1:8080/`
- 종료: 실행한 터미널에서 `Ctrl+C`
- 다른 포트: `-Port 8081`
- 브라우저 자동 실행 제외: `-NoOpen`

현재 씬은 Mock 모드이므로 FastAPI 서버 없이도 센서 17종과 AI 상태 표시를 확인할 수 있다. 실제 API 모드에서는 FastAPI를 별도로 실행하고 Unity WebGL 주소에 대한 CORS 허용이 필요하다.

Unity Editor에서 같은 프로젝트를 열어둔 상태에는 별도의 배치 모드 프로세스가 프로젝트를 동시에 열 수 없다. CLI를 실행하기 전에 해당 프로젝트의 Unity Editor 창을 닫는다.

## Unity Editor 메뉴

Editor에서도 같은 진입점을 실행할 수 있다.

```text
HydroTwin
└─ CLI
   ├─ Setup Project
   ├─ Validate Project
   ├─ Dump Main Scene
   └─ Build WebGL
```
