param(
    [string]$ProjectPath = 'D:\ai-first-project\unity',
    [string]$OutputPath = 'D:\ai-first-project\artifacts\unity\ai-labels\pro-build'
)
$ErrorActionPreference = 'Stop'
# 루트에서 실행하며 로컬 프로젝트에 기록된 버전으로만 빌드한다.
$versionLine = Get-Content -LiteralPath (Join-Path $ProjectPath 'ProjectSettings\ProjectVersion.txt') |
    Where-Object { $_ -match '^m_EditorVersion: ' } | Select-Object -First 1
$version = ($versionLine -replace '^m_EditorVersion: ', '').Trim()
$editorPath = "D:\UnityHub\Editor\$version\Editor\Unity.exe"
if (-not (Test-Path -LiteralPath $editorPath)) { throw "Unity $version 설치 경로를 확인하세요: $editorPath" }
$openEditor = Get-CimInstance Win32_Process -Filter "Name='Unity.exe'" |
    Where-Object { $_.CommandLine -like "*$ProjectPath*" }
if ($openEditor) { throw '이 프로젝트의 Unity 에디터를 닫은 뒤 다시 실행하세요.' }
$outputParent = Split-Path $OutputPath -Parent
New-Item -ItemType Directory -Path $outputParent -Force | Out-Null
$logPath = Join-Path $outputParent 'unity-build.log'
$arguments = "-batchmode -quit -job-worker-count 2 -projectPath `"$ProjectPath`" -executeMethod HydroTwin.Editor.HydroTwinCli.BuildWebGl -outputPath `"$OutputPath`" -logFile `"$logPath`""
$buildProcess = Start-Process -FilePath $editorPath -ArgumentList $arguments -WindowStyle Hidden -PassThru
# Unity가 남기는 라이선스 보조 프로세스가 아니라 에디터 자체의 종료만 기다린다.
$buildProcess.WaitForExit()
if ($buildProcess.ExitCode -ne 0) { throw "Unity 빌드 실패. 로그: $logPath" }
if (-not (Test-Path -LiteralPath (Join-Path $OutputPath 'Build\pro-build.loader.js'))) {
    throw "예상 빌드 파일이 없습니다. 출력 폴더 이름을 pro-build로 지정하세요. 로그: $logPath"
}
Write-Host "Unity 빌드 완료: $OutputPath"
Write-Host 'API를 다시 시작하면 새 빌드를 제공합니다: docker compose restart api'
