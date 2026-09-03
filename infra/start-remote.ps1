$ErrorActionPreference = 'Stop'
Set-Location (Split-Path $PSScriptRoot -Parent)
$env:HYDROTWIN_INFERENCE_MODULE = 'src.runtime.multi_inference'
$env:HYDROTWIN_REMOTE = '1'
if (-not $env:UNITY_WEBGL_HOST_PATH -and -not (Select-String -Path '.env' -Pattern '^UNITY_WEBGL_HOST_PATH=' -Quiet -ErrorAction SilentlyContinue) -and (Test-Path 'D:/ai-first-project/artifacts/unity/ai-labels/pro-build/Build')) {
    $env:UNITY_WEBGL_HOST_PATH = 'D:/ai-first-project/artifacts/unity/ai-labels/pro-build'
}
docker compose config --quiet
if ($LASTEXITCODE -ne 0) { throw '원격 Compose 설정 검증 실패' }
$statusPath = 'artifacts/runtime/retraining.json'
if (Test-Path -LiteralPath $statusPath) {
    $trainingState = Get-Content -LiteralPath $statusPath -Raw | ConvertFrom-Json
    if ($trainingState.status -in @('queued','running')) { throw '재학습 작업이 끝난 후 원격 모드로 전환하세요.' }
}
# 모델/볼륨/로컬 이력은 보존하고 로컬 생성 및 재학습 감지만 일시 중지한다.
docker compose stop producer monitor
if ($LASTEXITCODE -ne 0) { throw '로컬 생성·감지 중지 실패' }
docker compose up -d --build --no-deps --force-recreate inference api
if ($LASTEXITCODE -ne 0) { throw '원격 추론·API 시작 실패. docs/local-runtime.md의 로컬 복귀 절차를 확인하세요.' }
Write-Host '원격 3대 수신 모드: http://localhost:8000 (설비별 10초 수집 후 예측)'
Write-Host '로컬 producer/monitor는 중지됨. 원격 자동 재학습은 연결하지 않습니다.'
