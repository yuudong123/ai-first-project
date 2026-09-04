param([switch]$SkipBuild)
$ErrorActionPreference = 'Stop'
Set-Location $PSScriptRoot
# 모든 실행은 설비 3대와 멀티 설비 토픽만 사용한다.
$env:KAFKA_BROKER = 'kafka:29092'
$env:KAFKA_TOPIC = 'hydraulic.sensor.multi.raw'

function Set-DefaultEnvValue([string]$Name, [string]$Value) {
    $envPath = Join-Path $PSScriptRoot '.env'
    if (-not (Select-String -LiteralPath $envPath -Pattern "^$([regex]::Escape($Name))=" -Quiet -ErrorAction SilentlyContinue)) {
        Add-Content -LiteralPath $envPath -Value "$Name=$Value"
    }
}
# 이 PC의 기존 Unity 빌드 경로는 로컬 스크립트에서만 보완한다.
if (-not $env:UNITY_WEBGL_HOST_PATH -and -not (Select-String -Path '.env' -Pattern '^UNITY_WEBGL_HOST_PATH=' -Quiet -ErrorAction SilentlyContinue) -and (Test-Path 'D:/ai-first-project/artifacts/unity/ai-labels/pro-build/Build')) {
    $env:UNITY_WEBGL_HOST_PATH = 'D:/ai-first-project/artifacts/unity/ai-labels/pro-build'
}
# 비밀번호는 실행할 때 생성하며 저장소에 올리지 않는다.
if (-not (Test-Path -LiteralPath '.env')) {
    $secretBytes = New-Object byte[] 24
    $generator = [Security.Cryptography.RandomNumberGenerator]::Create()
    $generator.GetBytes($secretBytes)
    $generator.Dispose()
    $localPassword = [BitConverter]::ToString($secretBytes).Replace('-', '')
    $settings = "JENKINS_ADMIN_USER=admin`nJENKINS_ADMIN_PASSWORD=$localPassword`nTEMP_OFFSET_MIN=-4`nTEMP_OFFSET_MAX=4`nPRESSURE_OFFSET_PERCENT=10`nDRIFT_INTERVAL_MIN_SEC=60`nDRIFT_INTERVAL_MAX_SEC=1200`nDRIFT_RAMP_SEC=30`nINITIAL_NORMAL_SEC=120`n"
    [IO.File]::WriteAllText((Join-Path (Get-Location) '.env'), $settings)
}
$projectPath = $PSScriptRoot.Replace('\', '/')
Set-DefaultEnvValue 'PROJECT_HOST_DIR' $projectPath
Set-DefaultEnvValue 'DATA_HOST_DIR' "$projectPath/data"
Set-DefaultEnvValue 'MODEL_HOST_DIR' "$projectPath/models"
Set-DefaultEnvValue 'STATE_HOST_DIR' "$projectPath/artifacts/runtime"
Set-DefaultEnvValue 'GIT_REPOSITORY_URL' 'https://github.com/yuudong123/ai-first-project.git'
Set-DefaultEnvValue 'GIT_DEPLOY_BRANCH' 'dev'
docker compose config --quiet
if ($LASTEXITCODE -ne 0) { throw 'Compose 설정 검증 실패' }
if (-not $SkipBuild) {
    docker compose build
    if ($LASTEXITCODE -ne 0) { throw '이미지 빌드 실패' }
}
docker compose run --rm bootstrap
if ($LASTEXITCODE -ne 0) { throw '데이터 준비 또는 최초 학습 실패' }
docker compose up -d kafka api jenkins inference monitor producer
if ($LASTEXITCODE -ne 0) { throw '서비스 실행 실패' }
Write-Host '설비 3대 웹: http://localhost:8000 / Jenkins: http://localhost:8080'
Write-Host 'Jenkins 계정은 .env 파일에서 확인하세요. 재실행 시 기존 볼륨과 모델을 보존합니다.'
