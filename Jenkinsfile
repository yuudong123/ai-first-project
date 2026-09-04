pipeline {
    agent any
    environment {
        DOCKER_CREDENTIALS = credentials('docker-hub-credentials')
        IMAGE_NAME = 'yyaya/hydrotwin-app'
    }
    options { disableConcurrentBuilds(); timeout(time: 20, unit: 'MINUTES'); buildDiscarder(logRotator(numToKeepStr: '30')) }
    triggers {
        githubPush()
    }
    parameters { choice(name: 'TASK', choices: ['verify', 'retrain'], description: '검증 또는 감지된 계절 드리프트 재학습') }
    triggers { githubPush() }
    stages {
        stage('dev 소스 받기') {
            when { expression { params.TASK == 'verify' } }
            steps { checkout scm }
        }
        stage('후보 이미지 빌드와 테스트') {
            when { expression { params.TASK == 'verify' } }
            steps {
                sh '''
                    set -eu
                    candidate="hydrotwin-app:candidate-${BUILD_NUMBER}"
                    docker build --target app -t "$candidate" .
                    docker run --rm \
                      --mount "type=bind,source=${PROJECT_HOST_DIR}/data,target=/app/data,readonly" \
                      --mount "type=bind,source=${PROJECT_HOST_DIR}/models,target=/app/models,readonly" \
                      --mount "type=bind,source=${PROJECT_HOST_DIR}/artifacts,target=/app/artifacts" \
                      "$candidate" python -m pytest tests -q
                '''
            }
        }
        stage('dev 배포') {
            when { expression { params.TASK == 'verify' } }
            steps {
                sh '''
                    set -eu
                    cd /project
                    test "$(git branch --show-current)" = "${GIT_DEPLOY_BRANCH}"
                    test -z "$(git status --porcelain --untracked-files=all)"

                    previous_revision="$(git rev-parse HEAD)"
                    rollback_image="hydrotwin-app:rollback-${BUILD_NUMBER}"
                    candidate="hydrotwin-app:candidate-${BUILD_NUMBER}"
                    docker tag hydrotwin-app:local "$rollback_image"

                    rollback() {
                      echo '배포 실패: 이전 코드와 이미지로 복구합니다.'
                      git reset --hard "$previous_revision"
                      docker tag "$rollback_image" hydrotwin-app:local
                      docker compose up -d --no-deps --force-recreate producer inference monitor api
                    }
                    trap rollback ERR

                    git config --global --add safe.directory /project
                    git fetch --no-tags "$WORKSPACE" "$GIT_COMMIT"
                    git merge --ff-only "$GIT_COMMIT"
                    docker tag "$candidate" hydrotwin-app:local
                    docker compose up -d --no-deps --force-recreate producer inference monitor api

                    attempts=0
                    until docker exec hydrotwin-monitor python -m src.runtime.check; do
                      attempts=$((attempts + 1))
                      test "$attempts" -lt 18
                      sleep 5
                    done
                    trap - ERR
                    docker image rm "$rollback_image" "$candidate" >/dev/null 2>&1 || true
                '''
            }
        }
        stage('계절 증강 재학습과 성능 게이트') {
            when { expression { params.TASK == 'retrain' } }
            steps { sh 'docker exec hydrotwin-monitor python -m src.runtime.train_job' }
        }
        stage('실시간 서비스 확인') {
            steps { sh 'docker exec hydrotwin-monitor python -m src.runtime.check' }
        }
        stage('도커 이미지 빌드 및 푸시') {
            steps {
                sh '''
                    # 1. 젠킨스가 관리하는 도커 크레덴셜 비밀번호(_PSW)와 아이디(_USR)로 로그인
                    echo "$DOCKER_CREDENTIALS_PSW" | docker login -u "$DOCKER_CREDENTIALS_USR" --password-stdin
                    
                    # 2. 현재 상태의 컨테이너를 이미지로 커밋하거나 새로 빌드 (상황에 맞게 선택)
                    # 예시 A: 작동 중인 컨테이너의 변경사항을 이미지로 저장할 때 (docker commit)
                    docker commit hydrotwin-monitor $IMAGE_NAME:latest
                    
                    # 예시 B: 만약 Dockerfile로 새로 빌드한다면 아래 주석을 해제하세요
                    # docker build -t $IMAGE_NAME:latest .
                    
                    # 3. 도커 허브로 푸시
                    docker push $IMAGE_NAME:latest
                    
                    # 4. 보안을 위해 로그아웃
                    docker logout
                '''
            }
        }
    }
    post {
        failure {
            script {
                if (params.TASK == 'retrain') {
                    sh 'docker exec hydrotwin-monitor python -m src.runtime.train_job --mark-failed || true'
                }
            }
        }
    }
}
