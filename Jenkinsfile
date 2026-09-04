pipeline {
    agent any
    environment {
        IMAGE_NAME = 'yyaya/hydrotwin-app'
    }
    options { disableConcurrentBuilds(); timeout(time: 20, unit: 'MINUTES'); buildDiscarder(logRotator(numToKeepStr: '30')) }
    triggers {
        githubPush()
    }
    parameters {
        choice(name: 'TASK', choices: ['verify', 'retrain'], description: '검증 또는 감지된 계절 드리프트 재학습')
        booleanParam(name: 'PUSH_IMAGE', defaultValue: false, description: '검증된 이미지를 Docker Hub에 푸시')
    }
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
                    docker compose config --quiet
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
                    git config --global --add safe.directory /project
                    test "$(git branch --show-current)" = "${GIT_DEPLOY_BRANCH}"
                    test -z "$(git status --porcelain --untracked-files=all)"

                    previous_revision="$(git rev-parse HEAD)"
                    rollback_image="hydrotwin-app:rollback-${BUILD_NUMBER}"
                    candidate="hydrotwin-app:candidate-${BUILD_NUMBER}"
                    docker tag hydrotwin-app:local "$rollback_image"

                    rollback() {
                      echo '배포 실패: 이전 코드와 이미지로 복구합니다.'
                      git switch --detach "$previous_revision"
                      git branch -f "$GIT_DEPLOY_BRANCH" "$previous_revision"
                      git switch "$GIT_DEPLOY_BRANCH"
                      docker tag "$rollback_image" hydrotwin-app:local
                      docker compose up -d --no-deps --force-recreate producer inference monitor api
                      docker exec hydrotwin-monitor python -m src.runtime.check
                    }
                    trap rollback ERR

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
        stage('검증된 이미지 푸시') {
            when {
                allOf {
                    expression { params.TASK == 'verify' }
                    expression { params.PUSH_IMAGE }
                }
            }
            steps {
                withCredentials([usernamePassword(credentialsId: 'docker-hub-credentials', usernameVariable: 'DOCKERHUB_USER', passwordVariable: 'DOCKERHUB_TOKEN')]) {
                    sh '''
                        set -eu
                        revision="$(git rev-parse --short=12 HEAD)"
                        echo "$DOCKERHUB_TOKEN" | docker login -u "$DOCKERHUB_USER" --password-stdin
                        trap 'docker logout >/dev/null 2>&1 || true' EXIT
                        docker tag hydrotwin-app:local "$IMAGE_NAME:$revision"
                        docker tag hydrotwin-app:local "$IMAGE_NAME:latest"
                        docker push "$IMAGE_NAME:$revision"
                        docker push "$IMAGE_NAME:latest"
                    '''
                }
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
