pipeline {
    agent any
    options {
        disableConcurrentBuilds()
        timeout(time: 20, unit: 'MINUTES')
        buildDiscarder(logRotator(numToKeepStr: '30'))
    }
    triggers {
        githubPush()
        pollSCM('* * * * *')
    }
    stages {
        stage('Checkout dev') {
            steps {
                checkout scm
            }
        }
        stage('Build and test') {
            steps {
                sh '''
                    set -eu
                    candidate="hydrotwin-app:candidate-${BUILD_NUMBER}"
                    docker compose --env-file /project/.env -f docker-compose.yml config --quiet
                    docker build --target app -t "$candidate" .
                    docker run --rm \
                      --mount "type=bind,source=${PROJECT_HOST_DIR}/data,target=/app/data" \
                      --mount "type=bind,source=${PROJECT_HOST_DIR}/models,target=/app/models,readonly" \
                      --mount "type=bind,source=${PROJECT_HOST_DIR}/artifacts,target=/app/artifacts" \
                      "$candidate" python -m pytest tests -q
                '''
            }
        }
        stage('Administrator approval') {
            options {
                timeout(time: 10, unit: 'MINUTES')
            }
            steps {
                script {
                    def approver = env.JENKINS_ADMIN_USER?.trim() ?: 'admin'
                    input(
                        message: '검증된 빌드를 운영 환경에 배포하시겠습니까?',
                        ok: '운영 배포 승인',
                        submitter: approver
                    )
                }
            }
        }
        stage('Deploy') {
            steps {
                sh '''
                    set -eu
                    candidate="hydrotwin-app:candidate-${BUILD_NUMBER}"
                    docker tag "$candidate" hydrotwin-app:local
                    docker compose --env-file /project/.env -f docker-compose.yml up -d --no-build --no-deps --force-recreate producer inference monitor api
                '''
            }
        }
        stage('Verify') {
            steps {
                sh '''
                    set -eu
                    attempts=0
                    until docker exec hydrotwin-monitor python -m src.runtime.check; do
                      attempts=$((attempts + 1))
                      test "$attempts" -lt 18
                      sleep 5
                    done
                '''
            }
        }
    }
    post {
        always {
            sh 'docker image rm "hydrotwin-app:candidate-${BUILD_NUMBER}" >/dev/null 2>&1 || true'
        }
    }
}
