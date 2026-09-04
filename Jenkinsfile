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
    stages {
        stage('코드와 회귀 테스트') {
            steps { sh 'docker exec hydrotwin-monitor python -m pytest tests -q' }
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
        failure { sh 'docker exec hydrotwin-monitor python -m src.runtime.train_job --mark-failed || true' }
    }
}
