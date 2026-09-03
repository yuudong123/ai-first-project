pipeline {
    agent any
    options { disableConcurrentBuilds(); timeout(time: 20, unit: 'MINUTES'); buildDiscarder(logRotator(numToKeepStr: '30')) }
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
    }
    post {
        failure { sh 'docker exec hydrotwin-monitor python -m src.runtime.train_job --mark-failed || true' }
    }
}
