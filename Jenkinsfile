pipeline {
    agent any

    stages {
        stage('Checkout') {
            steps {
                echo '=== HydroTwin Checkout ==='
                checkout scm
            }
        }

        stage('Repository Check') {
            steps {
                echo '=== Repository Check ==='
                sh 'pwd'
                sh 'git log -1 --oneline'
                sh 'ls -la'
            }
        }
    }

    post {
        success {
            echo 'HydroTwin Pipeline SUCCESS'
        }
        failure {
            echo 'HydroTwin Pipeline FAILED'
        }
    }
}
