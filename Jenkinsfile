pipeline {
    agent any
    stages {
        stage('Git Clone') {
            steps {
                git branch: 'dev',
                url: 'https://github.com/yuudong123/ai-first-project.git'
            }
        }
        stage('Docker Build') {
            steps {
                sh'''
                docker build -t yyaya/hydrotwin:latest .
                '''
            }
        }
        stage('Run Container') {
            steps{
                sh'''
                docker rm -f hydrotwin || true
                docker run -d \
                --name hydrotwin\
                -p 8000:8000\
                yyaya/hydrotwin:latest
                '''
            }
        }
    }
}