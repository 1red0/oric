pipeline {
    agent {label 'docker-agent-jdk25'}

    tools {
        nodejs 'nodejs_26.1.0'
    }

    stages {
        stage('Check Node Version') {
            steps {
                sh 'node -v'
                sh 'npm -v'
                echo 'Hello'
            }
        }
    }
}