pipeline {
    agent {label 'nodejs'}

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