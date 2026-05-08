pipeline {
    agent {label 'docker-agent-jdk25'}

    tools {
        nodejs 'nodejs_26.1.0'
    }

    environment {
        APP_NAME = 'ORIC'
        DEPLOY_PATH = '/var/jenkins_home/deployments/ORIC'
    }

    stages {

        stage('Checkout') {
            steps {
                checkout scm
            }
        }

        stage('Install PNPM') {
            steps {
                sh 'npm install -g pnpm'
            }
        }

        stage('Install Dependencies') {
            steps {
                sh 'pnpm install'
            }
        }

        stage('Build Next.js') {
            steps {
                sh 'pnpm build'
            }
        }

        stage('Deploy') {
            steps {
                sh '''
                    echo "Deploying ${APP_NAME}..."

                    # Clean previous deployment
                    rm -rf ${DEPLOY_PATH}/*

                    # Copy application files
                    cp -r .next ${DEPLOY_PATH}/
                    cp -r public ${DEPLOY_PATH}/ || true
                    cp package.json ${DEPLOY_PATH}/
                    cp pnpm-lock.yaml ${DEPLOY_PATH}/ || true
                    cp next.config.* ${DEPLOY_PATH}/ || true

                    cd ${DEPLOY_PATH}

                    # Install production dependencies
                    pnpm install --prod

                    echo "${APP_NAME} deployed successfully."
                '''
            }
        }

        stage('Start Application') {
            steps {
                sh '''
                    cd ${DEPLOY_PATH}

                    # Stop previous instance
                    pkill -f "next start" || true

                    # Start Next.js app
                    nohup pnpm start > oric.log 2>&1 &
                '''
            }
        }
    }

    post {
        success {
            echo "${APP_NAME} deployment completed successfully."
        }

        failure {
            echo "${APP_NAME} deployment failed."
        }
    }
}