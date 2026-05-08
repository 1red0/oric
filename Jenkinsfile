pipeline {
    agent {label 'docker-agent-jdk25'}

    tools {
        nodejs 'nodejs_26.1.0'
    }

    environment {
        APP_NAME = 'ORIC'
        DEPLOY_PATH = '/Deployments/ORIC'
    }

    stages {

        stage('Checkout') {
            steps {
                checkout scm
            }
        }

        stage('Install Dependencies') {
            steps {
                sh 'npm install'
            }
        }

        stage('Build Next.js') {
            steps {
                sh 'npm run build'
            }
        }

        stage('Deploy') {
            steps {
                sh '''
                    echo "Deploying ${APP_NAME}..."

                    mkdir -p ${DEPLOY_PATH}

                    # Clean previous deployment
                    rm -rf ${DEPLOY_PATH}/*

                    # Copy Next.js build files
                    cp -r .next ${DEPLOY_PATH}/
                    cp -r public ${DEPLOY_PATH}/ || true
                    cp package.json ${DEPLOY_PATH}/
                    cp package-lock.json ${DEPLOY_PATH}/ || true
                    cp next.config.* ${DEPLOY_PATH}/ || true

                    # Install production dependencies
                    cd ${DEPLOY_PATH}
                    npm install --omit=dev

                    echo "${APP_NAME} deployed successfully."
                '''
            }
        }

        stage('Start Application') {
            steps {
                sh '''
                    cd ${DEPLOY_PATH}

                    # Stop old Next.js instance
                    pkill -f "next start" || true

                    # Start application
                    nohup npm run start > oric.log 2>&1 &
                '''
            }
        }
    }

    post {
        success {
            echo '${APP_NAME} deployment completed successfully.'
        }

        failure {
            echo '${APP_NAME} deployment failed.'
        }
    }
}