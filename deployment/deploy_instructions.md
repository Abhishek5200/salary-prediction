# 🚀 Deployment Guide - IBM Salary Prediction Project

This guide provides step-by-step instructions for deploying the AI Salary Prediction system to various environments.

## 📋 Deployment Options

1. [Local Development](#local-development)
2. [IBM Cloud Watson ML](#ibm-cloud-watson-ml)
3. [IBM Cloud Foundry](#ibm-cloud-foundry)
4. [Docker Deployment](#docker-deployment)
5. [Production Deployment](#production-deployment)

## 🏠 Local Development

### Prerequisites
- Python 3.8+
- Git
- Virtual environment capability

### Quick Start

```bash
# Clone repository
git clone https://github.com/your-username/salary-prediction.git
cd salary-prediction

# Setup virtual environment
python -m venv venv
source venv/bin/activate  # Windows: venv\Scripts\activate

# Install dependencies
pip install -r requirements.txt

# Generate sample data and train models
cd src
python data/sample_salary_data.py
python ensemble_models.py

# Run the application
cd ../web_app
python app.py
```

Visit `http://localhost:5000` to access the application.

## ☁️ IBM Cloud Watson ML

### Step 1: Create IBM Cloud Services

#### Watson Machine Learning
1. Log into [IBM Cloud Console](https://cloud.ibm.com)
2. Navigate to **Catalog** → **AI/Machine Learning** → **Watson Machine Learning**
3. Select **Lite** plan (free)
4. Create service and note credentials:
   - API Key
   - Instance ID
   - Service URL

#### Watson Assistant
1. In Catalog, search **Watson Assistant**
2. Create service instance
3. Launch Watson Assistant
4. Create new assistant or import configuration
5. Note credentials:
   - API Key
   - Assistant ID
   - Service URL

### Step 2: Configure Environment

Create `.env` file with credentials:

```env
# IBM Cloud Credentials
IBM_CLOUD_API_KEY=your_api_key_here

# Watson Machine Learning
WML_URL=https://us-south.ml.cloud.ibm.com
WML_INSTANCE_ID=your_wml_instance_id

# Watson Assistant
WATSON_ASSISTANT_API_KEY=your_assistant_api_key
WATSON_ASSISTANT_URL=your_assistant_url
WATSON_ASSISTANT_ID=your_assistant_id
```

### Step 3: Deploy Models

```bash
# Setup Watson integration
python src/ibm_watson_integration.py
```

This will:
- Create deployment space
- Upload trained models
- Deploy models for online scoring
- Test predictions

### Step 4: Import Watson Assistant Configuration

1. Open Watson Assistant dashboard
2. Go to **Actions** → **Upload/Download**
3. Upload `watson_assistant/assistant_config.json`
4. Train the assistant
5. Test conversation flow

## 🌐 IBM Cloud Foundry

### Prerequisites
- IBM Cloud CLI
- Cloud Foundry CLI

### Setup

```bash
# Install IBM Cloud CLI
curl -fsSL https://clis.cloud.ibm.com/install/linux | sh

# Login to IBM Cloud
ibmcloud login

# Target Cloud Foundry
ibmcloud target --cf
```

### Create `manifest.yml`

```yaml
applications:
- name: salary-prediction-app
  memory: 512M
  disk_quota: 1G
  instances: 1
  buildpacks:
    - python_buildpack
  env:
    FLASK_ENV: production
    IBM_CLOUD_API_KEY: ((api-key))
    WML_INSTANCE_ID: ((wml-instance-id))
    WATSON_ASSISTANT_API_KEY: ((assistant-api-key))
    WATSON_ASSISTANT_URL: ((assistant-url))
    WATSON_ASSISTANT_ID: ((assistant-id))
```

### Deploy

```bash
# Push to Cloud Foundry
cf push salary-prediction-app

# Check status
cf apps

# View logs
cf logs salary-prediction-app --recent
```

## 🐳 Docker Deployment

### Create Dockerfile

```dockerfile
FROM python:3.9-slim

WORKDIR /app

# Copy requirements first for better caching
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Copy application code
COPY . .

# Create non-root user
RUN useradd -m -u 1001 appuser && chown -R appuser:appuser /app
USER appuser

# Expose port
EXPOSE 5000

# Health check
HEALTHCHECK --interval=30s --timeout=30s --start-period=5s --retries=3 \
  CMD curl -f http://localhost:5000/health || exit 1

# Run application
CMD ["python", "web_app/app.py"]
```

### Build and Run

```bash
# Build image
docker build -t salary-prediction:latest .

# Run container
docker run -d \
  --name salary-prediction \
  -p 5000:5000 \
  --env-file .env \
  salary-prediction:latest

# Check logs
docker logs salary-prediction

# Stop container
docker stop salary-prediction
```

### Docker Compose

Create `docker-compose.yml`:

```yaml
version: '3.8'

services:
  app:
    build: .
    ports:
      - "5000:5000"
    env_file:
      - .env
    environment:
      - FLASK_ENV=production
    restart: unless-stopped
    healthcheck:
      test: ["CMD", "curl", "-f", "http://localhost:5000/health"]
      interval: 30s
      timeout: 10s
      retries: 3

  nginx:
    image: nginx:alpine
    ports:
      - "80:80"
    volumes:
      - ./nginx.conf:/etc/nginx/nginx.conf
    depends_on:
      - app
    restart: unless-stopped
```

Run with:

```bash
docker-compose up -d
```

## 🏭 Production Deployment

### Security Considerations

1. **Environment Variables**: Use secret management systems
2. **HTTPS**: Enable SSL/TLS certificates
3. **Authentication**: Implement proper auth mechanisms
4. **Rate Limiting**: Add API rate limiting
5. **Monitoring**: Setup logging and monitoring

### Performance Optimization

1. **Caching**: Implement Redis for model caching
2. **Load Balancing**: Use multiple app instances
3. **CDN**: Static asset delivery
4. **Database**: Use proper database for production data

### Monitoring Setup

#### Application Monitoring

```python
# Add to app.py
import logging
from flask import request
import time

@app.before_request
def log_request_info():
    logger.info('Request: %s %s', request.method, request.url)

@app.after_request
def log_response_info(response):
    logger.info('Response: %s', response.status_code)
    return response
```

#### Health Checks

```python
@app.route('/health')
def health_check():
    return {
        'status': 'healthy',
        'timestamp': datetime.now().isoformat(),
        'version': '1.0.0',
        'services': {
            'watson_ml': check_watson_ml(),
            'assistant': check_assistant()
        }
    }
```

### Backup and Recovery

1. **Model Backup**: Regular model versioning
2. **Data Backup**: Backup training data
3. **Configuration Backup**: Save environment configs
4. **Recovery Plan**: Document recovery procedures

## 🔧 Environment-Specific Configurations

### Development
```env
FLASK_ENV=development
FLASK_DEBUG=True
LOG_LEVEL=DEBUG
```

### Staging
```env
FLASK_ENV=staging
FLASK_DEBUG=False
LOG_LEVEL=INFO
```

### Production
```env
FLASK_ENV=production
FLASK_DEBUG=False
LOG_LEVEL=WARNING
```

## 📊 Scaling Considerations

### Horizontal Scaling
- Multiple app instances
- Load balancer configuration
- Session management

### Vertical Scaling
- Increase memory allocation
- CPU optimization
- Database performance tuning

### Auto-scaling
```yaml
# Kubernetes HPA example
apiVersion: autoscaling/v2
kind: HorizontalPodAutoscaler
metadata:
  name: salary-prediction-hpa
spec:
  scaleTargetRef:
    apiVersion: apps/v1
    kind: Deployment
    name: salary-prediction
  minReplicas: 2
  maxReplicas: 10
  metrics:
  - type: Resource
    resource:
      name: cpu
      target:
        type: Utilization
        averageUtilization: 70
```

## 🚨 Troubleshooting

### Common Deployment Issues

1. **Port Conflicts**
   ```bash
   # Check port usage
   lsof -i :5000
   # Kill process if needed
   kill -9 <PID>
   ```

2. **Environment Variables Not Loading**
   ```bash
   # Check if .env file exists and is readable
   ls -la .env
   # Test environment loading
   python -c "from dotenv import load_dotenv; load_dotenv(); import os; print(os.getenv('IBM_CLOUD_API_KEY'))"
   ```

3. **Watson ML Connection Issues**
   ```bash
   # Test credentials
   python -c "from ibm_watson_machine_learning import APIClient; client = APIClient({'url': 'your_url', 'apikey': 'your_key'}); print(client.version)"
   ```

4. **Model Loading Errors**
   ```bash
   # Check if model files exist
   ls -la models/
   # Test model loading
   python -c "import joblib; model = joblib.load('models/best_model.pkl'); print('Model loaded successfully')"
   ```

### Log Analysis

```bash
# View application logs
tail -f app.log

# Filter error logs
grep "ERROR" app.log

# Monitor real-time logs
tail -f app.log | grep -E "(ERROR|WARNING)"
```

## 📈 Performance Monitoring

### Key Metrics to Track

1. **Response Time**: API endpoint latency
2. **Throughput**: Requests per second
3. **Error Rate**: Failed requests percentage
4. **Resource Usage**: CPU, memory, disk usage
5. **Model Performance**: Prediction accuracy over time

### Monitoring Tools

- **Application**: New Relic, Datadog
- **Infrastructure**: Prometheus, Grafana
- **Logs**: ELK Stack, Splunk
- **Uptime**: Pingdom, StatusPage

## 🔄 CI/CD Pipeline

### GitHub Actions Example

```yaml
name: Deploy to IBM Cloud

on:
  push:
    branches: [ main ]

jobs:
  deploy:
    runs-on: ubuntu-latest
    
    steps:
    - uses: actions/checkout@v2
    
    - name: Setup Python
      uses: actions/setup-python@v2
      with:
        python-version: 3.9
        
    - name: Install dependencies
      run: |
        pip install -r requirements.txt
        
    - name: Run tests
      run: |
        python -m pytest tests/
        
    - name: Deploy to IBM Cloud
      run: |
        ibmcloud login --apikey ${{ secrets.IBM_CLOUD_API_KEY }}
        ibmcloud target --cf
        cf push
```

---

For additional help with deployment, refer to:
- [IBM Cloud Documentation](https://cloud.ibm.com/docs)
- [Watson ML Deployment Guide](https://dataplatform.cloud.ibm.com/docs/content/wsj/analyze-data/ml-deploy-general.html)
- [Project Issues](https://github.com/your-username/salary-prediction/issues)