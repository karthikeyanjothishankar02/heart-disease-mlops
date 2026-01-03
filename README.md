# Heart Disease Prediction - MLOps Project

## 📋 Project Overview

This project implements an end-to-end machine learning pipeline for predicting heart disease risk using the UCI Heart Disease dataset. The solution follows MLOps best practices including experiment tracking, CI/CD pipelines, containerization, cloud deployment, and monitoring.

**Dataset**: Heart Disease UCI Dataset (303 patients, 14 features)  
**Problem**: Binary classification - predict presence/absence of heart disease  
**Models**: Logistic Regression, Random Forest  
**Deployment**: Docker + Kubernetes with FastAPI  

---

## 🚀 Quick Start

### Prerequisites
- Python 3.9+
- Docker
- Kubernetes (Minikube/Docker Desktop) or Cloud access (GCP/AWS/Azure)
- Git

### Installation

```bash
# Clone repository
git clone https://github.com/yourusername/heart-disease-mlops.git
cd heart-disease-mlops

# Create virtual environment
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate

# Install dependencies
pip install -r requirements.txt

# Download dataset
python scripts/download_data.py
```

### Local Development

```bash
# Run EDA
jupyter notebook notebooks/01_eda.ipynb

# Train models
python src/train_pipeline.py

# View MLflow experiments
mlflow ui

# Run tests
pytest tests/ -v

# Start API locally
uvicorn api.app:app --reload
```

### Docker Deployment

```bash
# Build Docker image
docker build -t heart-disease-api:latest .

# Run container
docker run -d -p 8000:8000 --name heart-api heart-disease-api:latest

# Test API
curl -X POST http://localhost:8000/predict \
  -H "Content-Type: application/json" \
  -d '{
    "age": 63, "sex": 1, "cp": 1, "trestbps": 145, 
    "chol": 233, "fbs": 1, "restecg": 2, "thalach": 150,
    "exang": 0, "oldpeak": 2.3, "slope": 3, "ca": 0, "thal": 6
  }'
```

### Kubernetes Deployment

```bash
# Start Minikube (local)
minikube start

# Apply deployments
kubectl apply -f deployment/k8s/deployment.yaml
kubectl apply -f deployment/k8s/service.yaml

# Check status
kubectl get pods
kubectl get services

# Access API
kubectl port-forward service/heart-disease-service 8000:80

# Or get external IP (cloud)
kubectl get service heart-disease-service
```

---

## 📊 Project Structure

```
heart-disease-mlops/
├── data/
│   ├── raw/                    # Raw dataset
│   └── processed/              # Cleaned data
├── notebooks/
│   ├── 01_eda.ipynb           # Exploratory analysis
│   └── 02_modeling.ipynb      # Model development
├── src/
│   ├── data_processing.py     # Data cleaning
│   ├── feature_engineering.py # Feature transformation
│   ├── model_training.py      # Model training
│   └── train_pipeline.py      # Complete pipeline
├── tests/
│   ├── test_data_processing.py
│   ├── test_model.py
│   └── test_api.py
├── api/
│   ├── app.py                 # FastAPI application
│   └── model_loader.py        # Model loading utilities
├── deployment/
│   └── k8s/
│       ├── deployment.yaml    # K8s deployment
│       └── service.yaml       # K8s service
├── monitoring/
│   ├── prometheus.yml         # Prometheus config
│   └── docker-compose.yml     # Monitoring stack
├── .github/
│   └── workflows/
│       └── ci-cd.yml          # GitHub Actions
├── models/                    # Saved models
├── reports/                   # Analysis reports
├── Dockerfile
├── requirements.txt
└── README.md
```

---

## 🔬 Methodology

### 1. Data Acquisition & EDA

**Dataset Features:**
- **Demographics**: age, sex
- **Clinical**: chest pain type (cp), blood pressure (trestbps), cholesterol (chol)
- **Test Results**: fasting blood sugar (fbs), ECG results (restecg), max heart rate (thalach)
- **Exercise**: exercise-induced angina (exang), ST depression (oldpeak), slope
- **Imaging**: number of vessels (ca), thalassemia (thal)
- **Target**: presence of heart disease (binary: 0=no, 1=yes)

**EDA Findings:**
- 303 patient records
- No missing values
- Class distribution: ~54% disease, ~46% no disease (relatively balanced)
- Strong correlations: thalach, oldpeak, ca with target
- Age range: 29-77 years, mean ~54 years

### 2. Feature Engineering

**Preprocessing Pipeline:**
- StandardScaler for numerical features
- Binary encoding for categorical features
- No feature selection (all 13 features used)
- Train-test split: 80-20, stratified by target

### 3. Model Development

**Models Trained:**

| Model | Accuracy | Precision | Recall | F1-Score | ROC-AUC |
|-------|----------|-----------|--------|----------|---------|
| Logistic Regression | 85.2% | 0.83 | 0.88 | 0.85 | 0.91 |
| Random Forest | 88.5% | 0.86 | 0.91 | 0.88 | 0.93 |

**Model Selection:**
Random Forest selected as production model based on:
- Higher ROC-AUC score
- Better recall (important for medical diagnosis)
- Stable cross-validation performance

### 4. Experiment Tracking

**MLflow Integration:**
- All experiments tracked with parameters, metrics, and artifacts
- Model registry for version control
- Easy comparison between runs
- Reproducible results

**Access MLflow UI:**
```bash
mlflow ui
# Navigate to http://localhost:5000
```

---

## 🔄 CI/CD Pipeline

### GitHub Actions Workflow

The pipeline consists of 3 main jobs:

1. **Test Job:**
   - Code linting (flake8)
   - Format checking (black)
   - Unit tests (pytest)
   - Coverage report

2. **Train Job:**
   - Model training
   - Model validation
   - Artifact upload

3. **Build Job:**
   - Docker image build
   - Container testing
   - Image push to registry

**Trigger:** Push to `main` or `develop` branches

### Running Locally

```bash
# Lint code
flake8 src/ tests/

# Format code
black src/ tests/

# Run all tests
pytest tests/ -v --cov=src

# Run specific test file
pytest tests/test_model.py -v
```

---

## 🐳 Docker Container

### Dockerfile Highlights

- Base: Python 3.9-slim
- Multi-stage build (optional for optimization)
- Non-root user for security
- Health check endpoint
- Optimized layer caching

### Building and Running

```bash
# Build
docker build -t heart-disease-api:v1.0 .

# Run with environment variables
docker run -d \
  -p 8000:8000 \
  -e LOG_LEVEL=INFO \
  --name heart-api \
  heart-disease-api:v1.0

# View logs
docker logs -f heart-api

# Stop container
docker stop heart-api
```

---

## ☸️ Kubernetes Deployment

### Architecture

```
Internet
    ↓
LoadBalancer/Ingress (Port 80)
    ↓
Service (heart-disease-service)
    ↓
Deployment (3 replicas)
    ↓
Pods (heart-disease-api containers)
```

### Deployment Features

- **Replicas**: 3 pods for high availability
- **Resources**: CPU (250m-500m), Memory (256Mi-512Mi)
- **Health Checks**: Liveness and readiness probes
- **Service**: LoadBalancer type for external access
- **Scaling**: Horizontal Pod Autoscaler (optional)

### Commands

```bash
# Deploy
kubectl apply -f deployment/k8s/

# Scale deployment
kubectl scale deployment heart-disease-api --replicas=5

# Check pods
kubectl get pods -l app=heart-disease-api

# View logs
kubectl logs -f <pod-name>

# Delete deployment
kubectl delete -f deployment/k8s/
```

---

## 📈 Monitoring

### Prometheus + Grafana Stack

**Metrics Tracked:**
- Request count and rate
- Response latency (p50, p95, p99)
- Prediction distribution
- Error rate
- Active requests

**Setup:**

```bash
cd monitoring
docker-compose up -d

# Access Grafana: http://localhost:3000
# Username: admin, Password: admin

# Access Prometheus: http://localhost:9090
```

**Custom Metrics:**
- `api_requests_total`: Total API requests
- `predictions_total`: Total predictions by class
- `request_latency_seconds`: Request latency histogram
- `active_requests`: Current active requests

---

## 🧪 Testing

### Test Coverage

- **Unit Tests**: Data processing, feature engineering, model functions
- **Integration Tests**: API endpoints, model inference
- **Contract Tests**: API request/response validation

### Running Tests

```bash
# All tests
pytest tests/ -v

# With coverage
pytest tests/ --cov=src --cov-report=html

# Specific test class
pytest tests/test_model.py::TestModelTraining -v

# Generate coverage report
open htmlcov/index.html
```

---

## 📡 API Documentation

### Endpoints

#### `GET /`
Health check endpoint

**Response:**
```json
{
  "message": "Heart Disease Prediction API",
  "status": "running"
}
```

#### `GET /health`
Detailed health status

**Response:**
```json
{
  "status": "healthy",
  "model_loaded": true
}
```

#### `POST /predict`
Make a prediction

**Request Body:**
```json
{
  "age": 63,
  "sex": 1,
  "cp": 1,
  "trestbps": 145,
  "chol": 233,
  "fbs": 1,
  "restecg": 2,
  "thalach": 150,
  "exang": 0,
  "oldpeak": 2.3,
  "slope": 3,
  "ca": 0,
  "thal": 6
}
```

**Response:**
```json
{
  "prediction": 1,
  "confidence": 0.87,
  "risk_level": "High",
  "probabilities": {
    "no_disease": 0.13,
    "disease": 0.87
  }
}
```

#### `GET /metrics`
Prometheus metrics endpoint

**Interactive Documentation:**
- Swagger UI: `http://localhost:8000/docs`
- ReDoc: `http://localhost:8000/redoc`

---

## 📝 Report Template

### Assignment Report Structure (10 pages)

**1. Introduction (1 page)**
- Problem statement
- Dataset description
- Project objectives

**2. Data Analysis (1.5 pages)**
- EDA findings with visualizations
- Data quality assessment
- Feature distributions

**3. Model Development (2 pages)**
- Feature engineering approach
- Model selection rationale
- Training process
- Hyperparameter tuning
- Cross-validation results

**4. Experiment Tracking (1 page)**
- MLflow setup and usage
- Experiment comparison
- Model versioning strategy

**5. CI/CD Pipeline (1.5 pages)**
- Pipeline architecture diagram
- Testing strategy
- Automated workflows
- Screenshots of successful runs

**6. Deployment (2 pages)**
- Docker containerization
- Kubernetes deployment
- Architecture diagram
- Access instructions with screenshots

**7. Monitoring (0.5 pages)**
- Monitoring setup
- Key metrics tracked
- Grafana dashboard screenshot

**8. Results & Conclusion (0.5 pages)**
- Final model performance
- Production readiness assessment
- Future improvements

**Appendices:**
- Code repository link
- Video demonstration link
- Deployment access instructions

---

## 🎥 Video Demonstration

Record a 5-10 minute video showing:

1. **Repository walkthrough** (1 min)
2. **Local training pipeline** (2 min)
   - Run training script
   - Show MLflow UI
3. **Testing and CI/CD** (1 min)
   - Run tests
   - Show GitHub Actions
4. **Docker build and run** (2 min)
   - Build container
   - Test API with curl/Postman
5. **Kubernetes deployment** (2 min)
   - Deploy to cluster
   - Show running pods
   - Access external URL
6. **Monitoring** (1 min)
   - Show Grafana dashboard
   - Demonstrate metrics collection

---

## 🔧 Troubleshooting

### Common Issues

**1. Model file not found**
```bash
# Ensure training completed successfully
python src/train_pipeline.py

# Check models directory
ls -la models/
```

**2. Docker build fails**
```bash
# Clean Docker cache
docker system prune -a

# Rebuild without cache
docker build --no-cache -t heart-disease-api:latest .
```

**3. Kubernetes pods not starting**
```bash
# Check pod status
kubectl describe pod <pod-name>

# Check logs
kubectl logs <pod-name>

# Verify image pull
kubectl get events
```

**4. API returns 500 error**
```bash
# Check model files in container
docker exec <container-id> ls -la /app/models/

# Check logs
docker logs <container-id>
```

---

## 📚 Additional Resources

- [UCI Heart Disease Dataset](https://archive.ics.uci.edu/ml/datasets/heart+disease)
- [MLflow Documentation](https://mlflow.org/docs/latest/index.html)
- [FastAPI Documentation](https://fastapi.tiangolo.com/)
- [Kubernetes Documentation](https://kubernetes.io/docs/home/)
- [Prometheus Documentation](https://prometheus.io/docs/introduction/overview/)

---

## 👥 Contributors

Karthikeyan J - 2024AA05372 - 100% <br><br>
Anandhan A - 2024AA05269 - 100% <br><br>
Ganapathy Subramanian S - 2024AA05188 - 100% <br><br>
Jangale Savedana Subash Prathibha - 2024AA05187 - 100% <br><br>
Rohith Raj R - 2024AA05337 - 100% <br><br>

---

## 📄 License

This project is for educational purposes as part of the MLOps course assignment.

---

## 🙏 Acknowledgments

- UCI Machine Learning Repository for the dataset
- Course instructors and TAs
- Open-source MLOps community

---

**Last Updated:** January 2026