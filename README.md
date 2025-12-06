# Credit Risk MLOps Pipeline

[![CI/CD](https://github.com/YOUR_USERNAME/credit-risk-mlops/actions/workflows/main.yml/badge.svg)](https://github.com/YOUR_USERNAME/credit-risk-mlops/actions)

Production-ready MLOps pipeline for credit risk prediction using the UCI German Credit Dataset.

**Live Demo:** `https://credit-risk-mlops.onrender.com` (once deployed)

## Features

- 🤖 **3 ML Models**: Logistic Regression, Random Forest, XGBoost
- 📊 **MLflow Tracking**: Experiment tracking with metrics and artifacts
- 🔄 **DVC Pipeline**: Reproducible data and model versioning
- 🚀 **FastAPI**: Production-ready API with validation
- 🐳 **Docker**: One-command deployment
- ✅ **CI/CD**: GitHub Actions for testing and deployment
- 📈 **Monitoring**: Evidently drift reports

## Quick Start

### 1. Install Dependencies

```bash
# Using pip
pip install -e .

# Or using uv (faster)
pip install uv
uv pip install --system -e ".[dev]"
```

### 2. Run the Pipeline

```bash
# Download data
python src/data/download.py

# Preprocess
python src/data/preprocess.py

# Train models
python src/models/train.py
```

### 3. Start the API

```bash
uvicorn src.api.main:app --reload
```

### 4. Test the API

```bash
# Health check
curl http://localhost:8000/health

# Prediction
curl -X POST http://localhost:8000/predict \
  -H "Content-Type: application/json" \
  -d '{
    "status": 1, "duration": 24, "amount": 5000,
    "credit_history": 2, "purpose": 3, "savings": 1,
    "employment": 2, "installment_rate": 2, "personal_status": 2,
    "other_debtors": 0, "residence": 3, "property": 1,
    "age": 35, "other_plans": 0, "housing": 1,
    "existing_credits": 1, "job": 2, "dependents": 1,
    "telephone": 1, "foreign_worker": 0
  }'
```

## Docker Deployment

```bash
# Start all services (API + MLflow + PostgreSQL)
docker-compose -f docker/docker-compose.yml up -d

# View logs
docker-compose -f docker/docker-compose.yml logs -f api

# Stop
docker-compose -f docker/docker-compose.yml down
```

## Project Structure

```
credit-risk-mlops/
├── src/
│   ├── data/           # Download & preprocessing
│   ├── models/         # Training & prediction
│   └── api/            # FastAPI application
├── tests/              # Unit & integration tests
├── scripts/            # Monitoring & backup
├── docker/             # Docker configuration
├── notebooks/          # EDA & analysis
└── reports/            # Drift reports
```

## API Endpoints

| Endpoint | Method | Description |
|----------|--------|-------------|
| `/health` | GET | Health check |
| `/predict` | POST | Credit risk prediction |
| `/docs` | GET | Swagger documentation |

## Make Commands

```bash
make help         # Show all commands
make install      # Install dependencies
make pipeline     # Run full DVC pipeline
make run          # Start API server
make test         # Run tests
make docker-up    # Start Docker services
make report       # Generate drift report
```

## Tech Stack

| Component | Technology |
|-----------|------------|
| ML Framework | scikit-learn, XGBoost |
| API | FastAPI, Pydantic |
| Experiment Tracking | MLflow |
| Data Versioning | DVC |
| CI/CD | GitHub Actions |
| Deployment | Docker, Render.com |
| Monitoring | EvidentlyML |

## Cost: ₹0/month

All services use free tiers:
- **Render.com**: 750 hours/month
- **GitHub Actions**: 2000 min/month
- **Google Drive (DVC)**: 15 GB

## License

MIT
