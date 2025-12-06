# Credit Risk MLOps Pipeline: My First End-to-End Project

## Overview
I built a credit risk prediction system that takes loan applicant data and returns a risk score. The entire system is deployed online and available via API. Everything runs on free-tier services because as a fresher, I wanted to prove I could build production systems without spending money.

---

## 1. Why This Project

### The Gap I Noticed
Most ML projects I see from other freshers stop at Jupyter notebooks. When I looked at job descriptions, companies wanted experience with deployment, monitoring, and maintaining models—not just building them. 

This project is my attempt to bridge that gap. It answers the question: "Can you actually put a model into production?"

### The Real Problem
Manual credit assessment is slow and inconsistent. While my solution uses a simple dataset (UCI German Credit), it demonstrates the infrastructure needed for a real system.

### Tech Stack Choices
I evaluated several options before settling on my current stack:

**FastAPI over Flask:** FastAPI generates automatic documentation (Swagger UI), which is useful for demos. It also has built-in data validation with Pydantic.

**Render over Heroku:** Heroku removed its free tier. Render offers 750 free hours monthly, enough to keep an API running 24/7.

**scikit-learn + XGBoost over deep learning:** The dataset is tabular and small. Traditional ML works well here and doesn't require GPU resources.

**MLflow over Weights & Biases:** MLflow is open-source and free. I can host it myself instead of relying on external services.

Each choice involved tradeoffs. For example, Render has cold starts (5-8 seconds after inactivity), but the free tier made it worth accepting this limitation.

---

## 2. What I Built

### The System Architecture

```
User → FastAPI (Render) → ML Model → JSON Response
         ↑           ↓           ↓
     Pydantic    MLflow     Monitoring
    Validation  Tracking   (Evidently)
```

The flow is straightforward:
1. A user sends applicant data to the API
2. Pydantic validates the data format
3. The trained model makes a prediction
4. A JSON response returns the risk score

### Key Features
- **Public API**: Accessible at `https://credit-risk-mlops.onrender.com`
- **Automatic documentation**: Swagger UI at `/docs`
- **Model versioning**: MLflow tracks every experiment
- **Basic monitoring**: Drift detection with Evidently
- **CI/CD pipeline**: Automated testing and deployment via GitHub Actions

---

## 3. Technical Decisions & Tradeoffs

### Data Validation: Simple Over Complex
I used Pydantic for data validation instead of more complex tools like Great Expectations. Why? Pydantic integrates with FastAPI and validates each API request automatically. For a project this size, it's sufficient. If I were processing batch data from multiple sources, I'd consider Great Expectations.

### Monitoring: Practical Over Perfect
I set up Evidently to generate drift reports manually, not real-time dashboards with Prometheus/Grafana. The reason is simple: real-time monitoring requires more infrastructure than I could set up on free tiers. The weekly report still lets me check if the model is behaving unexpectedly.

### Model Explainability: Pre-computed Over Real-time
I generate SHAP plots during training and save them as images. Adding SHAP calculations to each API request would slow responses by 200-300ms. Since interviewers usually ask "Can you explain your model?" not "Can you explain this specific prediction?", the global plots work fine.

### Database Strategy: Backup Over Permanent
Render's free PostgreSQL database expires after 90 days. Instead of paying for permanent storage, I wrote a backup script that exports data to Google Drive weekly via DVC. If the database expires, I can restore from backup.

---

## 4. Challenges & Solutions

### Challenge 1: Feature Order Mismatch
**Problem:** The model gave random predictions (always ~50% probability) regardless of input.

**Debugging:** 
1. Verified the model loaded correctly ✓
2. Checked input values were correct ✓
3. Printed the feature array before prediction and discovered the issue: features were in alphabetical order, but the model expected training order.

**Solution:** I stopped converting the Pydantic model to dict and instead built the feature array explicitly, matching the training column order exactly.

### Challenge 2: Render Cold Starts
**Problem:** First request after 15+ minutes of inactivity took 10-15 seconds.

**Root Cause:** Render spins down free-tier containers during inactivity.

**Solution:** 
1. Used a smaller Docker base image (`python:3.11-slim`)
2. Pre-installed dependencies in Docker layers
3. Accepted this as a free-tier limitation

### Challenge 3: Pydantic Version Confusion
**Problem:** Deprecation warnings during testing.

**Cause:** AI-generated code used Pydantic v1 syntax. I had v2 installed.

**Solution:** Read the Pydantic migration guide and updated the syntax. Lesson: always check library versions when using code from tutorials or AI.

### Challenge 4: Dataset Encoding
**Problem:** The UCI German Credit dataset uses cryptic codes like "A11", "A12".

**Solution:** Found the data dictionary, mapped codes to meanings, but kept encoded values in the API. Users send codes; the model expects codes. I documented the mapping for clarity.

---

## 5. What Works Well

### Model Performance
- **Best model**: XGBoost (76.5% accuracy, 63.2% F1-score)
- **Training time**: ~30 seconds locally
- **Inference time**: <200ms (after cold start)

### API Performance
- **Latency**: 45ms median, 120ms 95th percentile
- **Throughput**: ~80 requests/second in load tests
- **Uptime**: >95% (Render free tier occasionally restarts)

### Development Experience
- **One-command setup**: `docker-compose up` runs the full stack locally
- **Automated tests**: GitHub Actions runs tests on every push
- **Easy deployment**: Push to main branch → automatic deployment to Render

---

## 6. What Could Be Better

### Known Limitations
1. **Cold starts**: 5-8 second delay after inactivity (Render free tier limitation)
2. **Manual monitoring**: Must run script to generate drift reports
3. **Database expiry**: PostgreSQL database disappears after 90 days
4. **No authentication**: API is publicly accessible (fine for a demo)

### If I Had More Time/Budget
1. **Add authentication**: API keys for security
2. **Real-time monitoring**: Prometheus + Grafana dashboards
3. **Automated retraining**: Trigger model retraining when drift detected
4. **Multiple environments**: Staging vs production deployment

---

## 7. What I Learned

### Technical Insights
1. **Feature ordering matters more than I expected** between training and inference
2. **Container optimization** affects cold start times significantly
3. **MLflow is simpler to set up** than I anticipated
4. **FastAPI's async support** doesn't matter much for CPU-bound ML models

### Process Insights
1. **Write tests early** - they caught several bugs before deployment
2. **Document decisions as you go** - easier than reconstructing later
3. **Start simple, then add complexity** - my first working version was much simpler
4. **Check library versions** - especially when following online tutorials

### About Production ML
1. **Deployment is just the beginning** - monitoring matters as much as building
2. **Free tiers have real limitations** - cold starts, database expiry
3. **Simple solutions often suffice** for small-scale projects
4. **Reproducibility requires discipline** - DVC and MLflow help significantly

---

## 8. Project Structure

```
credit-risk-mlops/
├── src/                    # Source code
│   ├── api/                # FastAPI application
│   ├── data/               # Data downloading & preprocessing
│   └── models/             # Training & inference logic
├── tests/                  # Pytest tests
├── scripts/                # Utility scripts
├── docker/                 # Docker configuration
├── .github/workflows/      # CI/CD pipeline
├── pyproject.toml          # Dependencies
├── docker-compose.yml      # Local development
└── README.md               # Setup instructions
```

The structure follows standard Python packaging conventions. Each module has a clear responsibility.

---

## 9. How to Run It

### Local Development
```bash
# Clone the repository
git clone https://github.com/yourusername/credit-risk-mlops

# Start all services
docker-compose up

# Train the model
docker exec -it credit-risk-app python src/models/train.py

# Run tests
docker exec -it credit-risk-app pytest
```

### Using the API
```bash
# Get health status
curl https://credit-risk-mlops.onrender.com/health

# Make a prediction
curl -X POST https://credit-risk-mlops.onrender.com/predict \
  -H "Content-Type: application/json" \
  -d '{"age": 35, "amount": 5000, "duration": 24, ...}'
```

---

## 10. For Interview Discussion

### What I'd Show
- Live API demo - make a real prediction
- Swagger documentation - show auto-generated API docs
- MLflow UI - demonstrate experiment tracking
- GitHub Actions logs - show CI/CD pipeline working

### Questions I Can Answer
- "How does your system handle bad input?" → Pydantic validation
- "What if the model performance drops?" → Drift detection with Evidently
- "How do you update the model?" → MLflow versioning + Render redeploy
- "What are the system's limitations?" → Cold starts, database expiry, manual monitoring

### What This Project Demonstrates
- End-to-end thinking - from data to deployed API
- Practical constraints - working within free-tier limitations
- Problem-solving - debugging and fixing real issues
- Learning ability - picking up new tools and concepts

---

## Final Thoughts

This is my first complete MLOps project. It's not perfect—there are limitations I'd address in a production system. But it shows I can take a model from a notebook to a working API with monitoring, versioning, and automation.

The most valuable part wasn't the final product, but the process: figuring out why features needed exact ordering, optimizing Docker builds to reduce cold starts, and learning how to track experiments properly.

I'm sharing this not as a finished solution, but as evidence that I can build and learn—and that I understand what it takes to put machine learning into practice.