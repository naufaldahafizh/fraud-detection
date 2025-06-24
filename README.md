# Credit Card Fraud Detection

This project focuses on detecting fraudulent transactions using machine learning techniques.

## Goal

Predict whether a transaction is fraudulent or not using features such as amount, time, and PCA components.

## Dataset

The dataset is not included in this repository due to its size (>100MB).  
You can download it from the [Kaggle Credit Card Fraud Dataset](https://www.kaggle.com/datasets/mlg-ulb/creditcardfraud).

Place the file as: `fraud-detection/data/creditcard.csv`

## Project Structure
```
fraud-detection/
│
├── notebooks/ # EDA, modeling, evaluation
│ └── fraud_detection.ipynb
│
├── src/ # Reusable modules (preprocessing, modeling)
│ ├── preprocessing.py
│ └── modeling.py
│
├── sql/ # (Optional) SQL exploration queries
│ ├── 01_exploration.sql
│ ├── 02_fraud_by_time.sql
│ ├── 03_user_behavior.sql
│ └── sql_exploration.ipynb
|
├── models/ # Saved model and preprocessing pipeline
│ ├── model.pkl
│ └── preprocessing.pkl
│
├── api/ # FastAPI application
│ └── app.py
│
├── logs/ # Inference logs
│ └── inference.log
│
├── Dockerfile # For Docker deployment
├── requirements.txt # Python dependencies
└── .github/workflows/ # CI/CD using GitHub Actions
  └── docker-build.yml
  └── python-ci.yml
```

---

## Project Workflow

### A. Data Analysis & Modeling (Python)

- Performed EDA: class imbalance, amount distribution, time patterns
- Feature Engineering: normalized amount, resampling with SMOTE
- Modeling: Logistic Regression, Random Forest, XGBoost
- Evaluation: ROC AUC, Precision-Recall Curve

📍 Location: `notebooks/fraud_detection.ipynb`

### B. API Deployment (FastAPI + Docker)

- Built REST API with FastAPI
- Exposes `/predict` endpoint for real-time fraud detection
- Includes input validation with Pydantic
- Dockerized with `Dockerfile` and tested locally

📍 Try locally:

```bash
docker build -t fraud-api .
docker run -p 8000:8000 fraud-api
Then visit: http://localhost:8000/docs
```

### C. Monitoring & Logging
- Logs each inference request (input + prediction) to logs/inference.log
- Future improvements: Prometheus/Grafana, log ingestion pipeline

### D. CI/CD with GitHub Actions
- Workflow: .github/workflows/docker-build.yml
- Triggers on every push to main
- Steps:
    - Checkout code
    - Install Python + dependencies
    - Test FastAPI import
    - Build Docker image


## Results
- Achieved high ROC-AUC score (above 0.97) using RandomForest and XGBoost
- Demonstrated full MLOps lifecycle from data → model → API → logging


License
MIT License





