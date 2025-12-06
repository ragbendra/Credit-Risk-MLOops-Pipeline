"""
Model training pipeline with MLflow tracking.

This script:
- Trains 3 models (LogReg, RandomForest, XGBoost)
- Logs parameters, metrics, and models to MLflow
- Saves the best model based on F1 score

Usage:
    python src/models/train.py
"""

import pandas as pd
import numpy as np
import mlflow
import mlflow.sklearn
import joblib
import json
from pathlib import Path
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from xgboost import XGBClassifier
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, 
    f1_score, roc_auc_score, confusion_matrix
)
import warnings
warnings.filterwarnings("ignore")


# MLflow configuration
EXPERIMENT_NAME = "credit-risk-prediction"
MLFLOW_TRACKING_URI = "http://localhost:5000"  # Local MLflow server


def load_data(train_path: str = "data/processed/train.csv",
              test_path: str = "data/processed/test.csv"):
    """Load train and test datasets."""
    train = pd.read_csv(train_path)
    test = pd.read_csv(test_path)
    
    X_train = train.drop("risk", axis=1)
    y_train = train["risk"]
    X_test = test.drop("risk", axis=1)
    y_test = test["risk"]
    
    print(f"📂 Loaded data: Train={len(X_train)}, Test={len(X_test)}")
    return X_train, X_test, y_train, y_test


def evaluate_model(model, X_test, y_test) -> dict:
    """Calculate evaluation metrics."""
    y_pred = model.predict(X_test)
    y_prob = model.predict_proba(X_test)[:, 1]
    
    metrics = {
        "accuracy": accuracy_score(y_test, y_pred),
        "precision": precision_score(y_test, y_pred),
        "recall": recall_score(y_test, y_pred),
        "f1": f1_score(y_test, y_pred),
        "auc_roc": roc_auc_score(y_test, y_prob),
    }
    
    return metrics


def get_models() -> dict:
    """Return dictionary of models to train."""
    return {
        "LogisticRegression": LogisticRegression(
            max_iter=1000,
            random_state=42,
            class_weight="balanced"
        ),
        "RandomForest": RandomForestClassifier(
            n_estimators=100,
            max_depth=10,
            random_state=42,
            class_weight="balanced",
            n_jobs=-1
        ),
        "XGBoost": XGBClassifier(
            n_estimators=100,
            max_depth=5,
            learning_rate=0.1,
            random_state=42,
            scale_pos_weight=2.33,  # Ratio of negative to positive
            eval_metric="logloss",
            use_label_encoder=False
        ),
    }


def train_models(use_mlflow: bool = True):
    """
    Train all models and log to MLflow.
    
    Args:
        use_mlflow: Whether to log to MLflow server (set False for local testing)
    """
    print("\n" + "="*50)
    print("🤖 MODEL TRAINING PIPELINE")
    print("="*50 + "\n")
    
    # Load data
    X_train, X_test, y_train, y_test = load_data()
    
    # Setup MLflow
    if use_mlflow:
        try:
            mlflow.set_tracking_uri(MLFLOW_TRACKING_URI)
            mlflow.set_experiment(EXPERIMENT_NAME)
            print(f"📊 MLflow tracking URI: {MLFLOW_TRACKING_URI}")
        except Exception as e:
            print(f"⚠️  MLflow server not available: {e}")
            print("   Training without MLflow logging...")
            use_mlflow = False
    
    # Get models
    models = get_models()
    
    # Track best model
    best_model = None
    best_model_name = None
    best_f1 = 0
    all_metrics = {}
    
    # Train each model
    for name, model in models.items():
        print(f"\n🔄 Training {name}...")
        
        if use_mlflow:
            with mlflow.start_run(run_name=name):
                # Train
                model.fit(X_train, y_train)
                
                # Evaluate
                metrics = evaluate_model(model, X_test, y_test)
                all_metrics[name] = metrics
                
                # Log to MLflow
                mlflow.log_params(model.get_params())
                mlflow.log_metrics(metrics)
                mlflow.sklearn.log_model(model, "model")
                
                print(f"   ✅ F1={metrics['f1']:.3f}, AUC={metrics['auc_roc']:.3f}")
        else:
            # Train without MLflow
            model.fit(X_train, y_train)
            metrics = evaluate_model(model, X_test, y_test)
            all_metrics[name] = metrics
            print(f"   ✅ F1={metrics['f1']:.3f}, AUC={metrics['auc_roc']:.3f}")
        
        # Track best
        if metrics["f1"] > best_f1:
            best_f1 = metrics["f1"]
            best_model = model
            best_model_name = name
    
    # Save best model
    models_dir = Path("models")
    models_dir.mkdir(parents=True, exist_ok=True)
    
    joblib.dump(best_model, models_dir / "best_model.pkl")
    print(f"\n🏆 Best model: {best_model_name} (F1={best_f1:.3f})")
    print(f"💾 Saved to models/best_model.pkl")
    
    # Save metrics for DVC
    reports_dir = Path("reports")
    reports_dir.mkdir(parents=True, exist_ok=True)
    
    with open(reports_dir / "metrics.json", "w") as f:
        json.dump({
            "best_model": best_model_name,
            "best_f1": best_f1,
            "all_models": all_metrics
        }, f, indent=2)
    
    print(f"📊 Metrics saved to reports/metrics.json")
    print("\n✅ Training complete!\n")
    
    return best_model, all_metrics


if __name__ == "__main__":
    # Try with MLflow, fall back to local if server not running
    train_models(use_mlflow=True)
