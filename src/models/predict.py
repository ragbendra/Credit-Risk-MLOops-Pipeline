"""
Prediction module for inference.

This module loads the trained model and provides prediction functionality.
"""

import joblib
import numpy as np
import pandas as pd
from pathlib import Path


class CreditRiskPredictor:
    """Credit risk prediction class."""
    
    def __init__(self, model_path: str = "models/best_model.pkl"):
        """
        Initialize the predictor.
        
        Args:
            model_path: Path to the trained model file.
        """
        self.model_path = Path(model_path)
        self.model = None
        self.load_model()
    
    def load_model(self):
        """Load the trained model from disk."""
        if not self.model_path.exists():
            raise FileNotFoundError(f"Model not found at {self.model_path}")
        
        self.model = joblib.load(self.model_path)
        print(f"✅ Model loaded from {self.model_path}")
    
    def predict(self, features: np.ndarray) -> dict:
        """
        Make a prediction for a single sample.
        
        Args:
            features: Feature array of shape (1, n_features)
            
        Returns:
            Dictionary with prediction, probability, and risk score.
        """
        if self.model is None:
            raise RuntimeError("Model not loaded")
        
        # Ensure 2D array
        if features.ndim == 1:
            features = features.reshape(1, -1)
        
        # Predict
        prob = self.model.predict_proba(features)[0][1]  # Probability of bad risk
        prediction = "bad" if prob > 0.5 else "good"
        
        return {
            "prediction": prediction,
            "probability": round(float(prob), 4),
            "risk_score": int(prob * 100),
        }
    
    def predict_batch(self, features: np.ndarray) -> list[dict]:
        """
        Make predictions for multiple samples.
        
        Args:
            features: Feature array of shape (n_samples, n_features)
            
        Returns:
            List of prediction dictionaries.
        """
        results = []
        for i in range(len(features)):
            result = self.predict(features[i])
            results.append(result)
        return results


# Convenience function
def load_predictor(model_path: str = "models/best_model.pkl") -> CreditRiskPredictor:
    """Load and return a predictor instance."""
    return CreditRiskPredictor(model_path)
