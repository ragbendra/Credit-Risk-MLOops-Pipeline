"""
Tests for FastAPI endpoints.

Run with: pytest tests/test_api.py -v
"""

import pytest
from fastapi.testclient import TestClient
import sys
from pathlib import Path

# Add src to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from src.api.main import app


client = TestClient(app)


class TestHealthEndpoint:
    """Tests for /health endpoint."""
    
    def test_health_returns_200(self):
        """Health check should return 200."""
        response = client.get("/health")
        assert response.status_code == 200
    
    def test_health_response_structure(self):
        """Health response should have correct structure."""
        response = client.get("/health")
        data = response.json()
        
        assert "status" in data
        assert "model_loaded" in data
        assert "version" in data
        assert data["status"] == "healthy"


class TestPredictEndpoint:
    """Tests for /predict endpoint."""
    
    @pytest.fixture
    def valid_request(self):
        """Valid prediction request."""
        return {
            "status": 1,
            "duration": 24,
            "amount": 5000,
            "credit_history": 2,
            "purpose": 3,
            "savings": 1,
            "employment": 2,
            "installment_rate": 2,
            "personal_status": 2,
            "other_debtors": 0,
            "residence": 3,
            "property": 1,
            "age": 35,
            "other_plans": 0,
            "housing": 1,
            "existing_credits": 1,
            "job": 2,
            "dependents": 1,
            "telephone": 1,
            "foreign_worker": 0,
        }
    
    def test_predict_valid_request(self, valid_request):
        """Valid request should return prediction (or 503 if no model)."""
        response = client.post("/predict", json=valid_request)
        
        # Either 200 (model loaded) or 503 (no model)
        assert response.status_code in [200, 503]
        
        if response.status_code == 200:
            data = response.json()
            assert "prediction" in data
            assert "probability" in data
            assert "risk_score" in data
            assert data["prediction"] in ["good", "bad"]
            assert 0 <= data["probability"] <= 1
            assert 0 <= data["risk_score"] <= 100
    
    def test_predict_invalid_age_too_young(self, valid_request):
        """Age < 18 should return 422."""
        valid_request["age"] = 10
        response = client.post("/predict", json=valid_request)
        assert response.status_code == 422
    
    def test_predict_invalid_age_too_old(self, valid_request):
        """Age > 100 should return 422."""
        valid_request["age"] = 150
        response = client.post("/predict", json=valid_request)
        assert response.status_code == 422
    
    def test_predict_invalid_amount_negative(self, valid_request):
        """Negative amount should return 422."""
        valid_request["amount"] = -1000
        response = client.post("/predict", json=valid_request)
        assert response.status_code == 422
    
    def test_predict_invalid_duration(self, valid_request):
        """Duration > 72 should return 422."""
        valid_request["duration"] = 100
        response = client.post("/predict", json=valid_request)
        assert response.status_code == 422
    
    def test_predict_missing_field(self, valid_request):
        """Missing required field should return 422."""
        del valid_request["age"]
        response = client.post("/predict", json=valid_request)
        assert response.status_code == 422


class TestRootEndpoint:
    """Tests for root endpoint."""
    
    def test_root_returns_info(self):
        """Root should return API info."""
        response = client.get("/")
        assert response.status_code == 200
        
        data = response.json()
        assert "message" in data
        assert "version" in data
        assert "docs" in data


class TestDocsEndpoint:
    """Tests for documentation endpoints."""
    
    def test_docs_available(self):
        """Swagger docs should be available."""
        response = client.get("/docs")
        assert response.status_code == 200
    
    def test_redoc_available(self):
        """ReDoc should be available."""
        response = client.get("/redoc")
        assert response.status_code == 200
