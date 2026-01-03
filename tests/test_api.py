# tests/test_api.py
import pytest
from fastapi.testclient import TestClient
import sys
import os

# Add parent directory to path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

# Mock the model loading for testing
import pickle
from unittest.mock import Mock, patch
import numpy as np

@pytest.fixture
def mock_model():
    """Create a mock model for testing"""
    model = Mock()
    model.predict.return_value = np.array([1])
    model.predict_proba.return_value = np.array([[0.3, 0.7]])
    return model

@pytest.fixture
def mock_preprocessor():
    """Create a mock preprocessor for testing"""
    preprocessor = Mock()
    preprocessor.transform.return_value = np.random.randn(1, 13)
    return preprocessor

@pytest.fixture
def client(mock_model, mock_preprocessor):
    """Create test client with mocked model"""
    with patch('pickle.load') as mock_load:
        mock_load.side_effect = [mock_model, mock_preprocessor]
        from api.app import app
        return TestClient(app)

class TestAPI:
    """Test API endpoints"""
    
    def test_root_endpoint(self, client):
        """Test root endpoint returns correct response"""
        response = client.get("/")
        assert response.status_code == 200
        assert "message" in response.json()
    
    def test_health_endpoint(self, client):
        """Test health check endpoint"""
        response = client.get("/health")
        assert response.status_code == 200
        assert response.json()["status"] == "healthy"
    
    def test_predict_endpoint_valid_input(self, client):
        """Test prediction endpoint with valid input"""
        test_data = {
            "age": 63.0,
            "sex": 1,
            "cp": 1,
            "trestbps": 145.0,
            "chol": 233.0,
            "fbs": 1,
            "restecg": 2,
            "thalach": 150.0,
            "exang": 0,
            "oldpeak": 2.3,
            "slope": 3,
            "ca": 0,
            "thal": 6
        }
        
        response = client.post("/predict", json=test_data)
        assert response.status_code == 200
        
        result = response.json()
        assert "prediction" in result
        assert "confidence" in result
        assert "risk_level" in result
        assert result["prediction"] in [0, 1]
    
    def test_predict_endpoint_missing_field(self, client):
        """Test prediction endpoint with missing field"""
        incomplete_data = {
            "age": 63.0,
            "sex": 1
            # Missing other required fields
        }
        
        response = client.post("/predict", json=incomplete_data)
        assert response.status_code == 422  # Validation error
    
    def test_predict_endpoint_invalid_types(self, client):
        """Test prediction endpoint with invalid data types"""
        invalid_data = {
            "age": "not_a_number",  # Should be float
            "sex": 1,
            "cp": 1,
            "trestbps": 145.0,
            "chol": 233.0,
            "fbs": 1,
            "restecg": 2,
            "thalach": 150.0,
            "exang": 0,
            "oldpeak": 2.3,
            "slope": 3,
            "ca": 0,
            "thal": 6
        }
        
        response = client.post("/predict", json=invalid_data)
        assert response.status_code == 422  # Validation error


if __name__ == "__main__":
    pytest.main([__file__, "-v"])