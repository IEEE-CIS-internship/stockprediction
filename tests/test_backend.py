import os
import sys
import pytest
from fastapi.testclient import TestClient

# Add workspace root to path to import backend
sys.path.append(os.path.dirname(os.path.dirname(__file__)))
from backend.app import app

client = TestClient(app)

def test_root():
    response = client.get("/")
    assert response.status_code == 200
    assert response.json()["status"] == "running"

def test_tickers():
    response = client.get("/api/tickers")
    assert response.status_code == 200
    assert "tickers" in response.json()
    tickers = response.json()["tickers"]
    assert len(tickers) > 0
    assert "ICICIBANK.NS" in tickers

def test_market_data():
    response = client.get("/api/data/ICICIBANK.NS?limit=10")
    assert response.status_code == 200
    assert "data" in response.json()
    assert len(response.json()["data"]) <= 10

def test_regime():
    response = client.get("/api/regime/ICICIBANK.NS")
    assert response.status_code == 200
    res_data = response.json()
    assert "active_regime" in res_data
    assert "probabilities" in res_data
    assert "Bullish" in res_data["probabilities"]

def test_forecast():
    response = client.get("/api/forecast/ICICIBANK.NS?horizon=3")
    assert response.status_code == 200
    res_data = response.json()
    assert "predicted_residual_return" in res_data
    assert "trading_signal" in res_data
    assert res_data["horizon"] == "3d"

def test_explain():
    response = client.get("/api/explain/ICICIBANK.NS?horizon=3")
    assert response.status_code == 200
    res_data = response.json()
    assert "feature_attributions" in res_data
    assert "recommendation_reasoning" in res_data
    assert "top_drivers" in res_data

def test_refresh():
    # Only test schema/response structure
    response = client.post("/api/refresh/ICICIBANK.NS")
    assert response.status_code == 200
    res_data = response.json()
    assert res_data["status"] == "success"
    assert "updated_records_count" in res_data
    assert "current_regime" in res_data
