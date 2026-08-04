"""
Test cho các endpoint FastAPI trong app.py.

RecommenderService bị "giả lập" (mock) hoàn toàn nên test không cần
artifacts/ thật hay kết nối PostgreSQL thật -> chạy được trên CI.
"""
import sys
from unittest.mock import MagicMock, patch

import pytest
from fastapi.testclient import TestClient


@pytest.fixture
def client():
    fake_service = MagicMock()
    fake_service.recommend.return_value = {
        "user_id": 1,
        "source": "trending",
        "recommendations": [{"product_id": 111, "score": None}],
    }

    with patch("model_utils.RecommenderService", return_value=fake_service):
        # Xoá cache import cũ (nếu có) để app.py chạy lại __init__
        # RecommenderService() bên trong context đã bị patch ở trên.
        sys.modules.pop("app", None)
        import app as app_module

        yield TestClient(app_module.app)

    sys.modules.pop("app", None)


def test_health_endpoint_returns_ok(client):
    response = client.get("/health")
    assert response.status_code == 200
    assert response.json() == {"status": "ok"}


def test_recommend_endpoint_returns_expected_shape(client):
    response = client.get("/recommend/1")
    assert response.status_code == 200
    data = response.json()
    assert data["user_id"] == 1
    assert data["source"] == "trending"
    assert data["recommendations"][0]["product_id"] == 111


def test_recommend_endpoint_accepts_top_n_param(client):
    response = client.get("/recommend/1?n=3")
    assert response.status_code == 200
