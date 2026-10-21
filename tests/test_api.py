"""
End-to-end unit tests for FastAPI endpoints.
Tests analysis, rewrite, feedback, analytics, model-status, and health routes.
"""

import pytest
from fastapi.testclient import TestClient
from server.app import app


@pytest.fixture
def client():
    with TestClient(app) as c:
        yield c


def test_health_endpoint(client):
    r = client.get("/api/health")
    assert r.status_code == 200
    data = r.json()
    assert data["status"] == "ok"
    assert "models" in data
    assert "model_a" in data["models"]
    assert "model_b" in data["models"]


def test_model_status_endpoint(client):
    r = client.get("/api/model-status")
    assert r.status_code == 200
    data = r.json()
    assert "model_a" in data
    assert "model_b" in data
    assert "is_installed" in data["model_a"]
    assert "is_demo_mode" in data["model_a"]


def test_analyze_text_endpoint(client):
    r = client.post("/api/analyze/text", json={"text": "you people are a plague on this country", "lang_hint": "en"})
    assert r.status_code == 200
    data = r.json()
    assert "detection" in data
    assert data["detection"]["is_hate"] is True
    assert "suggestions" in data
    assert "analysis_id" in data
    assert len(data["suggestions"]["rewrite"]) > 0


def test_rewrite_endpoint(client):
    r = client.post("/api/rewrite", json={"text": "you people are a plague", "tone": "factual", "language": "en"})
    assert r.status_code == 200
    data = r.json()
    assert "rewrites" in data
    assert data["requested_tone"] == "factual"
    assert len(data["rewrites"]) > 0


def test_feedback_and_history_endpoints(client):
    # 1. Run an analysis
    r_an = client.post("/api/analyze/text", json={"text": "test phrase for feedback", "lang_hint": "en"})
    an_id = r_an.json()["analysis_id"]

    # 2. Submit feedback
    r_fb = client.post("/api/feedback", json={
        "analysis_id": an_id,
        "correct_label": True,
        "note": "Accurate detection",
        "rating": "helpful"
    })
    assert r_fb.status_code == 200
    assert "feedback_id" in r_fb.json()

    # 3. Check history
    r_hist = client.get("/api/history")
    assert r_hist.status_code == 200
    hist = r_hist.json()
    assert any(h["id"] == an_id for h in hist)

    # 4. Check analytics
    r_stat = client.get("/api/analytics")
    assert r_stat.status_code == 200
    stats = r_stat.json()
    assert stats["total_analyses"] >= 1
    assert stats["total_feedback_submissions"] >= 1
