
import os
os.environ["CIVITAS_MOCK"] = "1"

from fastapi.testclient import TestClient
from server.app import app

def test_health():
    with TestClient(app) as client:
        r = client.get("/api/health")
        assert r.status_code == 200
        assert r.json()["status"] == "ok"

def test_analyze_text_empty():
    with TestClient(app) as client:
        r = client.post("/api/analyze/text", json={"text": ""})
        assert r.status_code == 400

def test_analyze_text_ok():
    with TestClient(app) as client:
        r = client.post("/api/analyze/text", json={"text": "hello world"})
        assert r.status_code == 200
        data = r.json()
        assert "detection" in data
        assert "hate_prob" in data["detection"]
        assert "suggestions" in data
        assert "timing_ms" in data

def test_feedback():
    with TestClient(app) as client:
        r = client.post("/api/feedback", json={"analysis_id": "test-123", "note": "test"})
        assert r.status_code == 200

def test_history():
    with TestClient(app) as client:
        r = client.get("/api/history")
        assert r.status_code == 200
        assert isinstance(r.json(), list)
