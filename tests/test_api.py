from fastapi.testclient import TestClient
import src.api as api

class FakeEngine:
    model = object()
    def search(self, q, mode="hybrid", top_k=5, semantic_weight=.7):
        return {"query":q,"mode":mode,"semantic_weight":semantic_weight,"results":[]}

def test_health_and_search(monkeypatch):
    api.engine=FakeEngine()
    c=TestClient(api.app)
    assert c.get("/health").status_code == 200
    assert c.get("/search",params={"q":"data pipelines"}).status_code == 200
