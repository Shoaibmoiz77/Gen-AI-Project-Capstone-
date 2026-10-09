from fastapi.testclient import TestClient

from groundwork.api import create_app


def _client(pipeline):
    return TestClient(create_app(pipeline))


def test_health(pipeline):
    with _client(pipeline) as c:
        body = c.get("/health").json()
    assert body["status"] == "ok" and body["chunks"] > 0


def test_ask_returns_cited_answer(pipeline):
    with _client(pipeline) as c:
        r = c.post("/ask", json={"question": "What is the on-call stipend?"})
    assert r.status_code == 200
    body = r.json()
    assert "$400" in body["answer"]
    assert body["sources"][0]["doc_id"] == "incident-response"
    assert body["sentences"][0]["chunk_ids"] == [body["sources"][0]["chunk_id"]]


def test_search_and_validation(pipeline):
    with _client(pipeline) as c:
        r = c.post("/search", json={"query": "parental leave", "k": 2, "mode": "bm25"})
        assert r.status_code == 200 and len(r.json()["results"]) == 2
        assert c.post("/search", json={"query": "x", "mode": "nope"}).status_code == 422
        assert c.post("/ask", json={"question": ""}).status_code == 422


def test_ui_is_served(pipeline):
    with _client(pipeline) as c:
        r = c.get("/")
    assert r.status_code == 200 and "Groundwork" in r.text


def test_documents_and_mode(pipeline):
    with _client(pipeline) as c:
        docs = c.get("/documents").json()["documents"]
        assert {d["doc_id"] for d in docs} >= {"expense-policy", "kestrel-x2-faq"}
        r = c.post("/ask", json={"question": "What is the on-call stipend?", "mode": "bm25"})
        assert r.status_code == 200
        assert any(x["cited"] for x in r.json()["retrieved"])
        assert c.post("/ask", json={"question": "x", "mode": "nope"}).status_code == 422
