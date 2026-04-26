from fastapi.testclient import TestClient


def test_valid_campus_query_full_flow_with_controlled_services(monkeypatch):
    import app.main as main

    client = TestClient(main.app)

    monkeypatch.setattr(main, "match_predefined_response", lambda question: None)

    async def fake_rag_query_async(question, top_k=4):
        assert question == "Where is the library?"
        return {
            "documents": ["The library is located near the cafeteria."],
            "domain_score": 0.88,
            "in_domain": True,
        }

    async def fake_generate_intent_response(question, documents):
        assert documents == ["The library is located near the cafeteria."]
        return {
            "message": '{"message":"The library is located near the cafeteria.","confidence":0.9}',
            "confidence": 0.9,
        }

    monkeypatch.setattr(main, "rag_query_async", fake_rag_query_async)
    monkeypatch.setattr(main, "generate_intent_response", fake_generate_intent_response)

    response = client.post("/chatbot/query", json={"question": "Where is the library?"})

    assert response.status_code == 200
    data = response.json()

    assert data["answer"] == "The library is located near the cafeteria."
    assert data["confidence"] == 0.9
    assert data["context_used"] == ["The library is located near the cafeteria."]
    assert data["domain_score"] == 0.88
    assert data["in_domain"] is True
    assert "latency_ms" in data


#out of domain integration test

def test_out_of_domain_query_triggers_fallback(monkeypatch):
    import app.main as main

    client = TestClient(main.app)

    monkeypatch.setattr(main, "match_predefined_response", lambda question: None)

    async def fake_rag_query_async(question, top_k=4):
        return {
            "documents": [],
            "domain_score": 0.12,
            "in_domain": False,
        }

    monkeypatch.setattr(main, "rag_query_async", fake_rag_query_async)

    response = client.post("/chatbot/query", json={"question": "Who won the NBA finals?"})

    assert response.status_code == 200
    data = response.json()

    assert data["answer"] == main.OUT_OF_DOMAIN_MESSAGE
    assert data["context_used"] == []
    assert data["in_domain"] is False

#empty retrieval test

def test_empty_retrieval_returns_safe_response(monkeypatch):
    import app.main as main

    client = TestClient(main.app)

    monkeypatch.setattr(main, "match_predefined_response", lambda question: None)

    async def fake_rag_query_async(question, top_k=4):
        return {
            "documents": [],
            "domain_score": 0.75,
            "in_domain": True,
        }

    monkeypatch.setattr(main, "rag_query_async", fake_rag_query_async)

    response = client.post("/chatbot/query", json={"question": "Where is the student affairs office?"})

    assert response.status_code == 200
    data = response.json()

    assert "not sure" in data["answer"].lower()
    assert data["confidence"] == 0.25
    assert data["context_used"] == []
    assert data["in_domain"] is True

#Malformed LLM output test
def test_malformed_llm_output_is_cleaned(monkeypatch):
    import app.main as main

    client = TestClient(main.app)

    monkeypatch.setattr(main, "match_predefined_response", lambda question: None)

    async def fake_rag_query_async(question, top_k=4):
        return {
            "documents": ["The sports center includes gym and court facilities."],
            "domain_score": 0.82,
            "in_domain": True,
        }

    async def fake_generate_intent_response(question, documents):
        return {
            "message": "{\\\"message\\\": \\\"The sports center includes gym and court facilities.\\\"}",
            "confidence": 0.7,
        }

    monkeypatch.setattr(main, "rag_query_async", fake_rag_query_async)
    monkeypatch.setattr(main, "generate_intent_response", fake_generate_intent_response)

    response = client.post("/chatbot/query", json={"question": "What facilities are in the sports center?"})

    assert response.status_code == 200
    data = response.json()

    assert data["answer"] == "The sports center includes gym and court facilities."
    assert data["confidence"] == 0.7
    assert data["in_domain"] is True


