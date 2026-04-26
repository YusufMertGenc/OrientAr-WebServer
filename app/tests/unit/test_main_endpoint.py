from __future__ import annotations


def test_chatbot_query_predefined_response(client, monkeypatch):
    import app.main as main

    monkeypatch.setattr(
        main,
        "match_predefined_response",
        lambda question: {"message": "OrientAR helps students.", "confidence": 0.95},
    )

    response = client.post("/chatbot/query", json={"question": "What is OrientAR?"})
    assert response.status_code == 200

    data = response.json()
    assert data["answer"] == "OrientAR helps students."
    assert data["confidence"] == 0.95
    assert data["in_domain"] is True
    assert data["context_used"] == []


def test_chatbot_query_out_of_domain(client, monkeypatch):
    import app.main as main

    monkeypatch.setattr(main, "match_predefined_response", lambda question: None)

    async def fake_rag_query_async(question, top_k=4):
        return {"documents": [], "domain_score": 0.12, "in_domain": False}

    monkeypatch.setattr(main, "rag_query_async", fake_rag_query_async)

    response = client.post("/chatbot/query", json={"question": "Who won the NBA finals?"})
    assert response.status_code == 200

    data = response.json()
    assert data["answer"] == main.OUT_OF_DOMAIN_MESSAGE
    assert data["in_domain"] is False
    assert data["context_used"] == []


def test_chatbot_query_no_documents_returns_safe_fallback(client, monkeypatch):
    import app.main as main

    monkeypatch.setattr(main, "match_predefined_response", lambda question: None)

    async def fake_rag_query_async(question, top_k=4):
        return {"documents": [], "domain_score": 0.8, "in_domain": True}

    monkeypatch.setattr(main, "rag_query_async", fake_rag_query_async)

    response = client.post("/chatbot/query", json={"question": "Where is the library?"})
    assert response.status_code == 200

    data = response.json()
    assert "not sure" in data["answer"].lower()
    assert data["confidence"] == 0.25
    assert data["in_domain"] is True


def test_chatbot_query_valid_flow_returns_cleaned_response(client, monkeypatch):
    import app.main as main

    monkeypatch.setattr(main, "match_predefined_response", lambda question: None)

    async def fake_rag_query_async(question, top_k=4):
        return {
            "documents": ["The library is next to the cafeteria."],
            "domain_score": 0.91,
            "in_domain": True,
        }

    async def fake_generate_intent_response(question, documents):
        return {
            "message": '{"message":"The library is next to the cafeteria.","confidence":0.88}',
            "confidence": 0.88,
        }

    monkeypatch.setattr(main, "rag_query_async", fake_rag_query_async)
    monkeypatch.setattr(main, "generate_intent_response", fake_generate_intent_response)

    response = client.post("/chatbot/query", json={"question": "Where is the library?"})
    assert response.status_code == 200

    data = response.json()
    assert data["answer"] == "The library is next to the cafeteria."
    assert data["confidence"] == 0.88
    assert data["in_domain"] is True
    assert data["domain_score"] == 0.91
    assert data["context_used"] == ["The library is next to the cafeteria."]
    assert "latency_ms" in data


def test_chatbot_query_empty_llm_message_falls_back(client, monkeypatch):
    import app.main as main

    monkeypatch.setattr(main, "match_predefined_response", lambda question: None)

    async def fake_rag_query_async(question, top_k=4):
        return {
            "documents": ["Student Affairs is in the admin building."],
            "domain_score": 0.85,
            "in_domain": True,
        }

    async def fake_generate_intent_response(question, documents):
        return {"message": "", "confidence": 0.9}

    monkeypatch.setattr(main, "rag_query_async", fake_rag_query_async)
    monkeypatch.setattr(main, "generate_intent_response", fake_generate_intent_response)

    response = client.post("/chatbot/query", json={"question": "Where is Student Affairs?"})
    assert response.status_code == 200

    data = response.json()
    assert "not sure" in data["answer"].lower()
    assert data["confidence"] == 0.25

def test_chatbot_query_missing_question_field(client):
    response = client.post("/chatbot/query", json={})
    assert response.status_code == 422


def test_chatbot_query_empty_question(client):
    response = client.post("/chatbot/query", json={"question": ""})
    assert response.status_code == 422