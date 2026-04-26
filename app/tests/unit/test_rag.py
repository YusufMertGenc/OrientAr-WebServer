from __future__ import annotations

import math


def test_norm_ws_collapses_whitespace():
    from app.rag import _norm_ws

    assert _norm_ws("   hello   world \n  test ") == "hello world test"


def test_clamp_for_embed_truncates_long_text():
    from app.rag import _clamp_for_embed

    text = "a" * 50
    result = _clamp_for_embed(text, max_chars=10)

    assert len(result) == 10


def test_clamp_for_query_truncates_long_text():
    from app.rag import _clamp_for_query

    text = "b" * 60
    result = _clamp_for_query(text, max_chars=12)

    assert len(result) == 12


def test_lexical_overlap():
    from app.rag import _lexical_overlap

    score = _lexical_overlap("library opening hours", "The library hours are 8 to 5")
    assert score > 0


def test_rerank_documents_prefers_better_combined_score(monkeypatch):
    import app.rag as rag

    monkeypatch.setattr(rag, "FINAL_DOCS", 1)

    query = "library hours"
    candidates = [
        ("random cafeteria information", 0.95),
        ("library hours are 8 to 5", 0.75),
    ]

    ranked = rag.rerank_documents(query, candidates)

    assert len(ranked) == 1
    assert ranked[0] == "library hours are 8 to 5"


def test_top_topics_returns_sorted_topics(monkeypatch):
    import app.rag as rag

    monkeypatch.setattr(rag, "_TOPIC_IDS", ["a", "b", "c"])
    monkeypatch.setattr(
        rag,
        "_TOPIC_EMBS",
        [
            [1.0, 0.0],
            [0.0, 1.0],
            [0.7, 0.7],
        ],
    )

    result = rag.top_topics([1.0, 0.0], top_n=2)

    assert len(result) == 2
    assert result[0][0] == "a"
    assert result[0][1] >= result[1][1]


def test_is_in_domain_true(monkeypatch):
    import app.rag as rag

    monkeypatch.setattr(rag, "_REBUILDING", False)
    monkeypatch.setattr(rag, "TOPIC_DOMAIN_GUARD", 0.38)
    monkeypatch.setattr(rag, "_TOPIC_IDS", ["topic1"])
    monkeypatch.setattr(rag, "_TOPIC_EMBS", [[1.0, 0.0]])
    monkeypatch.setattr(rag, "ollama_embed", lambda texts: [[1.0, 0.0]])

    in_domain, score = rag.is_in_domain("Where is the library?")

    assert in_domain is True
    assert score >= 0.38


def test_is_in_domain_false(monkeypatch):
    import app.rag as rag

    monkeypatch.setattr(rag, "_REBUILDING", False)
    monkeypatch.setattr(rag, "TOPIC_DOMAIN_GUARD", 0.90)
    monkeypatch.setattr(rag, "_TOPIC_IDS", ["topic1"])
    monkeypatch.setattr(rag, "_TOPIC_EMBS", [[1.0, 0.0]])
    monkeypatch.setattr(rag, "ollama_embed", lambda texts: [[0.0, 1.0]])

    in_domain, score = rag.is_in_domain("Tell me about Bitcoin price")

    assert in_domain is False
    assert score < 0.90

def test_is_in_domain_at_threshold(monkeypatch):
    import app.rag as rag

    monkeypatch.setattr(rag, "_REBUILDING", False)
    monkeypatch.setattr(rag, "TOPIC_DOMAIN_GUARD", 0.50)
    monkeypatch.setattr(rag, "_TOPIC_IDS", ["topic1"])
    monkeypatch.setattr(rag, "_TOPIC_EMBS", [[1.0, 0.0]])
    monkeypatch.setattr(rag, "ollama_embed", lambda texts: [[1.0, 1.0]])

    # cosine([1,1], [1,0]) = 1/sqrt(2) ≈ 0.707, not exact threshold
    # so patch top_topics directly for exact threshold behavior
    monkeypatch.setattr(rag, "top_topics", lambda q_emb, top_n=rag.TOPIC_TOP_N: [("topic1", 0.50)])

    in_domain, score = rag.is_in_domain("threshold case")
    assert in_domain is True
    assert score == 0.50


def test_rerank_documents_empty_candidates():
    from app.rag import rerank_documents

    ranked = rerank_documents("library hours", [])
    assert ranked == []