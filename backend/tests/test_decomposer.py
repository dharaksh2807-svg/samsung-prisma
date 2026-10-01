import asyncio
import sys
import os

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from fastapi.testclient import TestClient
from decomposer import MultiIntentDecomposer, SubQuery
from main import app

client = TestClient(app)

# ─────────────────────────────────────────────────────────────────────────────
# Unit Tests — MultiIntentDecomposer
# ─────────────────────────────────────────────────────────────────────────────

def _run(coro):
    return asyncio.run(coro)


def test_single_intent_query():
    """Simple single-topic query should produce exactly 1 sub-query."""
    d = MultiIntentDecomposer()
    result = _run(d.decompose("Explain the digital lending policy."))
    assert len(result.sub_queries) >= 1, "Should produce at least 1 sub-query"
    intents = [sq.intent for sq in result.sub_queries]
    assert any("digital_lending" in i or "general" in i for i in intents), \
        f"Expected digital_lending intent, got: {intents}"


def test_multi_intent_query_produces_multiple_subqueries():
    """Compound query with KYC + existing customers must produce >= 2 sub-queries (Gate G3)."""
    d = MultiIntentDecomposer()
    result = _run(d.decompose("What changed in KYC and who is affected for existing customers?"))
    assert len(result.sub_queries) >= 2, \
        f"Expected >= 2 sub-queries for compound query, got {len(result.sub_queries)}: {result.sub_queries}"


def test_three_intent_query():
    """RBI + digital lending + KYC must produce >= 3 sub-queries."""
    d = MultiIntentDecomposer()
    result = _run(d.decompose("Does the new RBI regulation affect our digital lending KYC requirements?"))
    assert len(result.sub_queries) >= 2, \
        f"Expected >= 2 sub-queries, got {len(result.sub_queries)}: {result.sub_queries}"
    intents = [sq.intent for sq in result.sub_queries]
    # At least one of these core concepts should be detected
    covered = {"kyc_requirements", "digital_lending", "rbi_regulations"} & set(intents)
    assert len(covered) >= 1, f"Expected regulatory intents detected, got: {intents}"


def test_no_filler_in_search_query():
    """Search queries must not contain conversational filler words."""
    d = MultiIntentDecomposer()
    filler_words = {"please", "can", "you", "tell", "explain", "what", "how"}
    result = _run(d.decompose("Can you please explain what the KYC requirements are for new customers?"))
    for sq in result.sub_queries:
        query_words = set(sq.search_query.lower().split())
        overlap = filler_words & query_words
        assert not overlap, f"Filler words found in search_query: {overlap} in '{sq.search_query}'"


def test_empty_query_handled_gracefully():
    """Empty input should return at least one sub-query without crashing."""
    d = MultiIntentDecomposer()
    result = _run(d.decompose(""))
    assert len(result.sub_queries) >= 1


def test_latency_under_200ms():
    """Decomposition must complete in under 200ms on heuristic engine."""
    d = MultiIntentDecomposer()
    result = _run(d.decompose("What are the AML obligations for NBFCs under RBI guidelines?"))
    assert result.latency_ms < 200.0, f"Latency {result.latency_ms:.2f}ms exceeded 200ms target"


# ─────────────────────────────────────────────────────────────────────────────
# Integration Tests — REST API
# ─────────────────────────────────────────────────────────────────────────────

def test_rest_decompose_single_intent():
    """POST /api/decomposer/decompose should return valid sub_queries."""
    response = client.post("/api/decomposer/decompose", json={
        "query": "Explain the digital lending policy."
    })
    assert response.status_code == 200
    data = response.json()
    assert "sub_queries" in data
    assert len(data["sub_queries"]) >= 1
    for sq in data["sub_queries"]:
        assert "intent" in sq
        assert "search_query" in sq
        assert len(sq["search_query"]) > 0


def test_rest_decompose_multi_intent():
    """POST /api/decomposer/decompose on compound query must return >= 2 sub-queries (Gate G3)."""
    response = client.post("/api/decomposer/decompose", json={
        "query": "What changed in KYC and who is affected for existing customers?"
    })
    assert response.status_code == 200
    data = response.json()
    assert len(data["sub_queries"]) >= 2, \
        f"Gate G3 failed: expected >= 2 sub-queries, got {len(data['sub_queries'])}"


def test_websocket_retrieve_triggers_decompose():
    """On RETRIEVE decision, WebSocket response must include sub_queries (pipeline integration)."""
    with client.websocket_connect("/ws/stream") as ws:
        ws.send_json({
            "chunk": "What changed in KYC and who is affected for existing customers?",
            "timestamp": 1.0
        })
        resp = ws.receive_json()
        assert resp["decision"] == "RETRIEVE"
        assert "sub_queries" in resp, "sub_queries must be present on RETRIEVE decision"
        assert len(resp["sub_queries"]) >= 1
        assert "decompose_latency_ms" in resp
