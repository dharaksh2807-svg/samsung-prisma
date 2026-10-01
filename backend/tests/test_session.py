import pytest
import asyncio
import sys
import os

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from fastapi.testclient import TestClient
from session import (
    SessionManager,
    DeltaQueryGenerator,
    EvidenceItem,
    SessionState
)
from main import app

client = TestClient(app)

def _run(coro):
    return asyncio.run(coro)

# ─── Unit Tests: SessionManager & DeltaQueryGenerator ────────────────────────

def test_session_initialization_and_state():
    mgr = SessionManager()
    session = mgr.record_initial_turn(
        session_id="sess_123",
        query="What are the KYC reverification rules?",
        answer="All customers must re-verify every 2 years [REG_001].",
        evidence=[EvidenceItem(doc_id="REG_001", text="KYC reverification every 2 years.")]
    )

    assert session.session_id == "sess_123"
    assert session.answer_version == 1
    assert len(session.history) == 1
    assert len(session.current_evidence) == 1
    assert session.current_evidence[0].doc_id == "REG_001"

    retrieved = mgr.get_session("sess_123")
    assert retrieved is not None
    assert retrieved.session_id == "sess_123"

def test_delta_query_generation_refinement():
    gen = DeltaQueryGenerator()
    res = _run(gen.generate(
        previous_query="What are the digital lending KYC requirements?",
        previous_answer="Digital lending requires video KYC [REG_002].",
        new_constraint="...only for existing customers"
    ))

    assert res["is_refinement"] is True
    assert "existing customers" in res["search_query"].lower()
    # Topic from previous query should be preserved
    assert any(kw in res["search_query"].lower() for kw in ["kyc", "lending"])

def test_delta_query_generation_unrelated():
    gen = DeltaQueryGenerator()
    res = _run(gen.generate(
        previous_query="What are the KYC reverification rules?",
        previous_answer="KYC must be renewed [REG_001].",
        new_constraint="What is the statutory liquidity ratio requirement for commercial banks?"
    ))

    # Completely different question should not merge with KYC
    assert "statutory liquidity ratio" in res["search_query"].lower()

def test_evidence_accumulation_and_version_bump():
    mgr = SessionManager()
    mgr.record_initial_turn(
        session_id="sess_abc",
        query="Explain digital lending rules",
        answer="Lenders must follow fair practice code [REG_005].",
        evidence=[EvidenceItem(doc_id="REG_005", text="Fair practice code.")]
    )

    # Apply refinement with a delta evidence item
    updated = mgr.apply_refinement(
        session_id="sess_abc",
        new_constraint="for foreign fintechs",
        refined_answer="Foreign fintechs must also maintain local escrow [REG_006].",
        delta_evidence=[EvidenceItem(doc_id="REG_006", text="Local escrow mandatory for foreign fintechs.")]
    )

    assert updated.answer_version == 2
    assert len(updated.history) == 2
    assert len(updated.current_evidence) == 2
    doc_ids = [e.doc_id for e in updated.current_evidence]
    assert "REG_005" in doc_ids
    assert "REG_006" in doc_ids

# ─── Integration Tests: FastAPI Endpoints ────────────────────────────────────

def test_api_init_session():
    payload = {
        "session_id": "test_api_session",
        "query": "What are digital lending guidelines?",
        "answer": "Digital lending guidelines require board approved policy [REG_010].",
        "evidence": [{"doc_id": "REG_010", "text": "Board approved policy mandatory."}]
    }
    response = client.post("/api/session/init", json=payload)
    assert response.status_code == 200
    data = response.json()
    assert data["session_id"] == "test_api_session"
    assert data["answer_version"] == 1
    assert len(data["current_evidence"]) == 1

def test_api_delta_query():
    # Make sure session exists
    client.post("/api/session/init", json={
        "session_id": "sess_delta_test",
        "query": "What are the KYC reverification rules?",
        "answer": "Every 2 years [REG_001].",
        "evidence": [{"doc_id": "REG_001", "text": "Every 2 years."}]
    })

    payload = {
        "session_id": "sess_delta_test",
        "new_constraint": "...only for high net worth individuals"
    }
    response = client.post("/api/session/delta_query", json=payload)
    assert response.status_code == 200
    data = response.json()
    assert data["session_id"] == "sess_delta_test"
    assert data["is_refinement"] is True
    assert "high net worth" in data["search_query"].lower()
    assert "REG_001" in data["cached_evidence_ids"]

def test_api_refine_session():
    # Setup initial session
    client.post("/api/session/init", json={
        "session_id": "sess_refine_flow",
        "query": "What are the audit rules for fintechs?",
        "answer": "Annual audit required [REG_030].",
        "evidence": [{"doc_id": "REG_030", "text": "Fintechs must conduct annual audit."}]
    })

    # Refine with new constraint and new delta evidence
    refine_payload = {
        "session_id": "sess_refine_flow",
        "new_constraint": "specifically for payment gateways",
        "delta_evidence": [{"doc_id": "REG_031", "text": "Payment gateways require quarterly security audits."}]
    }
    response = client.post("/api/session/refine", json=refine_payload)
    assert response.status_code == 200
    data = response.json()
    assert data["session_id"] == "sess_refine_flow"
    assert data["answer_version"] == 2
    assert "REG_030" in data["combined_evidence_ids"]
    assert "REG_031" in data["combined_evidence_ids"]
    assert len(data["refined_answer"]) > 0

def test_api_get_session_state():
    # Query state of previously created session
    response = client.get("/api/session/sess_refine_flow")
    assert response.status_code == 200
    data = response.json()
    assert data["session_id"] == "sess_refine_flow"
    assert data["answer_version"] == 2
    assert len(data["history"]) == 2

def test_api_get_session_not_found():
    response = client.get("/api/session/nonexistent_session_9999")
    assert response.status_code == 404
