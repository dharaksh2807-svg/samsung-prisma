import pytest
import asyncio
import sys
import os

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from fastapi.testclient import TestClient
from monitor import RegulatoryStreamMonitor, StreamMonitorInput
from main import app

client = TestClient(app)

def _run(coro):
    return asyncio.run(coro)


# ─── Test Data ───────────────────────────────────────────────────────────────

PREV_QUERY = "What are the KYC re-verification requirements for existing customers?"
PREV_ANSWER = (
    "Existing customers must complete KYC reverification by March 2027 [REG_014_SEC_4]. "
    "Video KYC is an acceptable method for digital lending customers [REG_015_SEC_2]."
)


def test_update_required_when_amendment_contradicts_previous_answer():
    """
    If the new doc explicitly amends a topic covered in the previous answer,
    update_required must be True and affected_claims must be non-empty.
    """
    mon = RegulatoryStreamMonitor()
    inp = StreamMonitorInput(
        previous_query=PREV_QUERY,
        previous_answer=PREV_ANSWER,
        new_doc_id="REG_AMEND_2026_01",
        new_document_text=(
            "Effective immediately, the KYC reverification deadline has been extended to December 2028. "
            "This supersedes all prior deadlines. Existing digital lending customers must now complete "
            "in-person reverification; video KYC is no longer acceptable."
        )
    )
    res = _run(mon.evaluate(inp))

    assert res.update_required is True
    assert len(res.affected_claims) > 0
    assert isinstance(res.reasoning, str) and len(res.reasoning) > 0


def test_no_update_when_amendment_is_unrelated():
    """
    If the new document is about an entirely different regulatory domain,
    update_required must be False.
    """
    mon = RegulatoryStreamMonitor()
    inp = StreamMonitorInput(
        previous_query=PREV_QUERY,
        previous_answer=PREV_ANSWER,
        new_doc_id="REG_EXPORT_2026_07",
        new_document_text=(
            "New guidelines for export credit agencies now mandate quarterly reporting "
            "to the Trade Finance Board. These regulations apply exclusively to cross-border transactions."
        )
    )
    res = _run(mon.evaluate(inp))

    assert res.update_required is False
    assert res.affected_claims == []
    assert isinstance(res.reasoning, str) and len(res.reasoning) > 0


def test_no_update_when_doc_mentions_topic_but_no_conflict_signal():
    """
    New doc may reference the same topic (KYC) but if it contains no
    conflict/change signal, the previous answer is still valid.
    """
    mon = RegulatoryStreamMonitor()
    inp = StreamMonitorInput(
        previous_query=PREV_QUERY,
        previous_answer=PREV_ANSWER,
        new_doc_id="REG_KYC_FAQ_2026",
        new_document_text=(
            "Frequently asked questions about the existing KYC framework for digital lending. "
            "Customers may refer to the official FAQ portal for clarifications."
        )
    )
    res = _run(mon.evaluate(inp))

    assert res.update_required is False


def test_output_model_structure():
    """StreamMonitorOutput must always contain all required fields."""
    mon = RegulatoryStreamMonitor()
    inp = StreamMonitorInput(
        previous_query="What is the capital reserve requirement?",
        previous_answer="Banks must maintain a 10% capital reserve [REG_CAP_01].",
        new_doc_id="REG_CAP_AMEND_2026",
        new_document_text=(
            "Capital reserve requirements have been revised. Banks must now maintain "
            "a 12% capital reserve, replacing prior requirements."
        )
    )
    res = _run(mon.evaluate(inp))

    assert hasattr(res, "update_required")
    assert hasattr(res, "affected_claims")
    assert hasattr(res, "reasoning")
    assert hasattr(res, "latency_ms")
    assert res.latency_ms >= 0.0
    assert res.update_required is True


def test_latency_under_50ms():
    """Deterministic heuristic fallback must resolve in under 50ms."""
    mon = RegulatoryStreamMonitor()
    inp = StreamMonitorInput(
        previous_query=PREV_QUERY,
        previous_answer=PREV_ANSWER,
        new_doc_id="REG_SPEED_TEST",
        new_document_text="KYC requirements amended to remove video KYC. This replaces all prior guidance."
    )
    res = _run(mon.evaluate(inp))
    assert res.latency_ms < 50.0


def test_rest_api_monitor_evaluate_endpoint_update_required():
    """POST /api/monitor/evaluate should detect an amendment conflict."""
    response = client.post("/api/monitor/evaluate", json={
        "previous_query": "What are the penalty provisions for non-compliance?",
        "previous_answer": "Non-compliant lending institutions face a fine of INR 10 lakh [REG_PEN_01].",
        "new_doc_id": "REG_PEN_AMEND_2026",
        "new_document_text": (
            "Penalty provisions have been revised. The fine for non-compliant lending is now "
            "amended to INR 50 lakh, effective immediately. This supersedes prior penalty provisions."
        )
    })
    assert response.status_code == 200
    data = response.json()
    assert "update_required" in data
    assert "affected_claims" in data
    assert "reasoning" in data
    assert data["update_required"] is True


def test_rest_api_monitor_evaluate_endpoint_no_update():
    """POST /api/monitor/evaluate should confirm no update for unrelated document."""
    response = client.post("/api/monitor/evaluate", json={
        "previous_query": "What are the KYC norms?",
        "previous_answer": "KYC requires Aadhaar verification [REG_KYC_01].",
        "new_doc_id": "REG_INFRA_2026",
        "new_document_text": "New infrastructure guidelines for government bond settlements issued today."
    })
    assert response.status_code == 200
    data = response.json()
    assert data["update_required"] is False
