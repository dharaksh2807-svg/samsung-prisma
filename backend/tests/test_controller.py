import pytest
import asyncio
from fastapi.testclient import TestClient
import sys
import os

# Add backend directory to sys.path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from controller import RetrievalController, ControllerDecision
from main import app

client = TestClient(app)

@pytest.fixture
def controller():
    return RetrievalController()

def test_wait_decision_on_incomplete_transcript(controller):
    """Rule 1: Return WAIT if the query is ambiguous, missing a core subject, or just starting."""
    incomplete_inputs = [
        "Does the new...",
        "What about the...",
        "Can you check if...",
        "Under the recent...",
        ""
    ]
    for inp in incomplete_inputs:
        result = asyncio.run(controller.evaluate(inp))
        assert result.decision == ControllerDecision.WAIT, f"Expected WAIT for '{inp}', got {result.decision}"
        assert result.confidence > 0.0
        assert len(result.reason) > 0
        assert result.latency_ms >= 0.0

def test_retrieve_decision_on_regulatory_subject(controller):
    """Rule 2: Return RETRIEVE the exact moment a clear regulatory topic, entity, or action appears."""
    retrieval_inputs = [
        "Does the new regulation affect digital lending",
        "What are the KYC norms for NBFCs?",
        "Are there any changes to FLDG arrangements with fintechs?",
        "RBI circular on capital adequacy ratio requirements"
    ]
    for inp in retrieval_inputs:
        result = asyncio.run(controller.evaluate(inp))
        assert result.decision == ControllerDecision.RETRIEVE, f"Expected RETRIEVE for '{inp}', got {result.decision}"
        assert result.confidence >= 0.8
        assert "regulatory" in result.reason.lower() or "topic" in result.reason.lower() or "llm" in result.reason.lower()

def test_no_retrieval_on_reformatting_request(controller):
    """Rule 3: Return NO-RETRIEVAL if the user is asking to reformat/summarize previous answer without new concepts."""
    reformat_inputs = [
        "Summarize that in three bullets.",
        "Make it shorter please.",
        "Can you rephrase the previous response in bullet points?",
        "Explain like I'm 5 without jargon."
    ]
    for inp in reformat_inputs:
        result = asyncio.run(controller.evaluate(inp, previous_context="The digital lending guidelines require escrow accounts..."))
        assert result.decision == ControllerDecision.NO_RETRIEVAL, f"Expected NO-RETRIEVAL for '{inp}', got {result.decision}"
        assert result.confidence >= 0.8

def test_latency_target(controller):
    """Verify latency is benchmarked and under 200ms target."""
    query = "Does the new regulation affect digital lending"
    result = asyncio.run(controller.evaluate(query))
    assert result.latency_ms < 200.0, f"Latency {result.latency_ms}ms exceeded 200ms target"

def test_rest_api_evaluate_endpoint():
    """Test POST /api/controller/evaluate REST endpoint."""
    response = client.post("/api/controller/evaluate", json={
        "transcript": "Does the new regulation affect digital lending"
    })
    assert response.status_code == 200
    data = response.json()
    assert data["decision"] == "RETRIEVE"
    assert "confidence" in data
    assert "reason" in data
    assert "latency_ms" in data

def test_websocket_streaming_lifecycle():
    """Test full streaming interaction over /ws/stream with progressive chunk evaluation."""
    with client.websocket_connect("/ws/stream") as websocket:
        # Step 1: User starts speaking
        websocket.send_json({"chunk": "Does the new...", "timestamp": 1.0})
        res1 = websocket.receive_json()
        assert res1["event"] == "CONTROLLER_DECISION"
        assert res1["decision"] == "WAIT"

        # Step 2: User completes subject
        websocket.send_json({"chunk": "Does the new regulation affect digital lending", "timestamp": 1.5})
        res2 = websocket.receive_json()
        assert res2["event"] == "CONTROLLER_DECISION"
        assert res2["decision"] == "RETRIEVE"

        # Step 3: User follows up with formatting request
        websocket.send_json({
            "chunk": "Summarize that in three bullets.", 
            "timestamp": 2.0,
            "previous_context": "Previous compliance answer text."
        })
        res3 = websocket.receive_json()
        assert res3["event"] == "CONTROLLER_DECISION"
        assert res3["decision"] == "NO-RETRIEVAL"
