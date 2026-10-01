import pytest
import asyncio
import sys
import os

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from fastapi.testclient import TestClient
from synthesizer import StatefulSynthesizer, SynthesizerInput
from main import app

client = TestClient(app)

def _run(coro):
    return asyncio.run(coro)

def test_synthesizer_mock_cites_evidence():
    """Gate G4 Prep: Synthesizer must cite the provided document IDs."""
    synth = StatefulSynthesizer()
    
    input_data = SynthesizerInput(
        current_query="What are the KYC rules for existing customers?",
        session_history=[],
        retrieved_chunks=[
            {"doc_id": "REG_014_SEC_4", "text": "Existing customers need reverification by March 2027."},
            {"doc_id": "REG_015_SEC_2", "text": "KYC can be done via video."}
        ]
    )
    
    result = _run(synth.synthesize(input_data))
    
    assert "REG_014_SEC_4" in result.answer_markdown
    assert "REG_015_SEC_2" in result.answer_markdown
    assert result.latency_ms >= 0.0

def test_synthesizer_mock_contradiction_handling():
    """Rule 2: Synthesizer must state contradiction if evidence conflicts."""
    synth = StatefulSynthesizer()
    
    input_data = SynthesizerInput(
        current_query="Is video KYC allowed?",
        session_history=[],
        retrieved_chunks=[
            {"doc_id": "REG_OLD", "text": "Video KYC is not permitted. This creates a conflict."},
            {"doc_id": "REG_NEW", "text": "Video KYC is now allowed."}
        ]
    )
    
    result = _run(synth.synthesize(input_data))
    
    assert "contradicts" in result.answer_markdown.lower() or "conflict" in result.answer_markdown.lower()
    assert "REG_OLD" in result.answer_markdown
    assert "REG_NEW" in result.answer_markdown

def test_synthesizer_mock_insufficient_evidence():
    """Rule 3: Synthesizer must state insufficient evidence if no chunks are provided."""
    synth = StatefulSynthesizer()
    
    input_data = SynthesizerInput(
        current_query="What is the capital city of France?",
        session_history=[],
        retrieved_chunks=[]
    )
    
    result = _run(synth.synthesize(input_data))
    
    assert result.answer_markdown.strip() == "Insufficient evidence in the current regulatory corpus."

def test_rest_api_synthesize():
    """Test POST /api/synthesizer/synthesize REST endpoint."""
    response = client.post("/api/synthesizer/synthesize", json={
        "current_query": "What are the KYC rules?",
        "session_history": ["User asked about KYC"],
        "retrieved_chunks": [
            {"doc_id": "DOC_1", "text": "KYC is mandatory."}
        ]
    })
    assert response.status_code == 200
    data = response.json()
    assert "answer_markdown" in data
    assert "DOC_1" in data["answer_markdown"]
    assert "latency_ms" in data
