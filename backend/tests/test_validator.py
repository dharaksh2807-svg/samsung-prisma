import pytest
import asyncio
import sys
import os

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from fastapi.testclient import TestClient
from validator import CitationValidator, ValidatorInput
from main import app

client = TestClient(app)

def _run(coro):
    return asyncio.run(coro)

def test_validator_all_valid_citations():
    """All citations exist in the valid corpus -> is_valid must be True."""
    val = CitationValidator()
    ans = "Digital lending guidelines require explicit consent [REG_014_SEC_4] and data localization [REG_015_SEC_2]."
    valid_ids = ["REG_014_SEC_4", "REG_015_SEC_2", "REG_099"]
    
    res = _run(val.validate(ValidatorInput(generated_answer=ans, valid_document_ids=valid_ids)))
    
    assert res.is_valid is True
    assert res.fabricated_ids_found == []
    assert res.clean_answer == ans
    assert res.latency_ms < 50.0  # Ultra-low latency

def test_validator_detects_fabricated_citations():
    """Any citation not in valid_document_ids must be flagged as fabricated and stripped from clean_answer."""
    val = CitationValidator()
    ans = "Existing customers must update KYC [REG_014_SEC_4] under penal provisions [FAKE_REG_999]."
    valid_ids = ["REG_014_SEC_4"]
    
    res = _run(val.validate(ValidatorInput(generated_answer=ans, valid_document_ids=valid_ids)))
    
    assert res.is_valid is False
    assert "FAKE_REG_999" in res.fabricated_ids_found
    assert "[FAKE_REG_999]" not in res.clean_answer
    assert "[REG_014_SEC_4]" in res.clean_answer

def test_validator_multiple_hallucinations_cleaned():
    """Multiple non-existent citations are extracted and cleanly removed."""
    val = CitationValidator()
    ans = "Clause A [HALLUCINATED_1] and Clause B [HALLUCINATED_2] apply."
    valid_ids = ["GENUINE_DOC_01"]
    
    res = _run(val.validate(ValidatorInput(generated_answer=ans, valid_document_ids=valid_ids)))
    
    assert res.is_valid is False
    assert set(res.fabricated_ids_found) == {"HALLUCINATED_1", "HALLUCINATED_2"}
    assert "[HALLUCINATED_1]" not in res.clean_answer
    assert "[HALLUCINATED_2]" not in res.clean_answer

def test_validator_no_citations_is_valid():
    """An answer without citations is valid (no fabricated IDs)."""
    val = CitationValidator()
    ans = "Insufficient evidence in the current regulatory corpus."
    valid_ids = ["REG_01"]
    
    res = _run(val.validate(ValidatorInput(generated_answer=ans, valid_document_ids=valid_ids)))
    
    assert res.is_valid is True
    assert res.fabricated_ids_found == []
    assert res.clean_answer == ans

def test_validator_latency_under_5ms():
    """Deterministic validation must execute in under 5ms."""
    val = CitationValidator()
    ans = "Requirement 1 [REG_1], Requirement 2 [REG_2], Requirement 3 [REG_3]."
    valid_ids = ["REG_1", "REG_2"]
    
    res = _run(val.validate(ValidatorInput(generated_answer=ans, valid_document_ids=valid_ids)))
    assert res.latency_ms < 5.0

def test_rest_api_validate_endpoint():
    """POST /api/validator/validate REST endpoint test."""
    response = client.post("/api/validator/validate", json={
        "generated_answer": "Banks must report quarterly [REG_VALID] and semi-annually [REG_FABRICATED].",
        "valid_document_ids": ["REG_VALID"]
    })
    assert response.status_code == 200
    data = response.json()
    assert data["is_valid"] is False
    assert "REG_FABRICATED" in data["fabricated_ids_found"]
    assert "[REG_FABRICATED]" not in data["clean_answer"]
    assert "[REG_VALID]" in data["clean_answer"]
