import os
import time
import asyncio
from typing import List, Dict, Optional
from pydantic import BaseModel, Field
from dotenv import load_dotenv

load_dotenv()

# ─────────────────────────────────────────────────────────────────────────────
# Output Models
# ─────────────────────────────────────────────────────────────────────────────

class SynthesizerInput(BaseModel):
    current_query: str
    session_history: List[str] = Field(default_factory=list)
    retrieved_chunks: List[Dict[str, str]] = Field(
        description="List of dicts with 'doc_id' and 'text' keys"
    )

class SynthesizerOutput(BaseModel):
    answer_markdown: str
    latency_ms: float = 0.0

# ─────────────────────────────────────────────────────────────────────────────
# Prompt 3 — Stateful RAG Synthesizer
# ─────────────────────────────────────────────────────────────────────────────

STATEFUL_SYNTHESIZER_PROMPT = """<SYSTEM_PERSONA>
You are RegulaStream, a highly accurate regulatory intelligence assistant. You must answer the user's query based ONLY on the provided retrieved regulatory chunks.
</SYSTEM_PERSONA>

<SESSION_CONTEXT>
Previous Queries: {session_history}
</SESSION_CONTEXT>

<RETRIEVED_EVIDENCE>
{retrieved_chunks}
</RETRIEVED_EVIDENCE>

<RULES>
1. EVERY factual claim you make MUST be followed by the exact Document ID in brackets, e.g., "Existing customers need reverification [REG_014_SEC_4]."
2. If the evidence contradicts itself, state the conflict and cite both sources.
3. If the evidence does not contain the answer, reply ONLY with: "Insufficient evidence in the current regulatory corpus."
4. Incorporate late constraints (from Session Context) to refine your answer.
</RULES>

<USER_QUERY>
{current_query}
</USER_QUERY>

Generate the final response in Markdown format. Ensure strict citation discipline. Do not output JSON."""


class StatefulSynthesizer:
    """
    Generates the final response grounded strictly in retrieved evidence.
    Enforces citation discipline (Gate G4 preparation).
    """

    def __init__(self, api_key: Optional[str] = None, model_name: Optional[str] = None):
        self.api_key = api_key or os.getenv("GEMINI_API_KEY") or os.getenv("GOOGLE_API_KEY")
        # For synthesis, a more capable model like gemini-1.5-pro or Sonnet 3.5 is ideal,
        # but we use gemini-3.8-flash as default for speed unless overridden.
        self.model_name = model_name or os.getenv("SYNTHESIZER_MODEL", "gemini-3.8-flash")
        
        from llm_client import get_llm_client
        self._llm = get_llm_client()

    # _init_client removed

    async def synthesize(self, input_data: SynthesizerInput) -> SynthesizerOutput:
        start = time.perf_counter()

        # Format retrieved evidence
        formatted_evidence = ""
        for chunk in input_data.retrieved_chunks:
            doc_id = chunk.get("doc_id", "UNKNOWN")
            text = chunk.get("text", "")
            formatted_evidence += f"[{doc_id}] {text}\n\n"

        if not formatted_evidence.strip():
            formatted_evidence = "No evidence retrieved."

        # Format session history
        formatted_history = "\n".join(input_data.session_history) if input_data.session_history else "None"

        # Build prompt
        prompt = STATEFUL_SYNTHESIZER_PROMPT.format(
            session_history=formatted_history,
            retrieved_chunks=formatted_evidence.strip(),
            current_query=input_data.current_query.strip()
        )

        # 1 — Try Gemini LLM
        if self._llm:
            try:
                response = await self._llm.generate_content(
                    model=self.model_name,
                    contents=prompt,
                )
                answer = response.text.strip()
                return SynthesizerOutput(
                    answer_markdown=answer,
                    latency_ms=(time.perf_counter() - start) * 1000
                )
            except Exception as e:
                print(f"[StatefulSynthesizer] LLM failed ({e}), falling back to mock.")

        # 2 — Mock fallback if no API key or network error
        answer = self._mock_synthesize(input_data)
        return SynthesizerOutput(
            answer_markdown=answer,
            latency_ms=(time.perf_counter() - start) * 1000
        )

    def _mock_synthesize(self, input_data: SynthesizerInput) -> str:
        """Deterministic fallback when API keys are exhausted."""
        if not input_data.retrieved_chunks:
            return "Insufficient evidence in the current regulatory corpus."
            
        answer = "**[SYSTEM FALLBACK: All LLM API keys exhausted (Rate limit exceeded)]**\n\nSince the LLM cannot be reached, here are the exact excerpts from the retrieved regulations:\n\n"
        for chunk in input_data.retrieved_chunks:
            doc_id = chunk.get("doc_id", "UNKNOWN")
            text = chunk.get("text", "")
            answer += f"**[{doc_id}]**\n> {text}\n\n"
            
        return answer.strip()
