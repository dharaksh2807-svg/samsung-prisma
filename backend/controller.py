import os
import time
import json
import re
from enum import Enum
from typing import Optional, Dict, Any
from pydantic import BaseModel, Field
from dotenv import load_dotenv

load_dotenv()

class ControllerDecision(str, Enum):
    WAIT = "WAIT"
    RETRIEVE = "RETRIEVE"
    NO_RETRIEVAL = "NO-RETRIEVAL"

class ControllerOutput(BaseModel):
    decision: ControllerDecision
    confidence: float = Field(ge=0.0, le=1.0)
    reason: str
    latency_ms: float = 0.0

RETRIEVAL_CONTROLLER_SYSTEM_PROMPT = """<SYSTEM_PERSONA>
You are the Retrieval Controller for RegulaStream, a real-time compliance system. 
You analyze an incomplete, live user transcript to predict if the user has provided enough intent to trigger a vector database search.
</SYSTEM_PERSONA>

<RULES>
1. Return "WAIT" if the query is ambiguous, missing a core subject, or just starting.
2. Return "RETRIEVE" the exact moment a clear regulatory topic, entity, or action appears (even if grammar is broken).
3. Return "NO-RETRIEVAL" if the user is asking to reformat the previous answer (e.g., "make it shorter") without new concepts.
</RULES>

<EXAMPLES>
Input: "Does the new..."
Output: {"decision": "WAIT", "confidence": 0.9, "reason": "Missing regulatory subject."}

Input: "Does the new regulation affect digital lending"
Output: {"decision": "RETRIEVE", "confidence": 0.95, "reason": "Core subject 'digital lending' identified."}

Input: "Summarize that in three bullets."
Output: {"decision": "NO-RETRIEVAL", "confidence": 0.98, "reason": "Formatting request, no new search needed."}
</EXAMPLES>

<OUTPUT_FORMAT>
Respond ONLY with a valid JSON object matching this schema:
{
  "decision": "WAIT" | "RETRIEVE" | "NO-RETRIEVAL",
  "confidence": number (0.0 to 1.0),
  "reason": "string"
}
</OUTPUT_FORMAT>"""

class RetrievalController:
    """
    Evaluates streaming user transcripts to decide when to trigger vector retrieval.
    Supports real-time Gemini LLM inference with deterministic fallback for ultra-low latency & resilience.
    """
    def __init__(self, api_key: Optional[str] = None, model_name: Optional[str] = None):
        self.api_key = api_key or os.getenv("GEMINI_API_KEY") or os.getenv("GOOGLE_API_KEY")
        self.openai_key = os.getenv("OPENAI_API_KEY")
        self.model_name = model_name or os.getenv("CONTROLLER_MODEL", "gemini-2.5-flash")
        self._gemini_client = None
        self._init_client()

    def _init_client(self):
        if self.api_key:
            try:
                from google import genai
                self._gemini_client = genai.Client(api_key=self.api_key)
            except Exception as e:
                print(f"[RetrievalController] Warning: Could not initialize google.genai Client: {e}")
                self._gemini_client = None

    async def evaluate(self, transcript: str, previous_context: Optional[str] = None) -> ControllerOutput:
        start_time = time.perf_counter()
        clean_text = transcript.strip()

        # If transcript is empty or whitespace
        if not clean_text:
            return ControllerOutput(
                decision=ControllerDecision.WAIT,
                confidence=1.0,
                reason="Empty transcript.",
                latency_ms=(time.perf_counter() - start_time) * 1000
            )

        # 1. Try Gemini LLM if key is configured
        if self._gemini_client:
            try:
                output = await self._call_gemini(clean_text, previous_context)
                output.latency_ms = (time.perf_counter() - start_time) * 1000
                return output
            except Exception as e:
                print(f"[RetrievalController] LLM inference failed ({e}), falling back to deterministic heuristic.")

        # 2. Deterministic Heuristic Engine (Sub-millisecond latency & offline fallback)
        output = self._deterministic_evaluate(clean_text, previous_context)
        output.latency_ms = (time.perf_counter() - start_time) * 1000
        return output

    async def _call_gemini(self, transcript: str, previous_context: Optional[str] = None) -> ControllerOutput:
        context_block = f"\nPrevious Answer Context: {previous_context}\n" if previous_context else ""
        prompt = f"{RETRIEVAL_CONTROLLER_SYSTEM_PROMPT}\n{context_block}\nInput: \"{transcript}\"\nOutput:"
        
        # Non-blocking async execution
        import asyncio
        response = await asyncio.to_thread(
            self._gemini_client.models.generate_content,
            model=self.model_name,
            contents=prompt,
        )
        
        raw_text = response.text.strip()
        # Clean markdown formatting if present
        if raw_text.startswith("```"):
            raw_text = re.sub(r"^```(?:json)?\n?", "", raw_text)
            raw_text = re.sub(r"\n?```$", "", raw_text)
            raw_text = raw_text.strip()

        data = json.loads(raw_text)
        decision_val = data.get("decision", "WAIT").upper()
        if decision_val not in ["WAIT", "RETRIEVE", "NO-RETRIEVAL"]:
            decision_val = "WAIT"
            
        return ControllerOutput(
            decision=ControllerDecision(decision_val),
            confidence=float(data.get("confidence", 0.9)),
            reason=str(data.get("reason", "Evaluated by LLM"))
        )

    def _deterministic_evaluate(self, transcript: str, previous_context: Optional[str] = None) -> ControllerOutput:
        """
        High-precision deterministic rule evaluator implementing the exact 3 prompt rules.
        """
        text_lower = transcript.lower()

        # Rule 3: NO-RETRIEVAL (Formatting, summarization, or length adjustments without new concepts)
        reformat_patterns = [
            r"^(summarize|summary|summarise)",
            r"^(make it|make this|shorten|expand|rephrase|rewrite)",
            r"^(bullet points|in bullets|in \d+ bullets|table format)",
            r"^(explain like i'?m|simplify that|more detail)",
            r"what did you just say",
            r"^can you rephrase",
        ]
        for pattern in reformat_patterns:
            if re.search(pattern, text_lower):
                return ControllerOutput(
                    decision=ControllerDecision.NO_RETRIEVAL,
                    confidence=0.96,
                    reason="Formatting or summarization request on previous response without new regulatory concepts."
                )

        # Regulatory & Compliance Keywords (triggers Rule 2)
        regulatory_entities = [
            "digital lending", "kyc", "aml", "anti-money laundering", "rbi", "sebi",
            "fintech", "compliance", "capital adequacy", "car", "nbfc", "fldg",
            "first loss default guarantee", "audit", "lending service provider", "lsp",
            "dlsa", "interest rate", "escrow", "data localization", "governance",
            "omap", "credit card", "foreign exchange", "fema", "reporting deadline",
            "penalty", "circular", "master direction", "prudential norm", "disbursement"
        ]

        # Check for regulatory concepts
        for entity in regulatory_entities:
            if entity in text_lower:
                return ControllerOutput(
                    decision=ControllerDecision.RETRIEVE,
                    confidence=0.95,
                    reason=f"Clear regulatory topic or entity identified: '{entity}'."
                )

        # General intent verbs + object indicators
        retrieval_verbs = ["policy regarding", "requirements for", "rules on", "deadline for", "mandate on"]
        for verb in retrieval_verbs:
            if verb in text_lower and len(text_lower.split()) >= 4:
                return ControllerOutput(
                    decision=ControllerDecision.RETRIEVE,
                    confidence=0.90,
                    reason=f"Actionable compliance query pattern detected: '{verb}'."
                )

        # Rule 1: WAIT (Query is ambiguous, missing a core subject, or just starting)
        words = text_lower.split()
        if len(words) < 5:
            return ControllerOutput(
                decision=ControllerDecision.WAIT,
                confidence=0.92,
                reason="Query is in early incomplete stage; awaiting core regulatory subject."
            )

        # General incomplete phrasing
        if any(text_lower.endswith(w) for w in ["the", "a", "an", "our", "for", "with", "about", "new", "does", "is"]):
            return ControllerOutput(
                decision=ControllerDecision.WAIT,
                confidence=0.88,
                reason="Trailing conjunction/preposition indicates user is still forming the subject."
            )

        # Fallback wait if no regulatory entity could be isolated yet
        return ControllerOutput(
            decision=ControllerDecision.WAIT,
            confidence=0.75,
            reason="Ambiguous intent; waiting for specific regulatory domain or entity."
        )
