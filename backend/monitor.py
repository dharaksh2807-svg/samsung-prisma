import os
import re
import json
import time
import asyncio
from typing import List, Optional
from pydantic import BaseModel, Field
from dotenv import load_dotenv

load_dotenv()

# ─────────────────────────────────────────────────────────────────────────────
# Models
# ─────────────────────────────────────────────────────────────────────────────

class StreamMonitorInput(BaseModel):
    previous_query: str = Field(..., description="The original user query")
    previous_answer: str = Field(..., description="The answer previously generated for the user")
    new_doc_id: str = Field(..., description="Document ID of the newly ingested regulatory amendment")
    new_document_text: str = Field(..., description="Full text content of the new regulatory document")

class StreamMonitorOutput(BaseModel):
    update_required: bool
    affected_claims: List[str] = Field(default_factory=list)
    reasoning: str
    latency_ms: float = 0.0

# ─────────────────────────────────────────────────────────────────────────────
# Prompt 5 — Regulatory Stream Monitor
# ─────────────────────────────────────────────────────────────────────────────

STREAM_MONITOR_PROMPT = """<SYSTEM_PERSONA>
You are a Regulatory Knowledge Stream Monitor. A new regulatory amendment has just been published, and you must determine if it invalidates a previous answer given to a user.
</SYSTEM_PERSONA>

<PREVIOUS_STATE>
Previous User Query: "{previous_query}"
Previous Answer Given: "{previous_answer}"
</PREVIOUS_STATE>

<NEW_EVIDENCE>
New Document ID: {new_doc_id}
Document Content: "{new_document_text}"
</NEW_EVIDENCE>

<RULES>
1. Read the Previous Answer and compare its factual claims against the New Evidence.
2. If the New Evidence contradicts, changes, or heavily modifies a claim in the Previous Answer, set `update_required` to true.
3. If the New Evidence is unrelated to the previous answer, set `update_required` to false.
</RULES>

<OUTPUT_FORMAT>
Respond ONLY with valid JSON:
{{
  "update_required": boolean,
  "affected_claims": ["List specific sentences or concepts from the previous answer that are now stale"],
  "reasoning": "Brief explanation"
}}
</OUTPUT_FORMAT>"""


# ─────────────────────────────────────────────────────────────────────────────
# Heuristic Overlap Keywords for Fast Deterministic Fallback
# ─────────────────────────────────────────────────────────────────────────────

# Key regulatory action verbs and modifiers whose presence in both documents
# suggests a high probability of relevance/conflict
REGULATORY_CONFLICT_SIGNALS = [
    "no longer", "hereby repealed", "supersedes", "overrides", "amended to",
    "extended to", "reduced to", "increased to", "must now", "is now required",
    "effective immediately", "waived", "exempted", "prohibited", "permitted",
    "deadline extended", "deadline reduced", "changed from", "replaces",
    "revised", "replaced by", "updated to", "cancelled"
]

# Topic extraction: noun-level bigrams that indicate regulatory subjects
SUBJECT_PATTERN = re.compile(
    r"\b(kyc|lending|loan|credit|npa|capital|reserve|compliance|reporting|"
    r"audit|deposit|interest rate|digital|onboarding|reverification|penalty|"
    r"fine|sanction|license|registration|data locali[sz]ation|consent)\b",
    re.IGNORECASE
)


class RegulatoryStreamMonitor:
    """
    Evaluates if a newly ingested regulatory document invalidates previous answers.
    Implements Prompt 5 with dual-mode execution:
    - LLM-first via Gemini for semantic conflict detection
    - Deterministic fallback via keyword + topic-overlap heuristics
    """

    def __init__(self, api_key: Optional[str] = None, model_name: Optional[str] = None):
        self.api_key = api_key or os.getenv("GEMINI_API_KEY") or os.getenv("GOOGLE_API_KEY")
        self.model_name = model_name or os.getenv("MONITOR_MODEL", "gemini-3.8-flash")
        
        from llm_client import get_llm_client
        self._llm = get_llm_client()

    def _deterministic_evaluate(self, input_data: StreamMonitorInput) -> StreamMonitorOutput:
        """
        Fast heuristic evaluation using topic overlap + conflict signal detection.
        Used when no LLM is available. Reliably handles the core logic for testing.
        """
        start = time.perf_counter()

        prev_answer_lower = input_data.previous_answer.lower()
        new_doc_lower = input_data.new_document_text.lower()

        # 1. Extract subject topics from both documents
        prev_topics = set(m.group(0).lower() for m in SUBJECT_PATTERN.finditer(prev_answer_lower))
        new_topics = set(m.group(0).lower() for m in SUBJECT_PATTERN.finditer(new_doc_lower))
        shared_topics = prev_topics & new_topics

        # 2. Check for explicit conflict/change signals in the new document
        conflict_signals_found = [
            signal for signal in REGULATORY_CONFLICT_SIGNALS
            if signal in new_doc_lower
        ]

        # 3. Decision logic
        update_required = bool(shared_topics and conflict_signals_found)

        # 4. If update required, identify affected claims (sentences in prev answer mentioning shared topics)
        affected_claims = []
        if update_required:
            sentences = [s.strip() for s in re.split(r'(?<=[.!?])\s+', input_data.previous_answer) if s.strip()]
            for sentence in sentences:
                sentence_lower = sentence.lower()
                if any(topic in sentence_lower for topic in shared_topics):
                    affected_claims.append(sentence)

        # 5. Build reasoning
        if update_required:
            reasoning = (
                f"New document [{input_data.new_doc_id}] shares regulatory topics "
                f"({', '.join(sorted(shared_topics))}) with the previous answer and contains "
                f"conflict signals: {conflict_signals_found[:3]}. "
                f"{len(affected_claims)} claim(s) may be stale."
            )
        elif shared_topics and not conflict_signals_found:
            reasoning = (
                f"New document [{input_data.new_doc_id}] covers related topics "
                f"({', '.join(sorted(shared_topics))}) but contains no explicit amendment/conflict signals. "
                f"Answer remains valid."
            )
        else:
            reasoning = (
                f"New document [{input_data.new_doc_id}] is unrelated to the previous answer's topics. "
                f"No update required."
            )

        return StreamMonitorOutput(
            update_required=update_required,
            affected_claims=affected_claims,
            reasoning=reasoning,
            latency_ms=(time.perf_counter() - start) * 1000
        )

    async def evaluate(self, input_data: StreamMonitorInput) -> StreamMonitorOutput:
        """
        Primary evaluation method. Tries Gemini LLM first for semantic analysis,
        then falls back to deterministic heuristic engine.
        """
        start = time.perf_counter()

        if self._llm:
            prompt = STREAM_MONITOR_PROMPT.format(
                previous_query=input_data.previous_query.replace('"', '\\"'),
                previous_answer=input_data.previous_answer.replace('"', '\\"'),
                new_doc_id=input_data.new_doc_id,
                new_document_text=input_data.new_document_text.replace('"', '\\"')
            )
            try:
                response = await self._llm.generate_content(
                    model=self.model_name,
                    contents=prompt,
                )
                text = response.text.strip()
                # Strip markdown code fences if present
                if text.startswith("```"):
                    lines = text.splitlines()
                    lines = lines[1:] if lines[0].startswith("```") else lines
                    lines = lines[:-1] if lines and lines[-1].startswith("```") else lines
                    text = "\n".join(lines).strip()

                parsed = json.loads(text)
                return StreamMonitorOutput(
                    update_required=bool(parsed.get("update_required", False)),
                    affected_claims=parsed.get("affected_claims", []),
                    reasoning=parsed.get("reasoning", "LLM reasoning unavailable."),
                    latency_ms=(time.perf_counter() - start) * 1000
                )
            except Exception as e:
                print(f"[RegulatoryStreamMonitor] LLM evaluation failed ({e}), using heuristic fallback.")

        # Deterministic fallback
        return self._deterministic_evaluate(input_data)
