import os
import time
import json
import re
import asyncio
from typing import Optional, List
from pydantic import BaseModel, Field
from dotenv import load_dotenv

load_dotenv()

# ─────────────────────────────────────────────────────────────────────────────
# Output Models
# ─────────────────────────────────────────────────────────────────────────────

class SubQuery(BaseModel):
    intent: str = Field(description="Short identifier for this sub-intent, e.g. 'kyc_changes'")
    search_query: str = Field(description="Concise keyword query optimised for vector DB retrieval")

class DecomposerOutput(BaseModel):
    sub_queries: List[SubQuery]
    latency_ms: float = 0.0

# ─────────────────────────────────────────────────────────────────────────────
# Prompt 2 — Multi-Intent Decomposer
# ─────────────────────────────────────────────────────────────────────────────

MULTI_INTENT_DECOMPOSER_PROMPT = """<SYSTEM_PERSONA>
You are a specialized query decomposer. Your task is to break down a complex, compound regulatory question into independent, parallel sub-queries optimized for a vector database.
</SYSTEM_PERSONA>

<RULES>
1. Identify all distinct entities, topics, and regulatory conditions.
2. If the query applies a condition to a topic, create a query for that intersection.
3. Keep queries concise, focusing on keywords (nouns, verbs, specific entities).
4. Do not include conversational filler in the sub_queries.
</RULES>

<EXAMPLES>
Input: "What changed in KYC and who is affected for existing customers?"
Output: {
  "sub_queries": [
    {"intent": "kyc_changes", "search_query": "KYC regulatory changes"},
    {"intent": "affected_entities", "search_query": "entities affected existing customers KYC"}
  ]
}

Input: "Explain the digital lending policy."
Output: {
  "sub_queries": [
    {"intent": "digital_lending_policy", "search_query": "digital lending policy requirements"}
  ]
}
</EXAMPLES>

<OUTPUT_FORMAT>
Respond ONLY with a valid JSON object:
{
  "sub_queries": [
    {
      "intent": "string",
      "search_query": "string"
    }
  ]
}
</OUTPUT_FORMAT>"""


class MultiIntentDecomposer:
    """
    Breaks a compound regulatory query into independent parallel sub-queries
    optimised for vector database retrieval (Gate G3).

    Uses Gemini LLM (recommended: Sonnet 4.6 or Gemini Pro) as primary engine,
    with a deterministic heuristic fallback for offline / low-latency use.
    """

    def __init__(self, api_key: Optional[str] = None, model_name: Optional[str] = None):
        self.api_key = api_key or os.getenv("GEMINI_API_KEY") or os.getenv("GOOGLE_API_KEY")
        self.model_name = model_name or os.getenv("DECOMPOSER_MODEL", "gemini-2.5-flash")
        self._gemini_client = None
        self._init_client()

    def _init_client(self):
        if self.api_key:
            try:
                from google import genai
                self._gemini_client = genai.Client(api_key=self.api_key)
            except Exception as e:
                print(f"[MultiIntentDecomposer] Warning: Could not initialise google.genai: {e}")

    async def decompose(self, query: str) -> DecomposerOutput:
        """
        Primary entry point.  Accepts a stable query string (post-RETRIEVE decision)
        and returns independent sub-queries for parallel vector search.
        """
        start = time.perf_counter()
        clean_query = query.strip()

        if not clean_query:
            return DecomposerOutput(
                sub_queries=[SubQuery(intent="empty", search_query="")],
                latency_ms=0.0
            )

        # 1 — Try Gemini LLM
        if self._gemini_client:
            try:
                output = await self._call_gemini(clean_query)
                output.latency_ms = (time.perf_counter() - start) * 1000
                return output
            except Exception as e:
                print(f"[MultiIntentDecomposer] LLM failed ({e}), falling back to heuristic.")

        # 2 — Deterministic heuristic fallback
        output = self._heuristic_decompose(clean_query)
        output.latency_ms = (time.perf_counter() - start) * 1000
        return output

    async def _call_gemini(self, query: str) -> DecomposerOutput:
        prompt = f"{MULTI_INTENT_DECOMPOSER_PROMPT}\n\nInput: \"{query}\"\nOutput:"

        response = await asyncio.to_thread(
            self._gemini_client.models.generate_content,
            model=self.model_name,
            contents=prompt,
        )

        raw = response.text.strip()
        # Strip markdown code fences if present
        raw = re.sub(r"^```(?:json)?\n?", "", raw)
        raw = re.sub(r"\n?```$", "", raw).strip()

        data = json.loads(raw)
        sub_queries = [
            SubQuery(intent=sq["intent"], search_query=sq["search_query"])
            for sq in data.get("sub_queries", [])
        ]
        return DecomposerOutput(sub_queries=sub_queries)

    # ──────────────────────────────────────────────────────────────────────────
    # Deterministic Heuristic Engine
    # ──────────────────────────────────────────────────────────────────────────

    # Regulatory concept patterns: (intent_label, [trigger_phrases])
    REGULATORY_CONCEPTS = [
        ("kyc_requirements",       ["kyc", "know your customer", "customer verification", "cdd", "customer due diligence"]),
        ("digital_lending",        ["digital lending", "online lending", "digital loan", "lending platform"]),
        ("aml_obligations",        ["aml", "anti-money laundering", "money laundering", "suspicious transaction", "str"]),
        ("nbfc_compliance",        ["nbfc", "non-banking financial", "shadow banking"]),
        ("rbi_regulations",        ["rbi", "reserve bank", "rbi circular", "master direction"]),
        ("sebi_regulations",       ["sebi", "securities board", "market regulator", "listing obligations"]),
        ("capital_adequacy",       ["capital adequacy", "car", "tier 1 capital", "tier 2 capital", "crar"]),
        ("interest_rate_policy",   ["interest rate", "lending rate", "mclr", "repo rate", "base rate"]),
        ("data_privacy",           ["data privacy", "data protection", "personal data", "pdpa", "dpdp"]),
        ("fldg_arrangements",      ["fldg", "first loss default guarantee", "credit guarantee"]),
        ("escrow_requirements",    ["escrow", "nodal account", "collection account"]),
        ("audit_compliance",       ["audit", "internal audit", "statutory audit", "rbi audit"]),
        ("foreign_exchange",       ["fema", "foreign exchange", "forex", "cross-border", "remittance"]),
        ("penalty_provisions",     ["penalty", "fine", "enforcement action", "violation", "non-compliance"]),
        ("reporting_requirements", ["reporting", "disclosure", "filing", "return", "reporting deadline"]),
        ("existing_customers",     ["existing customer", "current customer", "legacy customer", "existing borrower"]),
        ("new_customers",          ["new customer", "fresh applicant", "onboarding", "new borrower"]),
    ]

    def _heuristic_decompose(self, query: str) -> DecomposerOutput:
        text_lower = query.lower()
        matched: List[SubQuery] = []
        seen_intents = set()

        for intent_label, triggers in self.REGULATORY_CONCEPTS:
            for trigger in triggers:
                if trigger in text_lower and intent_label not in seen_intents:
                    # Build a concise keyword search query
                    search_query = self._build_search_query(trigger, text_lower)
                    matched.append(SubQuery(intent=intent_label, search_query=search_query))
                    seen_intents.add(intent_label)
                    break  # move to next concept once matched

        # Fallback: treat whole query as single intent
        if not matched:
            # Clean conversational filler words
            clean = re.sub(
                r"\b(please|can you|could you|tell me|what is|what are|explain|describe|how does|i want to know|i need)\b",
                "", text_lower
            )
            clean = re.sub(r"\s+", " ", clean).strip()
            matched.append(SubQuery(
                intent="general_regulatory_query",
                search_query=clean if clean else query
            ))

        return DecomposerOutput(sub_queries=matched)

    def _build_search_query(self, trigger: str, full_text: str) -> str:
        """
        Builds a concise keyword-focused search query from the trigger and surrounding context.
        Removes filler words and keeps noun/verb phrases.
        """
        # Extract up to 7-word window around the trigger
        words = full_text.split()
        filler = {
            "the", "a", "an", "is", "are", "was", "were", "does", "do", "did",
            "please", "can", "could", "would", "should", "will", "i", "me", "you",
            "tell", "explain", "what", "how", "who", "when", "where", "for",
            "about", "with", "on", "of", "to", "and", "or", "in", "any",
            "there", "that", "this", "our", "its", "their", "these", "those", "each"
        }
        keywords = [w.rstrip("?.!,") for w in words if w not in filler and len(w) > 2]
        # Prepend the trigger to ensure it's prominent
        if trigger not in " ".join(keywords):
            keywords.insert(0, trigger)
        return " ".join(keywords[:8])
