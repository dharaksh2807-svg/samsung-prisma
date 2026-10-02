import os
import re
import json
import time
import asyncio
import uuid
from typing import List, Dict, Optional, Any
from pydantic import BaseModel, Field
from dotenv import load_dotenv

load_dotenv()

# ─────────────────────────────────────────────────────────────────────────────
# Models & Schemas (Hackathon Gate G5: Late Refinement & State Preservation)
# ─────────────────────────────────────────────────────────────────────────────

class EvidenceItem(BaseModel):
    doc_id: str
    text: str
    metadata: Optional[Dict[str, Any]] = None

class SessionHistoryEntry(BaseModel):
    query: str
    answer: str
    version: int
    timestamp: float = Field(default_factory=time.time)

class SessionState(BaseModel):
    session_id: str
    history: List[SessionHistoryEntry] = Field(default_factory=list)
    current_evidence: List[EvidenceItem] = Field(default_factory=list)
    answer_version: int = 1
    created_at: float = Field(default_factory=time.time)
    updated_at: float = Field(default_factory=time.time)

class InitSessionRequest(BaseModel):
    session_id: Optional[str] = None
    query: str
    answer: str
    evidence: List[EvidenceItem] = Field(default_factory=list)

class DeltaQueryInput(BaseModel):
    session_id: str
    new_constraint: str

class DeltaQueryOutput(BaseModel):
    session_id: str
    is_refinement: bool
    search_query: str
    answer_version: int
    cached_evidence_ids: List[str] = Field(default_factory=list)
    latency_ms: float = 0.0

class LateRefinementRequest(BaseModel):
    session_id: str
    new_constraint: str
    delta_evidence: Optional[List[EvidenceItem]] = Field(default_factory=list)

class LateRefinementResponse(BaseModel):
    session_id: str
    is_refinement: bool
    delta_query: str
    answer_version: int
    combined_evidence_ids: List[str]
    refined_answer: str
    latency_ms: float = 0.0

# ─────────────────────────────────────────────────────────────────────────────
# Prompt 6 — Delta Query Generator (Gate G5)
# ─────────────────────────────────────────────────────────────────────────────

DELTA_QUERY_PROMPT = """<SYSTEM_PERSONA>
You are a Delta Query Generator. Your job is to take an incomplete or highly contextual user constraint and resolve it into a standalone search query using the previous conversation context.
</SYSTEM_PERSONA>

<CONTEXT>
Previous Query: "{previous_query}"
Previous Answer: "{previous_answer}"
</CONTEXT>

<NEW_CONSTRAINT>
User Input: "{new_constraint}"
</NEW_CONSTRAINT>

<RULES>
1. If the New Constraint is a continuation or refinement (e.g., "what about for minors?"), combine it with the core topic of the Previous Query.
2. If it is a completely new topic, output the New Constraint as-is.
3. Keep the output concise and optimized for vector similarity search (nouns and keywords).
</RULES>

<OUTPUT_FORMAT>
Respond ONLY with valid JSON:
{{
  "is_refinement": boolean,
  "search_query": "string"
}}
</OUTPUT_FORMAT>"""

# ─────────────────────────────────────────────────────────────────────────────
# Deterministic Refinement Signals
# ─────────────────────────────────────────────────────────────────────────────

REFINEMENT_PREFIX_PATTERNS = [
    r"^(\.\.\.|and\s+|only\s+|what\s+about\s+|how\s+about\s+|specifically\s+|for\s+|except\s+|including\s+|applied\s+to\s+|just\s+)",
    r"(existing\s+customers|new\s+users|digital\s+lending|minors|nri|tier\s*1|tier\s*2|small\s+banks)"
]

SUBJECT_KEYWORDS = [
    "kyc", "lending", "loan", "reverification", "audit", "deposit", "reporting",
    "compliance", "digital lending", "interest rate", "npa", "capital adequacy",
    "onboarding", "sanction", "data localization"
]

class DeltaQueryGenerator:
    """
    Generates standalone vector search queries from late user constraints.
    Supports dual execution:
    - Gemini 3.8 Flash / Pro for high-intelligence context resolution
    - Deterministic heuristic engine for sub-5ms low latency fallback
    """

    def __init__(self, api_key: Optional[str] = None, model_name: Optional[str] = None):
        self.api_key = api_key or os.getenv("GEMINI_API_KEY") or os.getenv("GOOGLE_API_KEY")
        self.model_name = model_name or os.getenv("DELTA_MODEL", "gemini-3.8-flash")
        
        from llm_client import get_llm_client
        self._llm = get_llm_client()

    def _deterministic_generate(
        self,
        previous_query: str,
        previous_answer: str,
        new_constraint: str
    ) -> Dict[str, Any]:
        """
        Fast heuristic resolver for delta constraints.
        Extracts core topic from previous context and merges with constraint.
        """
        clean_constraint = new_constraint.strip()
        constraint_lower = clean_constraint.lower()

        # Check for refinement signals
        is_refinement = False
        for pat in REFINEMENT_PREFIX_PATTERNS:
            if re.search(pat, constraint_lower, re.IGNORECASE):
                is_refinement = True
                break

        # Also consider short phrases (< 6 words) without explicit subjects as refinements
        words = clean_constraint.split()
        if len(words) <= 5 and not clean_constraint.lower().startswith(("what is", "how do", "explain", "describe")):
            is_refinement = True

        if is_refinement and previous_query:
            # Extract key topic keywords from previous query
            prev_lower = previous_query.lower()
            matched_topics = [kw for kw in SUBJECT_KEYWORDS if kw in prev_lower]
            
            # If no keyword matched, extract dominant noun phrase or simply combine
            if matched_topics:
                core_topic = " ".join(matched_topics)
            else:
                # Strip question words
                stripped_prev = re.sub(r"^(what\s+is|what\s+are|how\s+to|explain|describe)\s+", "", prev_lower, flags=re.IGNORECASE).strip(" ?.")
                core_topic = " ".join(stripped_prev.split()[:4])

            # Strip leading continuation noise from constraint
            cleaned_c = re.sub(r"^(\.\.\.|and\s+|only\s+|specifically\s+|just\s+)", "", clean_constraint, flags=re.IGNORECASE).strip()
            
            search_query = f"{core_topic} {cleaned_c}".strip()
        else:
            search_query = clean_constraint

        return {
            "is_refinement": is_refinement,
            "search_query": search_query
        }

    async def generate(
        self,
        previous_query: str,
        previous_answer: str,
        new_constraint: str
    ) -> Dict[str, Any]:
        """
        Resolves constraint into a standalone search query.
        """
        start = time.perf_counter()

        if self._llm:
            prompt = DELTA_QUERY_PROMPT.format(
                previous_query=previous_query.replace('"', '\\"'),
                previous_answer=previous_answer.replace('"', '\\"'),
                new_constraint=new_constraint.replace('"', '\\"')
            )
            try:
                response = await self._llm.generate_content(
                    model=self.model_name,
                    contents=prompt
                )
                text = response.text.strip()
                if text.startswith("```"):
                    lines = text.splitlines()
                    lines = lines[1:] if lines[0].startswith("```") else lines
                    lines = lines[:-1] if lines and lines[-1].startswith("```") else lines
                    text = "\n".join(lines).strip()

                parsed = json.loads(text)
                res = {
                    "is_refinement": bool(parsed.get("is_refinement", True)),
                    "search_query": str(parsed.get("search_query", new_constraint)),
                    "latency_ms": (time.perf_counter() - start) * 1000
                }
                return res
            except Exception as e:
                print(f"[DeltaQueryGenerator] LLM failed ({e}), falling back to deterministic heuristic.")

        fallback = self._deterministic_generate(previous_query, previous_answer, new_constraint)
        fallback["latency_ms"] = (time.perf_counter() - start) * 1000
        return fallback

# ─────────────────────────────────────────────────────────────────────────────
# Session State Manager (Hackathon Gate G5)
# ─────────────────────────────────────────────────────────────────────────────

class SessionManager:
    """
    Maintains user conversation state, cached retrieved chunks, and answer versions.
    Allows late refinements to query ONLY the delta and combine with cached context.
    """

    def __init__(self, delta_generator: Optional[DeltaQueryGenerator] = None):
        self._sessions: Dict[str, SessionState] = {}
        self.delta_generator = delta_generator or DeltaQueryGenerator()

    def check_semantic_cache(self, session_id: str, query: str, threshold: float = 0.90) -> Optional[str]:
        session = self.get_session(session_id)
        if not session or not session.history:
            return None
        import re
        from difflib import SequenceMatcher
        q_clean = re.sub(r"[^a-zA-Z0-9s]", "", query.lower()).strip()
        for entry in reversed(session.history):
            e_clean = re.sub(r"[^a-zA-Z0-9s]", "", entry.query.lower()).strip()
            if not q_clean or not e_clean: continue
            if SequenceMatcher(None, q_clean, e_clean).ratio() >= threshold:
                return entry.answer
        return None


    def get_or_create_session(self, session_id: Optional[str] = None) -> SessionState:
        # Prevent memory leak by capping max active sessions
        if len(self._sessions) > 1000:
            # Evict oldest 100 sessions
            sorted_sessions = sorted(self._sessions.items(), key=lambda x: x[1].updated_at)
            for k, _ in sorted_sessions[:100]:
                del self._sessions[k]
                
        sid = session_id or str(uuid.uuid4())
        if sid not in self._sessions:
            self._sessions[sid] = SessionState(session_id=sid)
        return self._sessions[sid]

    def get_session(self, session_id: str) -> Optional[SessionState]:
        return self._sessions.get(session_id)

    def record_initial_turn(
        self,
        session_id: str,
        query: str,
        answer: str,
        evidence: List[EvidenceItem]
    ) -> SessionState:
        session = self.get_or_create_session(session_id)
        session.history.append(SessionHistoryEntry(
            query=query,
            answer=answer,
            version=session.answer_version
        ))
        session.current_evidence = evidence
        session.updated_at = time.time()
        return session

    async def prepare_delta_query(
        self,
        session_id: str,
        new_constraint: str
    ) -> DeltaQueryOutput:
        """
        Inspects session state and generates the targeted delta query
        without clearing previously retrieved evidence chunks.
        """
        start = time.perf_counter()
        session = self.get_or_create_session(session_id)

        last_query = session.history[-1].query if session.history else ""
        last_answer = session.history[-1].answer if session.history else ""

        delta_info = await self.delta_generator.generate(
            previous_query=last_query,
            previous_answer=last_answer,
            new_constraint=new_constraint
        )

        cached_ids = [e.doc_id for e in session.current_evidence]

        return DeltaQueryOutput(
            session_id=session.session_id,
            is_refinement=delta_info["is_refinement"],
            search_query=delta_info["search_query"],
            answer_version=session.answer_version,
            cached_evidence_ids=cached_ids,
            latency_ms=(time.perf_counter() - start) * 1000
        )

    def apply_refinement(
        self,
        session_id: str,
        new_constraint: str,
        refined_answer: str,
        delta_evidence: Optional[List[EvidenceItem]] = None
    ) -> SessionState:
        """
        Updates session evidence by merging delta evidence with existing evidence,
        bumps answer_version, and records the new state.
        """
        session = self.get_or_create_session(session_id)
        session.answer_version += 1

        # Merge evidence avoiding duplicate doc_ids
        existing_doc_ids = {e.doc_id for e in session.current_evidence}
        if delta_evidence:
            for item in delta_evidence:
                if item.doc_id not in existing_doc_ids:
                    session.current_evidence.append(item)
                    existing_doc_ids.add(item.doc_id)

        session.history.append(SessionHistoryEntry(
            query=new_constraint,
            answer=refined_answer,
            version=session.answer_version
        ))
        session.updated_at = time.time()
        return session

    def get_combined_evidence(self, session_id: str) -> List[EvidenceItem]:
        session = self.get_session(session_id)
        if not session:
            return []
        return session.current_evidence

    def clear_session(self, session_id: str) -> bool:
        if session_id in self._sessions:
            del self._sessions[session_id]
            return True
        return False
