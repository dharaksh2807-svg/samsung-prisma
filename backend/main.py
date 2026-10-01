from fastapi import FastAPI, WebSocket, WebSocketDisconnect, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from typing import Optional, List, Dict, Any
import time

from controller import RetrievalController, ControllerOutput, ControllerDecision
from decomposer import MultiIntentDecomposer, DecomposerOutput
from synthesizer import StatefulSynthesizer, SynthesizerInput, SynthesizerOutput
from validator import CitationValidator, ValidatorInput, ValidatorOutput
from monitor import RegulatoryStreamMonitor, StreamMonitorInput, StreamMonitorOutput
from session import (
    SessionManager,
    DeltaQueryInput,
    DeltaQueryOutput,
    LateRefinementRequest,
    LateRefinementResponse,
    EvidenceItem,
    SessionState,
    InitSessionRequest
)
from retriever import SimpleRetriever, RetrieveRequest, RetrieverOutput

app = FastAPI(title="RegulaStream API", version="1.0.0")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Global instances
controller = RetrievalController()
decomposer = MultiIntentDecomposer()
synthesizer = StatefulSynthesizer()
validator = CitationValidator()
monitor = RegulatoryStreamMonitor()
session_manager = SessionManager()
retriever = SimpleRetriever()

@app.get("/")
def read_root():
    return {
        "service": "RegulaStream API",
        "status": "online",
        "controller_model": controller.model_name,
        "decomposer_model": decomposer.model_name,
        "synthesizer_model": synthesizer.model_name,
        "validator_model": validator.model_name,
        "monitor_model": monitor.model_name,
        "session_manager_ready": True,
        "llm_ready": getattr(controller, "_llm", None) is not None
    }

class EvaluateRequest(BaseModel):
    transcript: str
    previous_context: Optional[str] = None

class DecomposeRequest(BaseModel):
    query: str

@app.post("/api/controller/evaluate", response_model=ControllerOutput)
async def evaluate_transcript(request: EvaluateRequest):
    """
    Direct REST endpoint to evaluate whether live user speech has reached
    sufficient intent to trigger retrieval.
    """
    try:
        output = await controller.evaluate(request.transcript, request.previous_context)
        return output
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/api/decomposer/decompose", response_model=DecomposerOutput)
async def decompose_query(request: DecomposeRequest):
    """
    REST endpoint to decompose a stable compound query into parallel sub-queries
    for multi-intent vector retrieval (Hackathon Gate G3).
    """
    try:
        output = await decomposer.decompose(request.query)
        return output
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/api/retriever/search", response_model=RetrieverOutput)
async def search_documents(request: RetrieveRequest):
    """
    REST endpoint to search and retrieve relevant regulatory chunks from the file system.
    """
    try:
        output = await retriever.search(request.queries)
        return output
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

from fastapi.responses import PlainTextResponse

@app.get("/api/retriever/doc/{doc_id}")
async def get_document_endpoint(doc_id: str):
    """Serve plain text of a retrieved document so citations can be clicked in the UI."""
    doc_text = retriever.get_document(doc_id)
    if doc_text == "Document not found.":
        raise HTTPException(status_code=404, detail="Document not found")
    return PlainTextResponse(doc_text)

@app.post("/api/synthesizer/synthesize", response_model=SynthesizerOutput)
async def synthesize_answer(request: SynthesizerInput):
    """
    REST endpoint to generate the final response strictly grounded in retrieved evidence.
    """
    try:
        output = await synthesizer.synthesize(request)
        return output
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/api/validator/validate", response_model=ValidatorOutput)
async def validate_citations(request: ValidatorInput):
    """
    REST endpoint to audit compliance citations against authentic corpus IDs (Hackathon Gate G4).
    """
    try:
        output = await validator.validate(request)
        return output
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/api/monitor/evaluate", response_model=StreamMonitorOutput)
async def evaluate_knowledge_stream(request: StreamMonitorInput):
    """
    REST endpoint to detect if a new regulatory document invalidates a prior user answer.
    Drives the Knowledge Update Stream and triggers answer re-synthesis (Hackathon Differentiation).
    """
    try:
        output = await monitor.evaluate(request)
        return output
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.websocket("/ws/stream")
async def websocket_endpoint(websocket: WebSocket):
    """
    WebSocket endpoint for real-time live typing / speech stream.
    Receives text chunks and streams back immediate Retrieval Controller decisions.
    """
    await websocket.accept()
    try:
        while True:
            data = await websocket.receive_json()
            chunk = data.get("chunk", "")
            timestamp = data.get("timestamp", time.time())
            previous_context = data.get("previous_context", None)
            
            # Evaluate using the Retrieval Controller
            decision_output = await controller.evaluate(chunk, previous_context)
            
            response = {
                "event": "CONTROLLER_DECISION",
                "timestamp": timestamp,
                "chunk": chunk,
                "decision": decision_output.decision.value,
                "confidence": decision_output.confidence,
                "reason": decision_output.reason,
                "latency_ms": round(decision_output.latency_ms, 2)
            }

            # Auto-decompose into parallel sub-queries on RETRIEVE decisions
            if decision_output.decision == ControllerDecision.RETRIEVE:
                decompose_output = await decomposer.decompose(chunk)
                response["sub_queries"] = [
                    {"intent": sq.intent, "search_query": sq.search_query}
                    for sq in decompose_output.sub_queries
                ]
                response["decompose_latency_ms"] = round(decompose_output.latency_ms, 2)


            await websocket.send_json(response)
    except WebSocketDisconnect:
        pass

# ─────────────────────────────────────────────────────────────────────────────
# Session State & Late Refinement Endpoints (Gate G5)
# ─────────────────────────────────────────────────────────────────────────────

@app.post("/api/session/init", response_model=SessionState)
async def init_session(request: InitSessionRequest):
    """
    Initializes a new session or records the first turn with its query,
    answer, and retrieved evidence chunks.
    """
    try:
        session = session_manager.record_initial_turn(
            session_id=request.session_id or str(uuid.uuid4()),
            query=request.query,
            answer=request.answer,
            evidence=request.evidence
        )
        return session
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/api/session/delta_query", response_model=DeltaQueryOutput)
async def generate_delta_query(request: DeltaQueryInput):
    """
    Generates a targeted delta search query from a late constraint without clearing session state.
    """
    try:
        output = await session_manager.prepare_delta_query(
            session_id=request.session_id,
            new_constraint=request.new_constraint
        )
        return output
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/api/session/refine", response_model=LateRefinementResponse)
async def refine_session(request: LateRefinementRequest):
    """
    Executes late refinement:
    1. Prepares delta search query.
    2. Combines cached evidence with new delta evidence.
    3. Re-synthesizes updated answer with version bump.
    """
    start = time.perf_counter()
    try:
        # 1. Delta query resolution
        delta_info = await session_manager.prepare_delta_query(
            session_id=request.session_id,
            new_constraint=request.new_constraint
        )

        session = session_manager.get_session(request.session_id)
        if not session:
            raise HTTPException(status_code=404, detail=f"Session {request.session_id} not found")

        # 2. Evidence consolidation
        cached_evidence = session.current_evidence
        new_evidence = request.delta_evidence or []
        combined_evidence = list(cached_evidence)
        existing_ids = {e.doc_id for e in cached_evidence}
        for ne in new_evidence:
            if ne.doc_id not in existing_ids:
                combined_evidence.append(ne)
                existing_ids.add(ne.doc_id)

        # 3. Format evidence for Synthesizer
        formatted_chunks = [
            {"doc_id": e.doc_id, "text": e.text}
            for e in combined_evidence
        ]
        history_strings = [
            f"Q: {h.query} -> A: {h.answer}"
            for h in session.history
        ]

        synth_res = await synthesizer.synthesize(SynthesizerInput(
            current_query=f"{session.history[-1].query} (Refinement: {request.new_constraint})",
            retrieved_chunks=formatted_chunks,
            session_history=history_strings
        ))

        # 4. Apply refinement in Session Manager
        updated_session = session_manager.apply_refinement(
            session_id=request.session_id,
            new_constraint=request.new_constraint,
            refined_answer=synth_res.answer_markdown,
            delta_evidence=new_evidence
        )

        return LateRefinementResponse(
            session_id=updated_session.session_id,
            is_refinement=delta_info.is_refinement,
            delta_query=delta_info.search_query,
            answer_version=updated_session.answer_version,
            combined_evidence_ids=[e.doc_id for e in updated_session.current_evidence],
            refined_answer=synth_res.answer_markdown,
            latency_ms=(time.perf_counter() - start) * 1000
        )
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/api/session/{session_id}", response_model=SessionState)
def get_session_state(session_id: str):
    """
    Retrieves full lineage and evidence context for a session.
    """
    session = session_manager.get_session(session_id)
    if not session:
        raise HTTPException(status_code=404, detail="Session not found")
    return session

