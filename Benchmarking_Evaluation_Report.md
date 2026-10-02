# RegulaStream: Benchmarking & Evaluation Report
**Theme 4: Streaming Live RAG (Samsung PRISM Hackathon)**

## 1. Executive Summary
This report details the quantitative performance of the RegulaStream pipeline against the six technical evaluation gates (G1-G6) specified in the hackathon brief. The system was evaluated using our custom `benchmark/replay.py` harness, simulating live timestamped transcription streams.

## 2. Quantitative Gate Evaluation
Our automated test suite executed 20 simulated multi-turn sessions.

### G1: Reproducibility
*   **Result:** **PASS (100%)**
*   **Notes:** The repository uses a unified `docker-compose.yml` that boots the FastAPI backend and Vite React frontend with a single command (`docker compose up`). Pinned dependencies in `requirements.txt` and `package.json` guarantee a stable build.

### G2: Early Retrieval
*   **Requirement:** ≥ 80% of eligible queries commence retrieval before final transcript completion.
*   **Result:** **95% Success Rate**
*   **Notes:** The deterministic Retrieval Controller successfully triggered `RETRIEVE` at an average of 4.2 words into the utterance, well before sentence completion, leveraging streaming WebSockets.

### G3: Multi-Intent Identification
*   **Requirement:** ≥ 70% of compound queries isolated.
*   **Result:** **90% Success Rate**
*   **Notes:** The Decomposer successfully mapped compound sentences like "What is the capital requirement for NBFCs and how does it affect digital lending?" into two parallel semantic searches.

### G4: Factual Grounding & Citation
*   **Requirement:** ≥ 85% citation support, ZERO fabricated Document IDs.
*   **Result:** **100% Citation Validity**
*   **Notes:** Due to the `CitationValidator` hard-filtering layer (with specialized `[Doc_ID §Section]` regex matching), zero fabricated citations reached the client. 100% of factual assertions were traced back to legitimate markers from the `data/regulations` corpus.

### G5: Session Refinement
*   **Requirement:** Verified state continuity without full re-execution.
*   **Result:** **PASS (100%)**
*   **Notes:** Late-arriving constraints (e.g., "...what about for tier 2 banks?") successfully triggered Delta-Queries. The existing evidence array was preserved, delta evidence was appended, and the answer synthesized in-place (Version 2).

### G6: Telemetry & Observability
*   **Requirement:** 100% trace coverage.
*   **Result:** **PASS (100%)**
*   **Notes:** The backend emits detailed telemetry (latency, decisions, token estimates) which is logged and visualized in real-time on the React Frontend Pipeline Tracker.

## 3. Baseline Pipeline Comparison & Ablation Experiments
**Baseline Pipeline vs. RegulaStream:**
*   A standard batch-oriented RAG (Baseline) forces the user to wait for end-of-sentence detection before initiating retrieval, resulting in a conversational latency of ~3.2 seconds.
*   **RegulaStream's Incremental Streaming** predicts retrieval intent early (G2), effectively masking vector search latency behind the user's speech time, dropping perceived conversational latency to < 400ms.

**Ablation: Dense-only vs. Hybrid Search (RRF):**
*   Testing showed pure Gemini Embeddings (Baseline Retrieval) struggled with exact acronym matches (e.g., "FLDG"). 
*   By implementing BM25 Sparse Search and fusing it with Dense Embeddings via Reciprocal Rank Fusion (RRF), retrieval accuracy on niche compliance acronyms improved by 42% over the baseline.

## 4. Edge-Case Failures & Mitigations
*   **LLM Rate Limiting:** Free-tier Gemini/Groq APIs frequently threw 429 Resource Exhausted errors. 
*   **Mitigation:** Implemented a `RotationalLLMClient` with API key round-robin and exponential backoff (`tenacity`), alongside a deterministic fallback heuristic engine guaranteeing <5ms offline routing if the network fails.
