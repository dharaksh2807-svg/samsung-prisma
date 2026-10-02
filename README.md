# RegulaStream: Streaming Live RAG 
**Theme 4 Hackathon Submission (Samsung PRISM)**

## 1. Executive Summary
RegulaStream abandons the traditional static, batch-oriented Retrieval-Augmented Generation (RAG) turn cycle. It is built for real-time support, processing incoming transcript streams incrementally. It successfully achieves sub-millisecond retrieval triggers, multi-intent query decomposition, and stateful session refinement without hallucinating citations.

## 2. Technical Evaluation Gates (G1–G6)
Our implementation has been rigorously benched against all six hard evaluation gates.

*   ✅ **G1: Reproducibility:** Containerized environment. Launch the full stack (Frontend + Backend) with a single command: `docker compose up --build`.
*   ✅ **G2: Early Retrieval:** WebSocket streaming evaluates chunks in real-time, commencing retrieval speculatively before the user finishes typing/speaking.
*   ✅ **G3: Multi-Intent Identification:** Compound queries are caught by our `MultiIntentDecomposer` and parallelized (e.g., "What is KYC and how does it affect NRIs?").
*   ✅ **G4: Factual Grounding:** Zero fabricated Document IDs. The `CitationValidator` uses deterministic regex to strip any LLM hallucinations not present in the indexed corpus.
*   ✅ **G5: Session Refinement:** Late-arriving constraints trigger Delta-Queries via the `SessionManager`, applying updates in-place without clearing history or re-executing full-corpus searches.
*   ✅ **G6: Telemetry & Observability:** The pipeline emits structured trace coverage across all micro-stages. (See the Live Pipeline Tracker in the UI).

## 3. Hard Engineering Rules Adherence
*   **Corpus Isolation:** All evidence is strictly derived from the provided local `data/regulations` corpus.
*   **Dense/Sparse Hybrid Scoring:** We eschewed naive semantic search in favor of a robust BM25 (Sparse) + Gemini (Dense) pipeline mapped via Reciprocal Rank Fusion.
*   **Architectural Parsimony:** Built efficiently using asynchronous Python, avoiding heavy multi-agent orchestration frameworks to maximize speed-to-cost ratio.

## 4. Deliverables Checklist
This repository includes all required hackathon deliverables:
*   [x] **Reproducible Repository:** (`docker-compose.yml`, `requirements.txt`, `package.json`).
*   [x] **System Architecture Brief (≤ 6 pages):** Provided as `System_Architecture_Brief.md` in the repository root.
*   [x] **Benchmarking & Evaluation Report:** Provided as `Benchmarking_Evaluation_Report.md` in the repository root.
*   [x] **Telemetry & Observability Schema:** Formal JSON Schema provided in `telemetry_schema.json`, explicitly modeling latency, answer version lineage, and inference cost estimations.
*   [x] **System Demonstration Video:** *(To be submitted by user).*

## 5. Running the Pipeline Locally

**Boot the System:**
```bash
docker compose up --build
```
*Frontend runs on `http://localhost:5173` | Backend runs on `http://localhost:8000`*

**Run the Automated Benchmark Replay:**
```bash
python3 benchmark/replay.py
```
