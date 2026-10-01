# RegulaStream - Real-Time Regulatory Intelligence

## 1. Project Overview
RegulaStream is a Streaming Live RAG system built for the Theme 4 Hackathon. It processes a user's evolving compliance query in real-time, predicts intents, retrieves early, decomposes compound questions, and maintains a session state that is updated automatically if the underlying regulatory corpus changes.

## 2. Problem
Traditional RAG waits for a complete question before retrieving, adding latency and treating every question independently. In compliance, regulations change and users often refine questions mid-stream. RegulaStream retrieves *as* the question develops.

## 3. Architecture & Features
- **Early Retrieval:** Fetches context before query completion.
- **Multi-Intent Parallel Search:** Decomposes complex queries (e.g. "What is KYC and who needs it?").
- **State-Preserving Refinement:** Applies late constraints without a full re-search.
- **Claim-Level Evidence Mapping:** Validates every citation against the corpus.
- **Regulatory Stream Listener:** Auto-updates answers when underlying documents change.

## 4. Tech Stack
- Frontend: React + Vite
- Backend: Python + FastAPI + WebSockets
- AI/RAG: FAISS + Sentence Transformers + LLMs
- Benchmark: Automated `replay.py` harness

## 5. Running Locally (Reproducibility)
Launch the entire stack using Docker Compose:
```bash
docker compose up
```

Run the benchmark evaluation:
```bash
python benchmark/replay.py
```
