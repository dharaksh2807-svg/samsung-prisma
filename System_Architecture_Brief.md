# RegulaStream: System Architecture Brief
**Theme 4: Streaming Live RAG (Samsung PRISM Hackathon)**

## 1. System Design Rationale
RegulaStream is built on an event-driven, streaming micro-pipeline architecture. It abandons the traditional batch-oriented RAG cycle in favor of **Incremental Streaming**, processing WebSockets chunk-by-chunk. To satisfy the "Architectural Parsimony" constraint, we opted against overly complex agentic frameworks. Instead, we use highly deterministic heuristics for low-latency routing, backed by Google Gemini for deep semantic resolution.

## 2. Retrieval Trigger Logic (Controller)
The **Retrieval Controller** uses a dual-engine architecture:
*   **Primary Filter (Deterministic):** A sub-millisecond regex/keyword engine that evaluates every transcript chunk. It triggers `WAIT` if the utterance lacks a regulatory subject, `NO-RETRIEVAL` for formatting requests ("make this shorter"), and `RETRIEVE` upon detecting compliance entities (e.g., "KYC", "digital lending").
*   **Secondary Evaluator (LLM):** If the heuristic is uncertain, an async Gemini call evaluates the semantic completeness of the sentence.

## 3. Query Decomposition Strategy
Compound queries (e.g., "What are the KYC rules and how does it affect NRIs?") are piped to the **Multi-Intent Decomposer**.
*   We utilize structural parsing to split the utterance into `[Query A]` and `[Query B]`.
*   These sub-queries are executed in parallel against the Vector Index, preventing token exhaustion and keeping context windows pure.

## 4. Dense/Sparse Hybrid Scoring & Fusion
Our Vector Engine strictly avoids naive dense-only retrieval:
1.  **Sparse Scoring:** A custom BM25 implementation scores keyword overlap and frequency.
2.  **Dense Scoring:** Gemini `text-embedding-004` evaluates deep semantic cosine similarity.
3.  **Reciprocal Rank Fusion (RRF):** The scores are mathematically fused to produce a deduplicated, highly accurate top-K chunk list.

## 5. Data Provenance & Factual Grounding
To guarantee zero hallucination (Gate G4), our **Session-Aware Synthesizer** operates under strict prompts to inject exact `[DOC_ID]` citations. 
*   **Citation Validator:** An interception layer that cross-references all emitted `[DOC_ID]` tags against the genuine retrieved corpus. Fabricated citations are aggressively stripped before reaching the client.

## 6. Late-Arriving Constraints & Session Refinement
When a user adds a constraint ("what about existing users?"), our **Session Manager** executes a Delta-Query. It preserves the previously retrieved evidence array in memory, fetches ONLY the delta context, merges them, and re-synthesizes the answer in-place (Gate G5).

## 7. Failure Mode Mitigations
*   **Rate Limits:** Rotational LLM client with Tenacity backoff guarantees uptime.
*   **Out of Domain:** Strict fallback prompts guarantee the system answers "Insufficient evidence" rather than hallucinating external knowledge.
