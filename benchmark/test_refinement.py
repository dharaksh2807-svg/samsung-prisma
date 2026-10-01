import requests
import time

BASE_URL = "http://localhost:8000/api"
SESSION_ID = "test-session-123"

print("--- 1. FIRST QUERY ---")
query_1 = "What are the KYC requirements for digital lending platforms?"
print(f"User: {query_1}")
dec_1 = requests.post(f"{BASE_URL}/decomposer/decompose", json={"query": query_1}).json()
search_queries_1 = [sq["search_query"] for sq in dec_1.get("sub_queries", [])]
ret_1 = requests.post(f"{BASE_URL}/retriever/search", json={"queries": search_queries_1}).json()
chunks_1 = ret_1.get("chunks", [])

synth_1 = requests.post(f"{BASE_URL}/synthesizer/synthesize", json={
    "current_query": query_1,
    "retrieved_chunks": chunks_1,
    "session_history": []
}).json()
ans_1 = synth_1.get("answer_markdown", "")
print(f"System:\n{ans_1}\n")

# Save session
requests.post(f"{BASE_URL}/session/init", json={
    "session_id": SESSION_ID,
    "query": query_1,
    "answer": ans_1,
    "evidence": [{"doc_id": c["doc_id"], "text": c["text"]} for c in chunks_1]
})

print("--- 2. LATE REFINEMENT ---")
query_2 = "What if the customer is taking a $3,000 crypto loan?"
print(f"User: {query_2}")
dec_2 = requests.post(f"{BASE_URL}/decomposer/decompose", json={"query": query_2}).json()
search_queries_2 = [sq["search_query"] for sq in dec_2.get("sub_queries", [])]
ret_2 = requests.post(f"{BASE_URL}/retriever/search", json={"queries": search_queries_2}).json()
chunks_2 = ret_2.get("chunks", [])

# Combine old evidence with new
existing_ids = {c["doc_id"] for c in chunks_1}
all_chunks = list(chunks_1)
for c in chunks_2:
    if c["doc_id"] not in existing_ids:
        all_chunks.append(c)

synth_2 = requests.post(f"{BASE_URL}/synthesizer/synthesize", json={
    "current_query": f"{query_1} (Refinement: {query_2})",
    "retrieved_chunks": all_chunks,
    "session_history": [{"query": query_1, "answer": ans_1}]
}).json()
print(f"System:\n{synth_2.get('answer_markdown', '')}\n")

