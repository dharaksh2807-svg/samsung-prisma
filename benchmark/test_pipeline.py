import requests
import time

BASE_URL = "http://localhost:8000/api"
QUERY = "What are the KYC requirements for digital lending platforms?"

print(f"--- TESTING REGULASTREAM PIPELINE LOCAL ---")
print(f"Query: {QUERY}\n")

# 1. Decompose
t0 = time.time()
res = requests.post(f"{BASE_URL}/decomposer/decompose", json={"query": QUERY}).json()
print(f"1. Decomposer ({(time.time()-t0)*1000:.0f}ms): {res.get('sub_queries', res)}")

# 2. Retrieve
t0 = time.time()
search_queries = [sq["search_query"] for sq in res.get("sub_queries", [])]
evidence = []
ret_res = requests.post(f"{BASE_URL}/retriever/search", json={"queries": search_queries}).json()
evidence.extend(ret_res.get("chunks", ret_res))
print(f"2. Retriever ({(time.time()-t0)*1000:.0f}ms): Found {len(evidence)} chunks.")

# 3. Synthesize
t0 = time.time()
synth_res = requests.post(f"{BASE_URL}/synthesizer/synthesize", json={
    "current_query": QUERY,
    "retrieved_chunks": evidence,
    "session_history": []
}).json()
print(f"3. Synthesizer ({(time.time()-t0)*1000:.0f}ms):\n{synth_res.get('answer_markdown', synth_res)}\n")

# 4. Validate
t0 = time.time()
val_res = requests.post(f"{BASE_URL}/validator/validate", json={
    "generated_answer": synth_res.get("answer_markdown", ""),
    "valid_document_ids": [e.get("doc_id", "REGULATION_001") for e in evidence if type(e) is dict]
}).json()
print(f"4. Validator ({(time.time()-t0)*1000:.0f}ms): Valid? {val_res.get('is_valid')}")
if val_res.get("fabricated_ids_found"):
    print(f"Fabricated IDs: {val_res['fabricated_ids_found']}")

