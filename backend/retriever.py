import os
import re
import math
from typing import List, Dict, Any
from pydantic import BaseModel
from google import genai
import asyncio

class RetrieveRequest(BaseModel):
    queries: List[str]

class RetrieverOutput(BaseModel):
    chunks: List[Dict[str, Any]]

def cosine_similarity(v1, v2):
    dot_product = sum(a * b for a, b in zip(v1, v2))
    mag1 = math.sqrt(sum(a * a for a in v1))
    mag2 = math.sqrt(sum(b * b for b in v2))
    if mag1 == 0 or mag2 == 0:
        return 0.0
    return dot_product / (mag1 * mag2)

class BM25:
    def __init__(self, corpus: List[List[str]], k1=1.5, b=0.75):
        self.k1 = k1
        self.b = b
        self.corpus_size = len(corpus)
        self.avgdl = sum(len(doc) for doc in corpus) / self.corpus_size if self.corpus_size else 0
        self.doc_freqs = []
        self.idf = {}
        self.doc_len = []
        
        df = {}
        for doc in corpus:
            self.doc_len.append(len(doc))
            frequencies = {}
            for word in doc:
                frequencies[word] = frequencies.get(word, 0) + 1
            self.doc_freqs.append(frequencies)
            for word in set(doc):
                df[word] = df.get(word, 0) + 1
                
        for word, freq in df.items():
            self.idf[word] = math.log(1 + (self.corpus_size - freq + 0.5) / (freq + 0.5))
            
    def get_scores(self, query: List[str]) -> List[float]:
        scores = [0.0] * self.corpus_size
        for idx in range(self.corpus_size):
            score = 0.0
            doc_len = self.doc_len[idx]
            frequencies = self.doc_freqs[idx]
            for q_term in query:
                if q_term not in frequencies:
                    continue
                freq = frequencies[q_term]
                numerator = self.idf.get(q_term, 0) * freq * (self.k1 + 1)
                denominator = freq + self.k1 * (1 - self.b + self.b * doc_len / self.avgdl)
                score += numerator / denominator
            scores[idx] = score
        return scores

class SimpleRetriever:
    def __init__(self):
        current_dir = os.path.dirname(os.path.abspath(__file__))
        path1 = os.path.join(current_dir, "data", "regulations")
        path2 = os.path.join(os.path.dirname(current_dir), "data", "regulations")
        self.data_dir = path1 if os.path.exists(path1) else path2
        self.chunks = []
        self.bm25 = None
        self.embeddings = []
        self.gemini_client = None
        if os.environ.get("GEMINI_API_KEY"):
            self.gemini_client = genai.Client(api_key=os.environ.get("GEMINI_API_KEY"))
            
        self._load_documents()
        self._initialize_indices()
        print(f"[SimpleRetriever] Loaded and indexed {len(self.chunks)} chunks from {self.data_dir}")

    def _load_documents(self):
        if not os.path.exists(self.data_dir):
            return
        for filename in os.listdir(self.data_dir):
            if filename.endswith(".md"):
                filepath = os.path.join(self.data_dir, filename)
                with open(filepath, "r", encoding="utf-8") as f:
                    content = f.read()
                    doc_id = filename.replace(".md", "").upper()
                    
                    # Split into sections/chunks
                    sections = re.split(r'\n(?=§|\#)', content)
                    for i, section in enumerate(sections):
                        sec_text = section.strip()
                        if not sec_text: continue
                        
                        exact_doc_id = doc_id
                        match = re.match(r'^(§|\#)\s*([A-Za-z0-9_\-\.]+)', sec_text)
                        if match:
                            exact_doc_id = f"{doc_id} §{match.group(2)}"
                            
                        self.chunks.append({
                            "doc_id": exact_doc_id,
                            "chunk_id": f"{doc_id}_CHUNK_{i}",
                            "text": sec_text,
                            "filename": filename
                        })
                        
    def _tokenize(self, text: str) -> List[str]:
        return re.findall(r'\w+', text.lower())

    def _initialize_indices(self):
        if not self.chunks:
            return
            
        # 1. Sparse Index (BM25)
        tokenized_corpus = [self._tokenize(chunk["text"]) for chunk in self.chunks]
        self.bm25 = BM25(tokenized_corpus)
        
        # 2. Dense Index (Gemini Embeddings)
        if self.gemini_client:
            try:
                print(f"[SimpleRetriever] Generating dense embeddings for {len(self.chunks)} chunks...")
                texts = [c["text"] for c in self.chunks]
                # API limit allows batch embedding
                response = self.gemini_client.models.embed_content(
                    model="text-embedding-004", 
                    contents=texts
                )
                self.embeddings = response.embeddings if hasattr(response, 'embeddings') else []
                if isinstance(self.embeddings, list) and len(self.embeddings) > 0 and hasattr(self.embeddings[0], 'values'):
                    self.embeddings = [e.values for e in self.embeddings]
            except Exception as e:
                print(f"[SimpleRetriever] Failed to generate embeddings: {e}")
                self.embeddings = []

    async def get_dense_query_embedding(self, query: str) -> List[float]:
        if not self.gemini_client:
            return []
        try:
            # Run in executor to avoid blocking async loop if client is sync
            loop = asyncio.get_event_loop()
            def embed():
                return self.gemini_client.models.embed_content(
                    model="text-embedding-004", 
                    contents=query
                )
            response = await loop.run_in_executor(None, embed)
            if hasattr(response, 'embeddings') and len(response.embeddings) > 0:
                if hasattr(response.embeddings[0], 'values'):
                    return response.embeddings[0].values
        except Exception as e:
            print(f"[SimpleRetriever] Dense query failed: {e}")
        return []

    async def search(self, queries: List[str]) -> RetrieverOutput:
        if not self.chunks:
            return RetrieverOutput(chunks=[])
            
        final_scored_chunks = {} # chunk_id -> (score, chunk)
        
        for q in queries:
            # 1. Sparse Scoring
            tokenized_q = self._tokenize(q)
            sparse_scores = self.bm25.get_scores(tokenized_q)
            
            # 2. Dense Scoring
            dense_scores = [0.0] * len(self.chunks)
            if self.embeddings:
                q_emb = await self.get_dense_query_embedding(q)
                if q_emb:
                    for i, doc_emb in enumerate(self.embeddings):
                        dense_scores[i] = cosine_similarity(q_emb, doc_emb)
            
            # 3. Hybrid Fusion (Reciprocal Rank Fusion)
            # Rank sparse
            sparse_ranks = {i: rank for rank, (i, score) in enumerate(sorted(enumerate(sparse_scores), key=lambda x: x[1], reverse=True))}
            dense_ranks = {i: rank for rank, (i, score) in enumerate(sorted(enumerate(dense_scores), key=lambda x: x[1], reverse=True))}
            
            for i, chunk in enumerate(self.chunks):
                # Only consider if there's some relevance
                if sparse_scores[i] == 0.0 and dense_scores[i] < 0.2:
                    continue
                    
                rrf_score = (1.0 / (60 + sparse_ranks[i])) + (1.0 / (60 + dense_ranks[i]))
                chunk_id = chunk["chunk_id"]
                
                # Deduplicate by keeping the max score across all sub-queries
                if chunk_id not in final_scored_chunks or rrf_score > final_scored_chunks[chunk_id][0]:
                    final_scored_chunks[chunk_id] = (rrf_score, chunk)

        # Sort and take top K
        sorted_results = sorted(final_scored_chunks.values(), key=lambda x: x[0], reverse=True)
        top_k = [item[1] for item in sorted_results[:5]] # Return top 5 chunks overall
        
        # Strip chunk_id for standard output
        clean_results = []
        for c in top_k:
            clean_results.append({
                "doc_id": c["doc_id"],
                "text": c["text"]
            })
            
        return RetrieverOutput(chunks=clean_results)

    def get_document(self, doc_id: str) -> str:
        # Reconstruct doc from chunks
        texts = [c["text"] for c in self.chunks if c["doc_id"] == doc_id]
        if not texts:
            return "Document not found."
        return "\n\n".join(texts)
