import os
from typing import List, Dict, Any
from pydantic import BaseModel

class RetrieveRequest(BaseModel):
    queries: List[str]

class RetrieverOutput(BaseModel):
    chunks: List[Dict[str, Any]]

class SimpleRetriever:
    def __init__(self):
        current_dir = os.path.dirname(os.path.abspath(__file__))
        path1 = os.path.join(current_dir, "data", "regulations") # e.g. /app/data/regulations
        path2 = os.path.join(os.path.dirname(current_dir), "data", "regulations") # e.g. local backend/../data
        self.data_dir = path1 if os.path.exists(path1) else path2
        self.documents = []
        self._load_documents()
        print(f"[SimpleRetriever] Loaded {len(self.documents)} documents from {self.data_dir}")

    def _load_documents(self):
        if not os.path.exists(self.data_dir):
            return
        for filename in os.listdir(self.data_dir):
            if filename.endswith(".md"):
                filepath = os.path.join(self.data_dir, filename)
                with open(filepath, "r", encoding="utf-8") as f:
                    content = f.read()
                    # A very simple chunking strategy for the hackathon: use the whole file as a chunk
                    doc_id = filename.replace(".md", "").upper()
                    self.documents.append({
                        "doc_id": doc_id,
                        "text": content,
                        "filename": filename
                    })

    async def search(self, queries: List[str]) -> RetrieverOutput:
        results = []
        for doc in self.documents:
            results.append({
                "doc_id": doc["doc_id"],
                "text": doc["text"]
            })
        return RetrieverOutput(chunks=results)

    def get_document(self, doc_id: str) -> str:
        for doc in self.documents:
            if doc["doc_id"] == doc_id:
                return doc["text"]
        return "Document not found."
