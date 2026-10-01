import os
import asyncio
from google import genai

class RotationalLLMClient:
    def __init__(self):
        self.clients = []
        
        # Load multiple keys: GEMINI_API_KEY, GEMINI_API_KEY_1, GEMINI_API_KEY_2, etc.
        keys = []
        base_key = os.getenv("GEMINI_API_KEY")
        if base_key: keys.append(base_key)
        
        for i in range(1, 10):
            k = os.getenv(f"GEMINI_API_KEY_{i}")
            if k and k not in keys:
                keys.append(k)
                
        for k in keys:
            try:
                self.clients.append(genai.Client(api_key=k))
            except Exception as e:
                print(f"[RotationalLLMClient] Failed to init client for key {k[:5]}...: {e}")
                
        self.current_index = 0
        if self.clients:
            print(f"[RotationalLLMClient] Initialized with {len(self.clients)} API keys.")
        else:
            print(f"[RotationalLLMClient] WARNING: No Gemini API keys found.")

    async def generate_content(self, model: str, contents: str):
        # 1. Try Groq if configured!
        groq_key = os.getenv("GROQ_API_KEY")
        if groq_key:
            import httpx
            async with httpx.AsyncClient() as client:
                res = await client.post(
                    "https://api.groq.com/openai/v1/chat/completions",
                    headers={"Authorization": f"Bearer {groq_key}"},
                    json={
                        "model": "openai/gpt-oss-120b", 
                        "messages": [{"role": "user", "content": contents}]
                    },
                    timeout=30.0
                )
                if res.status_code == 200:
                    class GroqResponse:
                        def __init__(self, t): self.text = t
                    return GroqResponse(res.json()["choices"][0]["message"]["content"])
                else:
                    print(f"[RotationalLLMClient] Groq failed with {res.status_code}: {res.text}, falling back to Gemini.")

        # 2. Try Gemini
        if not self.clients:
            raise Exception("No valid LLM API keys configured.")
            
        attempts = 0
        max_attempts = len(self.clients)
        last_error = None
        
        while attempts < max_attempts:
            client = self.clients[self.current_index]
            try:
                response = await asyncio.to_thread(
                    client.models.generate_content,
                    model=model,
                    contents=contents,
                )
                return response
            except Exception as e:
                error_str = str(e)
                last_error = e
                if "429" in error_str or "RESOURCE_EXHAUSTED" in error_str or "400" in error_str:
                    print(f"[RotationalLLMClient] Key {self.current_index + 1}/{len(self.clients)} exhausted or invalid. Rotating...")
                    self.current_index = (self.current_index + 1) % len(self.clients)
                    attempts += 1
                else:
                    raise e
                    
        raise Exception(f"All configured API keys have exhausted their quota. Last error: {last_error}")

# Singleton instance
global_llm_client = None

def get_llm_client():
    global global_llm_client
    if global_llm_client is None:
        global_llm_client = RotationalLLMClient()
    return global_llm_client
