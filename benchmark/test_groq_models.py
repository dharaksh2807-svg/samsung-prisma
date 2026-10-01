import os
import requests

api_key = os.environ.get("GROQ_API_KEY")
if not api_key:
    from dotenv import load_dotenv
    load_dotenv(dotenv_path="backend/.env")
    api_key = os.environ.get("GROQ_API_KEY")

headers = {
    "Authorization": f"Bearer {api_key}",
    "Content-Type": "application/json"
}

resp = requests.get("https://api.groq.com/openai/v1/models", headers=headers)
if resp.status_code == 200:
    models = resp.json().get("data", [])
    for m in models:
        print(f"Model: {m['id']}")
else:
    print(f"Error {resp.status_code}: {resp.text}")

