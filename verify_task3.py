import asyncio
import os
import sys

# Ensure backend directory is in path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "backend")))

from synthesizer import StatefulSynthesizer, SynthesizerInput

async def verify_task3():
    print("=" * 60)
    print("VERIFYING TASK 3: Stateful RAG Synthesizer")
    print("=" * 60)

    # Note: Ensure GEMINI_API_KEY is available if we want real LLM responses
    synth = StatefulSynthesizer()
    
    if synth._gemini_client:
        print(f"✅ Gemini client loaded using model: {synth.model_name}")
    else:
        print("⚠️ Gemini client NOT loaded. Proceeding with deterministic fallback.")

    print("\n--- TEST CASE 1: Standard Evidence with Citation (Gate G4) ---")
    input1 = SynthesizerInput(
        current_query="What are the video KYC requirements for existing customers?",
        session_history=["User asked about digital onboarding"],
        retrieved_chunks=[
            {"doc_id": "REG_014_SEC_4", "text": "Existing customers need reverification by March 2027."},
            {"doc_id": "REG_015_SEC_2", "text": "Video KYC can be used for reverification."}
        ]
    )
    print(f"Query: {input1.current_query}")
    print(f"Evidence provided: {len(input1.retrieved_chunks)} chunks")
    
    result1 = await synth.synthesize(input1)
    
    print("\n[Synthesizer Output]")
    print(result1.answer_markdown)
    print(f"\nLatency: {result1.latency_ms:.2f} ms")
    
    # Verification
    if "REG_014_SEC_4" in result1.answer_markdown and "REG_015_SEC_2" in result1.answer_markdown:
        print("✅ SUCCESS: Found both citations in output.")
    else:
        print("❌ FAILURE: Missing citations in output.")

    print("\n--- TEST CASE 2: Conflicting Evidence ---")
    input2 = SynthesizerInput(
        current_query="Is physical verification mandatory? (Check for conflicts)",
        session_history=[],
        retrieved_chunks=[
            {"doc_id": "REG_2010_01", "text": "Physical verification is mandatory for all accounts."},
            {"doc_id": "REG_2025_03", "text": "Physical verification is no longer mandatory if Video KYC is used."}
        ]
    )
    print(f"Query: {input2.current_query}")
    print(f"Evidence provided: {len(input2.retrieved_chunks)} chunks")
    
    result2 = await synth.synthesize(input2)
    
    print("\n[Synthesizer Output]")
    print(result2.answer_markdown)
    print(f"\nLatency: {result2.latency_ms:.2f} ms")

    print("\n--- TEST CASE 3: Insufficient Evidence ---")
    input3 = SynthesizerInput(
        current_query="What are the capital requirements for micro-lenders?",
        session_history=[],
        retrieved_chunks=[]
    )
    print(f"Query: {input3.current_query}")
    print(f"Evidence provided: {len(input3.retrieved_chunks)} chunks")
    
    result3 = await synth.synthesize(input3)
    
    print("\n[Synthesizer Output]")
    print(result3.answer_markdown)
    print(f"\nLatency: {result3.latency_ms:.2f} ms")
    
    if "Insufficient evidence" in result3.answer_markdown:
        print("✅ SUCCESS: Correctly identified insufficient evidence.")
    else:
        print("❌ FAILURE: Did not trigger insufficient evidence rule.")

    print("\nVerification Complete.")
    print("=" * 60)

if __name__ == "__main__":
    asyncio.run(verify_task3())
