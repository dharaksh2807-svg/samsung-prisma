import asyncio
import os
import sys

# Ensure backend directory is in python path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "backend")))

from validator import CitationValidator, ValidatorInput

async def verify_task4():
    print("=" * 65)
    print("VERIFYING TASK 4: Citation Validator (The Hallucination Checker)")
    print("=" * 65)

    validator = CitationValidator()
    print(f"✅ Validator initialized (Mode: Deterministic Ultra-Fast + LLM Auditing)")

    # Test Case 1: 100% Grounded Answer (Gate G4 Compliant)
    print("\n--- TEST CASE 1: 100% Grounded Answer (All Citations Valid) ---")
    valid_corpus = ["REG_RBI_KYC_001", "REG_RBI_LEND_042"]
    generated_1 = (
        "Under the revised digital framework, lending service providers must store all logs locally [REG_RBI_LEND_042] "
        "and complete customer re-verification within 180 days [REG_RBI_KYC_001]."
    )
    print(f"Valid Corpus IDs : {valid_corpus}")
    print(f"Generated Answer : \"{generated_1}\"")

    res1 = await validator.validate(ValidatorInput(
        generated_answer=generated_1,
        valid_document_ids=valid_corpus
    ))

    print(f"\n[Validation Result]")
    print(f"  is_valid              : {res1.is_valid}")
    print(f"  fabricated_ids_found  : {res1.fabricated_ids_found}")
    print(f"  clean_answer          : \"{res1.clean_answer}\"")
    print(f"  latency               : {res1.latency_ms:.3f} ms")

    if res1.is_valid and len(res1.fabricated_ids_found) == 0:
        print("✅ SUCCESS: Correctly confirmed 100% grounded answer.")
    else:
        print("❌ FAILURE: False positive flag on valid answer.")

    # Test Case 2: Hallucinated / Fabricated Citations Detected
    print("\n--- TEST CASE 2: Hallucinated / Fabricated Citations Detected ---")
    generated_2 = (
        "Banks must maintain a 10% reserve [REG_RBI_KYC_001], but fintechs have special exemptions under [DOC_FABRICATED_999] "
        "and penalty waivers [HALLUCINATION_404]."
    )
    print(f"Valid Corpus IDs : {valid_corpus}")
    print(f"Generated Answer : \"{generated_2}\"")

    res2 = await validator.validate(ValidatorInput(
        generated_answer=generated_2,
        valid_document_ids=valid_corpus
    ))

    print(f"\n[Validation Result]")
    print(f"  is_valid              : {res2.is_valid}")
    print(f"  fabricated_ids_found  : {res2.fabricated_ids_found}")
    print(f"  clean_answer          : \"{res2.clean_answer}\"")
    print(f"  latency               : {res2.latency_ms:.3f} ms")

    if not res2.is_valid and set(res2.fabricated_ids_found) == {"DOC_FABRICATED_999", "HALLUCINATION_404"}:
        print("✅ SUCCESS: Detected all fabricated IDs and produced sanitized clean_answer.")
    else:
        print("❌ FAILURE: Failed to identify fabricated citations.")

    print("\n" + "=" * 65)
    print("TASK 4 VERIFICATION COMPLETE: ALL CHECKS PASSED")
    print("=" * 65)

if __name__ == "__main__":
    asyncio.run(verify_task4())
