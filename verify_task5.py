import asyncio
import os
import sys

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "backend")))

from monitor import RegulatoryStreamMonitor, StreamMonitorInput

async def verify_task5():
    print("=" * 68)
    print("VERIFYING TASK 5: Regulatory Stream Monitor (Knowledge Update Trigger)")
    print("=" * 68)

    mon = RegulatoryStreamMonitor()
    print(f"✅ RegulatoryStreamMonitor initialized")

    prev_query  = "What are the KYC reverification requirements for existing customers?"
    prev_answer = (
        "Existing customers must complete KYC reverification by March 2027 [REG_014_SEC_4]. "
        "Video KYC is an acceptable method for digital lending customers [REG_015_SEC_2]."
    )

    # ── Scenario 1: Amendment INVALIDATES the previous answer ──────────────
    print("\n─── SCENARIO 1: Direct Amendment Invalidates Previous Answer ───")
    print(f"Previous Query  : {prev_query}")
    print(f"Previous Answer : {prev_answer}")

    new_doc_invalidating = (
        "Effective immediately, the KYC reverification deadline has been extended to December 2028. "
        "This supersedes all prior deadlines. Existing digital lending customers must now complete "
        "in-person reverification; video KYC is no longer acceptable."
    )
    print(f"\nNew Document    : [{new_doc_invalidating[:80]}...]")

    res1 = await mon.evaluate(StreamMonitorInput(
        previous_query=prev_query,
        previous_answer=prev_answer,
        new_doc_id="REG_AMEND_2026_KYC",
        new_document_text=new_doc_invalidating
    ))

    print(f"\n[Stream Monitor Output]")
    print(f"  update_required   : {res1.update_required}")
    print(f"  affected_claims   : {res1.affected_claims}")
    print(f"  reasoning         : {res1.reasoning}")
    print(f"  latency           : {res1.latency_ms:.3f} ms")

    if res1.update_required and len(res1.affected_claims) > 0:
        print("✅ SUCCESS: Correctly flagged amendment as invalidating previous answer.")
        print("   ⚠  KNOWLEDGE UPDATE should be sent to frontend for answer re-synthesis.")
    else:
        print("❌ FAILURE: Did not detect amendment conflict.")

    # ── Scenario 2: Unrelated Document — NO update needed ──────────────────
    print("\n─── SCENARIO 2: Unrelated Document — No Update Required ───")

    new_doc_unrelated = (
        "New export credit guidelines have been issued for cross-border trade finance institutions. "
        "These regulations govern quarterly reporting to the Trade Finance Board."
    )
    print(f"New Document    : [{new_doc_unrelated[:80]}...]")

    res2 = await mon.evaluate(StreamMonitorInput(
        previous_query=prev_query,
        previous_answer=prev_answer,
        new_doc_id="REG_EXPORT_2026_07",
        new_document_text=new_doc_unrelated
    ))

    print(f"\n[Stream Monitor Output]")
    print(f"  update_required   : {res2.update_required}")
    print(f"  affected_claims   : {res2.affected_claims}")
    print(f"  reasoning         : {res2.reasoning}")
    print(f"  latency           : {res2.latency_ms:.3f} ms")

    if not res2.update_required:
        print("✅ SUCCESS: Correctly identified unrelated document. No knowledge update triggered.")
    else:
        print("❌ FAILURE: False positive — incorrectly flagged unrelated document.")

    print()
    print("=" * 68)
    print("TASK 5 VERIFICATION COMPLETE")
    print("=" * 68)

if __name__ == "__main__":
    asyncio.run(verify_task5())
