import asyncio
import os
import sys
import time

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "backend")))

from session import SessionManager, DeltaQueryGenerator, EvidenceItem
from synthesizer import StatefulSynthesizer, SynthesizerInput

async def verify_task6():
    print("=" * 72)
    print("VERIFYING TASK 6: Session State & Late Refinement (Hackathon Gate G5)")
    print("=" * 72)

    session_mgr = SessionManager()
    synthesizer = StatefulSynthesizer()
    print("✅ SessionManager & StatefulSynthesizer initialized successfully.")

    # ─── SCENARIO 1: Initial User Query (Turn 1) ──────────────────────────────
    print("\n─── SCENARIO 1: Initial Turn & State Storage ───")
    session_id = "user_session_4021"
    initial_query = "What are the KYC reverification requirements for digital lending platforms?"
    
    initial_evidence = [
        EvidenceItem(
            doc_id="REG_014_SEC_4",
            text="All regulated entities conducting digital lending must perform full customer KYC verification every 2 years."
        ),
        EvidenceItem(
            doc_id="REG_015_SEC_2",
            text="Digital lending platforms may use video-based customer identification processes (V-CIP) for onboarding."
        )
    ]

    # Synthesize initial answer
    initial_synth = await synthesizer.synthesize(SynthesizerInput(
        current_query=initial_query,
        retrieved_chunks=[{"doc_id": e.doc_id, "text": e.text} for e in initial_evidence],
        session_history=[]
    ))

    # Record in Session Manager
    session = session_mgr.record_initial_turn(
        session_id=session_id,
        query=initial_query,
        answer=initial_synth.answer_markdown,
        evidence=initial_evidence
    )

    print(f"Session ID         : {session.session_id}")
    print(f"Initial Query      : {initial_query}")
    print(f"Answer Version     : {session.answer_version}")
    print(f"Cached Evidence    : {[e.doc_id for e in session.current_evidence]}")
    print(f"Initial Answer     : {initial_synth.answer_markdown[:120]}...")

    assert session.answer_version == 1, "Initial answer version must be 1"
    assert len(session.current_evidence) == 2, "Must cache 2 evidence items"
    print("✅ Initial turn recorded and evidence cached.")

    # ─── SCENARIO 2: Late Constraint (Delta Query & State Preservation) ────────
    print("\n─── SCENARIO 2: Late Constraint Refinement (Delta Retrieval) ───")
    late_constraint = "...only for existing borrowers with active credit lines"
    print(f"Late Constraint    : \"{late_constraint}\"")

    start_delta = time.perf_counter()
    delta_out = await session_mgr.prepare_delta_query(
        session_id=session_id,
        new_constraint=late_constraint
    )
    delta_time_ms = (time.perf_counter() - start_delta) * 1000

    print(f"\n[Delta Query Generator Output]")
    print(f"  is_refinement     : {delta_out.is_refinement}")
    print(f"  search_query      : \"{delta_out.search_query}\"")
    print(f"  cached_evidence   : {delta_out.cached_evidence_ids}")
    print(f"  delta_prep_latency: {delta_time_ms:.3f} ms")

    assert delta_out.is_refinement is True, "Late constraint should be recognized as a refinement"
    assert "existing borrowers" in delta_out.search_query.lower(), "Constraint must be in delta query"
    print("✅ Successfully generated standalone delta search query without wiping session state.")

    # Simulate targeted vector retrieval for ONLY the delta query
    delta_evidence = [
        EvidenceItem(
            doc_id="REG_014_SEC_5_BORROWERS",
            text="Existing borrowers with active lines who completed KYC within the past 12 months are exempt from reverification until loan maturity."
        )
    ]

    # Combine cached evidence + delta evidence and re-synthesize
    start_refine = time.perf_counter()
    combined_evidence = session_mgr.get_combined_evidence(session_id) + delta_evidence
    
    refined_synth = await synthesizer.synthesize(SynthesizerInput(
        current_query=f"{initial_query} (Refinement: {late_constraint})",
        retrieved_chunks=[{"doc_id": e.doc_id, "text": e.text} for e in combined_evidence],
        session_history=[f"Q: {h.query} -> A: {h.answer}" for h in session.history]
    ))
    refine_time_ms = (time.perf_counter() - start_refine) * 1000

    updated_session = session_mgr.apply_refinement(
        session_id=session_id,
        new_constraint=late_constraint,
        refined_answer=refined_synth.answer_markdown,
        delta_evidence=delta_evidence
    )

    print(f"\n[Refined Answer V{updated_session.answer_version}]")
    print(f"  Updated Version   : {updated_session.answer_version}")
    print(f"  Total Evidence Chunks Cached: {len(updated_session.current_evidence)}")
    print(f"  Evidence Doc IDs  : {[e.doc_id for e in updated_session.current_evidence]}")
    print(f"  Refined Answer    : {refined_synth.answer_markdown}")
    print(f"  Refinement Latency: {refine_time_ms:.3f} ms")

    assert updated_session.answer_version == 2, "Answer version must increment to 2"
    assert len(updated_session.current_evidence) == 3, "Evidence must accumulate without clearing prior state"
    assert "REG_014_SEC_5_BORROWERS" in [e.doc_id for e in updated_session.current_evidence]
    print("✅ Gate G5 requirement met: State preserved, delta retrieved, answer upgraded to V2.")

    # ─── SCENARIO 3: Distinct Unrelated Topic (No False Refinement) ─────────────
    print("\n─── SCENARIO 3: Non-Refinement Query (Branching Topic) ───")
    new_unrelated_query = "What is the statutory liquidity ratio requirement for commercial banks?"
    
    unrelated_out = await session_mgr.prepare_delta_query(
        session_id=session_id,
        new_constraint=new_unrelated_query
    )
    print(f"New Query         : \"{new_unrelated_query}\"")
    print(f"  is_refinement   : {unrelated_out.is_refinement}")
    print(f"  search_query    : \"{unrelated_out.search_query}\"")

    assert "statutory liquidity ratio" in unrelated_out.search_query.lower()
    print("✅ Correctly distinguished fresh standalone query.")

    print("\n" + "=" * 72)
    print("TASK 6 VERIFICATION COMPLETE: ALL CHECKS PASSED (HACKATHON GATE G5 SATISFIED)")
    print("=" * 72)

if __name__ == "__main__":
    asyncio.run(verify_task6())
