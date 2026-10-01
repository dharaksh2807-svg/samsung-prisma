"""
RegulaStream — Benchmark Runner (pytest suite)
===============================================
Validates that the Benchmark Replay Harness correctly:
  1. Produces a well-formed telemetry report.
  2. Passes Gate G1 (Reproducibility) against the deterministic heuristic engine.
  3. Achieves Gate G6 (Trace Coverage) across all pipeline stages.
  4. Generates valid JSON output to disk.

These tests run WITHOUT a live server — they drive the harness logic directly
through FastAPI's TestClient to keep CI dependency-free (no `aiohttp` required,
no Docker, no network).

Architecture:
  - _run_offline_harness() calls each pipeline endpoint via TestClient and
    populates TraceEvent objects exactly as the async harness would.
  - The gate evaluators (_evaluate_g1, _evaluate_g6) are imported from the
    harness module and evaluated against those TraceEvent lists.
  - This avoids testing the HTTP stack (covered by other test files) and focuses
    exclusively on the gate logic, report schema, and reproducibility contract.
"""

from __future__ import annotations

import json
import os
import sys
import time
import tempfile
from typing import Any, Dict, List, Optional

import pytest

# ── Ensure backend package is importable
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))
# ── Ensure benchmark package is importable
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..", "benchmark")))

from fastapi.testclient import TestClient
from main import app

# Import harness components
import replay as harness

client = TestClient(app)


# ─────────────────────────────────────────────────────────────────────────────
# Offline Harness Driver
# ─────────────────────────────────────────────────────────────────────────────

def _post_via_testclient(path: str, payload: Dict[str, Any]) -> tuple:
    """Synchronous equivalent of harness._post using FastAPI TestClient."""
    t0 = time.perf_counter()
    try:
        resp = client.post(path, json=payload)
        latency_ms = (time.perf_counter() - t0) * 1000
        try:
            body = resp.json()
        except Exception:
            body = {"_raw": resp.text}
        return resp.status_code, latency_ms, body
    except Exception as exc:
        return -1, (time.perf_counter() - t0) * 1000, {"error": str(exc)}


def _run_offline_harness(num_runs: int = 2) -> List[harness.TraceEvent]:
    """
    Drives all SEED_SCENARIOS through the FastAPI TestClient and returns
    a list of TraceEvent objects identical in structure to what the async
    harness produces.
    """
    SEED = "OFFLINE_TEST_SEED"
    all_traces: List[harness.TraceEvent] = []

    for run_idx in range(num_runs):
        for scenario in harness.SEED_SCENARIOS:
            payload = dict(scenario["payload"])

            # ── Resolve runtime session ID for session scenarios
            if payload.get("session_id") == "__RUNTIME_SESSION_ID__":
                sid = f"{SEED}_{scenario['scenario_id']}_run{run_idx}"
                payload["session_id"] = sid

                # Pre-init the session
                init_payload = {"session_id": sid, **scenario["_init_payload"]}
                client.post("/api/session/init", json=init_payload)

            http_status, wall_ms, body = _post_via_testclient(
                scenario["endpoint"], payload
            )

            # Server latency
            server_ms: Optional[float] = None
            for key in ("latency_ms",):
                if key in body and isinstance(body[key], (int, float)):
                    server_ms = float(body[key])
                    break

            passed, failure_reason = harness._assert_scenario(scenario, http_status, body)

            # Build snapshot matching harness logic
            snapshot_keys = [
                "decision", "confidence", "is_valid", "update_required",
                "is_refinement", "search_query", "answer_version",
                "fabricated_ids_found", "latency_ms",
            ]
            snapshot = {k: body[k] for k in snapshot_keys if k in body}
            if "sub_queries" in body:
                snapshot["sub_query_count"] = len(body["sub_queries"])

            all_traces.append(harness.TraceEvent(
                scenario_id=scenario["scenario_id"],
                stage=scenario["stage"],
                endpoint=scenario["endpoint"],
                run_index=run_idx,
                http_status=http_status,
                latency_ms=wall_ms,
                server_latency_ms=server_ms,
                passed=passed,
                failure_reason=failure_reason,
                response_snapshot=snapshot,
            ))

    return all_traces


# ─────────────────────────────────────────────────────────────────────────────
# Fixtures
# ─────────────────────────────────────────────────────────────────────────────

@pytest.fixture(scope="module")
def trace_events() -> List[harness.TraceEvent]:
    """Run the offline harness twice (2 passes for G1 detection)."""
    return _run_offline_harness(num_runs=2)


@pytest.fixture(scope="module")
def benchmark_report(trace_events) -> harness.BenchmarkReport:
    return harness._build_report(
        base_url="http://localhost:8000 (TestClient)",
        concurrency=1,
        trace_events=trace_events,
        server_reachable=True,
    )


# ─────────────────────────────────────────────────────────────────────────────
# Schema & Structure Tests
# ─────────────────────────────────────────────────────────────────────────────

class TestReportSchema:
    """Validates that the BenchmarkReport has the expected structure."""

    def test_report_is_not_none(self, benchmark_report):
        assert benchmark_report is not None

    def test_total_runs_matches_expectation(self, benchmark_report):
        expected = len(harness.SEED_SCENARIOS) * 2  # 2 runs
        assert benchmark_report.total_runs == expected, (
            f"Expected {expected} total runs, got {benchmark_report.total_runs}"
        )

    def test_success_rate_is_percentage(self, benchmark_report):
        assert 0.0 <= benchmark_report.success_rate_pct <= 100.0

    def test_avg_latency_is_positive(self, benchmark_report):
        assert benchmark_report.avg_latency_ms >= 0.0

    def test_p95_latency_gte_avg(self, benchmark_report):
        assert benchmark_report.p95_latency_ms >= benchmark_report.avg_latency_ms - 0.1

    def test_gate_results_present(self, benchmark_report):
        assert len(benchmark_report.gate_results) == 2
        gate_names = {gr.gate for gr in benchmark_report.gate_results}
        assert "G1" in gate_names
        assert "G6" in gate_names

    def test_all_stages_have_latency_entry(self, benchmark_report):
        required = {"controller", "decomposer", "synthesizer", "validator", "monitor", "session"}
        missing = required - set(benchmark_report.stage_avg_latency_ms.keys())
        assert not missing, f"Missing stage latency entries: {missing}"

    def test_stage_avg_latency_values_positive(self, benchmark_report):
        for stage, avg in benchmark_report.stage_avg_latency_ms.items():
            assert avg >= 0.0, f"Stage {stage} has negative avg latency"

    def test_server_reachable_flag_true(self, benchmark_report):
        assert benchmark_report.server_reachable is True


# ─────────────────────────────────────────────────────────────────────────────
# Trace Event Tests
# ─────────────────────────────────────────────────────────────────────────────

class TestTraceEvents:
    """Validates individual trace events for completeness and correctness."""

    def test_every_scenario_has_two_traces(self, trace_events):
        by_sid: Dict[str, int] = {}
        for ev in trace_events:
            by_sid[ev.scenario_id] = by_sid.get(ev.scenario_id, 0) + 1
        for sid, count in by_sid.items():
            assert count == 2, f"Scenario {sid} has {count} traces, expected 2"

    def test_all_http_statuses_are_200(self, trace_events):
        failures = [ev for ev in trace_events if ev.http_status != 200]
        if failures:
            details = [
                f"{ev.scenario_id}: HTTP {ev.http_status} — {ev.failure_reason}"
                for ev in failures
            ]
            pytest.fail("Some scenarios returned non-200 status:\n" + "\n".join(details))

    def test_all_latencies_are_non_negative(self, trace_events):
        for ev in trace_events:
            assert ev.latency_ms >= 0.0, f"{ev.scenario_id} has negative latency"

    def test_no_trace_exceeds_5000ms(self, trace_events):
        slow = [ev for ev in trace_events if ev.latency_ms > 5000]
        assert not slow, f"Slow traces (>5s): {[ev.scenario_id for ev in slow]}"


# ─────────────────────────────────────────────────────────────────────────────
# Gate G1 — Reproducibility
# ─────────────────────────────────────────────────────────────────────────────

class TestGateG1:
    """
    Gate G1: Deterministic pipeline stages must produce identical field values
    across multiple replay passes given the same seed inputs.
    """

    def test_g1_gate_passes(self, benchmark_report):
        g1 = next(gr for gr in benchmark_report.gate_results if gr.gate == "G1")
        assert g1.passed, f"Gate G1 FAILED: {g1.detail}"

    def test_controller_retrieve_decision_is_stable(self, trace_events):
        runs = [ev for ev in trace_events if ev.scenario_id == "S0_CONTROLLER_RETRIEVE"]
        assert runs, "No traces found for S0_CONTROLLER_RETRIEVE"
        decisions = {ev.response_snapshot.get("decision") for ev in runs}
        assert len(decisions) == 1, f"Non-deterministic: saw decisions {decisions}"
        assert "RETRIEVE" in decisions

    def test_controller_wait_decision_is_stable(self, trace_events):
        runs = [ev for ev in trace_events if ev.scenario_id == "S1_CONTROLLER_WAIT"]
        assert runs
        decisions = {ev.response_snapshot.get("decision") for ev in runs}
        assert len(decisions) == 1
        assert "WAIT" in decisions

    def test_controller_no_retrieval_is_stable(self, trace_events):
        runs = [ev for ev in trace_events if ev.scenario_id == "S2_CONTROLLER_NO_RETRIEVAL"]
        assert runs
        decisions = {ev.response_snapshot.get("decision") for ev in runs}
        assert len(decisions) == 1
        assert "NO-RETRIEVAL" in decisions

    def test_validator_valid_flag_is_stable(self, trace_events):
        runs = [ev for ev in trace_events if ev.scenario_id == "S5_VALIDATOR_VALID"]
        assert runs
        flags = {ev.response_snapshot.get("is_valid") for ev in runs}
        assert len(flags) == 1
        assert True in flags

    def test_validator_hallucination_flag_is_stable(self, trace_events):
        runs = [ev for ev in trace_events if ev.scenario_id == "S6_VALIDATOR_HALLUCINATION"]
        assert runs
        flags = {ev.response_snapshot.get("is_valid") for ev in runs}
        assert len(flags) == 1
        assert False in flags

    def test_g1_evidence_is_non_empty(self, benchmark_report):
        g1 = next(gr for gr in benchmark_report.gate_results if gr.gate == "G1")
        assert len(g1.evidence) > 0, "G1 gate produced no evidence entries"


# ─────────────────────────────────────────────────────────────────────────────
# Gate G6 — Trace Coverage
# ─────────────────────────────────────────────────────────────────────────────

class TestGateG6:
    """
    Gate G6: Every pipeline stage must have at least one passing, timed trace event.
    """

    def test_g6_gate_passes(self, benchmark_report):
        g6 = next(gr for gr in benchmark_report.gate_results if gr.gate == "G6")
        assert g6.passed, f"Gate G6 FAILED: {g6.detail}"

    def test_controller_stage_has_passing_trace(self, trace_events):
        passing = [ev for ev in trace_events if ev.stage == "controller" and ev.passed]
        assert passing, "No passing trace for 'controller' stage"

    def test_decomposer_stage_has_passing_trace(self, trace_events):
        passing = [ev for ev in trace_events if ev.stage == "decomposer" and ev.passed]
        assert passing, "No passing trace for 'decomposer' stage"

    def test_synthesizer_stage_has_passing_trace(self, trace_events):
        passing = [ev for ev in trace_events if ev.stage == "synthesizer" and ev.passed]
        assert passing, "No passing trace for 'synthesizer' stage"

    def test_validator_stage_has_passing_trace(self, trace_events):
        passing = [ev for ev in trace_events if ev.stage == "validator" and ev.passed]
        assert passing, "No passing trace for 'validator' stage"

    def test_monitor_stage_has_passing_trace(self, trace_events):
        passing = [ev for ev in trace_events if ev.stage == "monitor" and ev.passed]
        assert passing, "No passing trace for 'monitor' stage"

    def test_session_stage_has_passing_trace(self, trace_events):
        passing = [ev for ev in trace_events if ev.stage == "session" and ev.passed]
        assert passing, "No passing trace for 'session' stage"

    def test_every_passing_trace_has_measured_latency(self, trace_events):
        bad = [ev for ev in trace_events if ev.passed and ev.latency_ms <= 0.0]
        assert not bad, (
            f"Passing traces with zero/negative latency: {[ev.scenario_id for ev in bad]}"
        )

    def test_g6_evidence_lists_all_stages(self, benchmark_report):
        g6 = next(gr for gr in benchmark_report.gate_results if gr.gate == "G6")
        required = {"controller", "decomposer", "synthesizer", "validator", "monitor", "session"}
        covered_in_evidence = set()
        for line in g6.evidence:
            for stage in required:
                if stage in line:
                    covered_in_evidence.add(stage)
        assert covered_in_evidence == required, (
            f"G6 evidence missing stages: {required - covered_in_evidence}"
        )


# ─────────────────────────────────────────────────────────────────────────────
# JSON Output Tests
# ─────────────────────────────────────────────────────────────────────────────

class TestJSONOutput:
    """Validates that the harness serialises a valid JSON report to disk."""

    def test_report_serialises_to_valid_json(self, benchmark_report):
        serialised = harness._serialize_report(benchmark_report)
        raw = json.dumps(serialised, default=str)
        parsed = json.loads(raw)
        assert isinstance(parsed, dict)

    def test_serialised_report_contains_required_top_level_keys(self, benchmark_report):
        required_keys = [
            "generated_at", "base_url", "total_runs", "successful_runs",
            "failed_runs", "success_rate_pct", "avg_latency_ms", "p95_latency_ms",
            "stage_avg_latency_ms", "gate_results", "trace_events",
            "g1_passed", "g6_passed", "all_gates_passed",
        ]
        serialised = harness._serialize_report(benchmark_report)
        for key in required_keys:
            assert key in serialised, f"Missing top-level key: {key!r}"

    def test_report_writes_to_disk_and_is_readable(self, benchmark_report):
        with tempfile.NamedTemporaryFile(
            suffix=".json", delete=False, mode="w", encoding="utf-8"
        ) as f:
            tmp_path = f.name
            json.dump(harness._serialize_report(benchmark_report), f, default=str)

        with open(tmp_path, "r", encoding="utf-8") as f:
            loaded = json.load(f)

        os.unlink(tmp_path)
        assert loaded["total_runs"] == benchmark_report.total_runs
        assert loaded["g1_passed"] == benchmark_report.g1_passed
        assert loaded["g6_passed"] == benchmark_report.g6_passed

    def test_trace_events_in_json_are_complete(self, benchmark_report):
        serialised = harness._serialize_report(benchmark_report)
        for ev in serialised["trace_events"]:
            assert "scenario_id" in ev
            assert "stage" in ev
            assert "latency_ms" in ev
            assert "passed" in ev

    def test_gate_results_in_json_have_correct_shape(self, benchmark_report):
        serialised = harness._serialize_report(benchmark_report)
        for gr in serialised["gate_results"]:
            assert "gate" in gr
            assert "passed" in gr
            assert "detail" in gr
            assert "evidence" in gr
            assert isinstance(gr["evidence"], list)


# ─────────────────────────────────────────────────────────────────────────────
# Scenario-Specific Correctness Tests
# ─────────────────────────────────────────────────────────────────────────────

class TestScenarioAssertions:
    """Fine-grained per-scenario validation independent of gate evaluation."""

    def test_s3_decomposer_returns_multiple_sub_queries(self, trace_events):
        runs = [
            ev for ev in trace_events
            if ev.scenario_id == "S3_DECOMPOSER_MULTI_INTENT" and ev.passed
        ]
        assert runs, "S3 decomposer scenario never passed"
        for ev in runs:
            count = ev.response_snapshot.get("sub_query_count", 0)
            assert count >= 2, f"Expected >=2 sub-queries, got {count}"

    def test_s7_monitor_requires_update_on_amendment(self, trace_events):
        runs = [
            ev for ev in trace_events
            if ev.scenario_id == "S7_MONITOR_UPDATE_REQUIRED" and ev.passed
        ]
        assert runs, "S7 monitor scenario never passed"
        for ev in runs:
            assert ev.response_snapshot.get("update_required") is True

    def test_s8_monitor_no_update_on_unrelated_doc(self, trace_events):
        runs = [
            ev for ev in trace_events
            if ev.scenario_id == "S8_MONITOR_NO_UPDATE" and ev.passed
        ]
        assert runs, "S8 monitor scenario never passed"
        for ev in runs:
            assert ev.response_snapshot.get("update_required") is False

    def test_s9_session_delta_query_is_refinement(self, trace_events):
        runs = [
            ev for ev in trace_events
            if ev.scenario_id == "S9_SESSION_DELTA_QUERY" and ev.passed
        ]
        assert runs, "S9 session delta query scenario never passed"
        for ev in runs:
            assert ev.response_snapshot.get("is_refinement") is True
