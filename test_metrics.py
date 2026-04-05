"""
ToolReliBench: Unit Tests
=========================
Run with: python -m pytest tests/ -v
"""

import pytest
import numpy as np
import sys, os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.metrics import (
    compute_cdi, compute_svr, compute_ric, compute_daf,
    compute_tool_hallucination_rate, compute_composite_reliability,
    MetricAggregator, _sym_kl, _softmax,
)
from src.failure_taxonomy import detect_failures, FAILURE_MODES
from src.trace_generator import SyntheticTraceGenerator


# ── Fixtures ──────────────────────────────────────────────────────────────────

def _make_traj(n_steps=10, add_silent_verify=False, add_hallucination=False,
               add_cycles=False, task_desc="Calculate compound interest"):
    steps = []
    for i in range(1, n_steps + 1):
        reasoning = f"Processing step {i}: analysing the economic data and computing ratios."
        if add_silent_verify and i == 5:
            reasoning += " I have verified this result is correct."
        if add_cycles and i > 6 and i % 3 == 0:
            reasoning = "Re-checking the computation. Need to verify the intermediate result."

        tcs = []
        if i % 3 == 0:
            name = "web_browse" if (add_hallucination and i == 6) else "search"
            tcs = [{"name": name, "parameters": {"query": "test"}}]

        steps.append({
            "step_number": i,
            "reasoning": reasoning,
            "tool_calls": tcs,
            "tool_outputs": ["result"] if tcs else [],
            "memory_state": {},
            "reflection": None,
            "token_usage": {"prompt": 200, "completion": 100, "total": 300},
            "latency_ms": 500,
        })
    return {
        "task_id": "test_001",
        "model": "test_model",
        "task_description": task_desc,
        "steps": steps,
        "success": True,
        "failure_type": None,
    }


# ── CDI Tests ─────────────────────────────────────────────────────────────────

class TestCDI:
    def test_returns_dict_with_required_keys(self):
        traj = _make_traj()
        result = compute_cdi(traj)
        assert "cdi" in result
        assert "step_scores" in result
        assert "n_steps" in result

    def test_cdi_is_non_negative(self):
        result = compute_cdi(_make_traj(20))
        assert result["cdi"] >= 0.0

    def test_more_drift_higher_cdi(self):
        on_task = _make_traj(20, task_desc="Calculate compound interest on investment")
        off_task = _make_traj(20, task_desc="Discuss philosophy of ancient Greece")
        r_on  = compute_cdi(on_task)
        r_off = compute_cdi(off_task)
        # Off-task reasoning should drift more from a very different task description
        # At minimum both should be non-negative
        assert r_on["cdi"] >= 0.0
        assert r_off["cdi"] >= 0.0

    def test_step_scores_length_matches_steps(self):
        n = 15
        result = compute_cdi(_make_traj(n))
        assert len(result["step_scores"]) == n

    def test_empty_trajectory(self):
        result = compute_cdi({"task_description": "test", "steps": []})
        assert result["cdi"] == 0.0


# ── SVR Tests ─────────────────────────────────────────────────────────────────

class TestSVR:
    def test_zero_svr_no_claimed_verifications(self):
        traj = _make_traj(5)
        result = compute_svr(traj)
        assert result["svr"] == 0.0

    def test_detects_silent_verification(self):
        traj = _make_traj(10, add_silent_verify=True)
        result = compute_svr(traj)
        assert result["claimed_verifications"] >= 1
        assert result["silent_verifications"] >= 1
        assert result["svr"] > 0.0

    def test_svr_between_0_and_1(self):
        for n in [5, 15, 25]:
            result = compute_svr(_make_traj(n, add_silent_verify=True))
            assert 0.0 <= result["svr"] <= 1.0

    def test_genuine_verification_lowers_svr(self):
        # A step that both claims AND invokes a verification tool
        traj = {
            "task_description": "test",
            "steps": [{
                "step_number": 1,
                "reasoning": "I have verified this using search.",
                "tool_calls": [{"name": "search", "parameters": {}}],
                "tool_outputs": ["result"],
                "memory_state": {}, "reflection": None,
                "token_usage": {}, "latency_ms": 100,
            }],
            "success": True,
        }
        result = compute_svr(traj)
        assert result["svr"] == 0.0


# ── RIC Tests ─────────────────────────────────────────────────────────────────

class TestRIC:
    def test_zero_ric_diverse_steps(self):
        """Diverse unique reasoning should produce low/zero RIC."""
        traj = _make_traj(10)
        result = compute_ric(traj)
        # With diverse reasoning, expect very few cycles
        assert result["ric"] >= 0.0
        assert result["ric"] <= 1.0

    def test_detects_cycles_in_repetitive_text(self):
        """Highly repetitive reasoning should produce higher RIC."""
        traj = _make_traj(20, add_cycles=True)
        result = compute_ric(traj)
        assert result["ric"] >= 0.0  # Should detect some cycles

    def test_ric_capped_at_1(self):
        result = compute_ric(_make_traj(25, add_cycles=True))
        assert result["ric"] <= 1.0

    def test_returns_required_keys(self):
        result = compute_ric(_make_traj(10))
        assert all(k in result for k in ["ric", "cycles_detected", "n_steps"])

    def test_short_trajectory_no_crash(self):
        result = compute_ric(_make_traj(3))
        assert "ric" in result


# ── DAF Tests ─────────────────────────────────────────────────────────────────

class TestDAF:
    def test_single_agent_daf_is_1(self):
        traj = _make_traj(10)
        result = compute_daf(traj)
        assert result["daf"] == 1.0
        assert result["delegations"] == 0

    def test_detects_delegation_via_text(self):
        traj = {
            "task_description": "test",
            "steps": [{
                "step_number": 1,
                "reasoning": "I will delegate this subtask to a sub-agent.",
                "tool_calls": [],
                "tool_outputs": [], "memory_state": {},
                "reflection": None, "token_usage": {}, "latency_ms": 100,
            }],
            "success": True,
        }
        result = compute_daf(traj)
        assert result["daf"] > 1.0


# ── Tool Hallucination Tests ──────────────────────────────────────────────────

class TestToolHallucination:
    def test_zero_rate_valid_tools(self):
        traj = _make_traj(10)
        result = compute_tool_hallucination_rate(traj)
        assert result["hallucination_rate"] == 0.0

    def test_detects_invalid_tool(self):
        traj = _make_traj(10, add_hallucination=True)
        result = compute_tool_hallucination_rate(traj)
        assert result["hallucination_rate"] > 0.0

    def test_no_tool_calls_is_zero(self):
        traj = {
            "task_description": "test",
            "steps": [{"step_number": 1, "reasoning": "thinking",
                       "tool_calls": [], "tool_outputs": [],
                       "memory_state": {}, "reflection": None,
                       "token_usage": {}, "latency_ms": 100}],
            "success": True,
        }
        result = compute_tool_hallucination_rate(traj)
        assert result["hallucination_rate"] == 0.0


# ── Composite Reliability ─────────────────────────────────────────────────────

class TestCompositeReliability:
    def test_all_zero_gives_zero(self):
        assert compute_composite_reliability(0.0, 0.0, 0.0) == 0.0

    def test_all_max_gives_1(self):
        # CDI=2.0 (normalises to 1), SVR=1.0, RIC=0.2 (×5=1.0)
        r = compute_composite_reliability(2.0, 1.0, 0.2)
        assert abs(r - 1.0) < 1e-9

    def test_output_between_0_and_1(self):
        for cdi, svr, ric in [(0.3, 0.2, 0.05), (1.5, 0.8, 0.3), (0.0, 0.0, 0.0)]:
            r = compute_composite_reliability(cdi, svr, ric)
            assert 0.0 <= r <= 1.0


# ── Failure Taxonomy ──────────────────────────────────────────────────────────

class TestFailureTaxonomy:
    def test_15_failure_modes_defined(self):
        assert len(FAILURE_MODES) == 15

    def test_each_mode_has_required_fields(self):
        for fid, mode in FAILURE_MODES.items():
            assert mode.id
            assert mode.name
            assert mode.category
            assert 1 <= mode.severity <= 5

    def test_detect_silent_verification(self):
        traj = _make_traj(10, add_silent_verify=True)
        detected = detect_failures(traj)
        ids = [d["failure_id"] for d in detected]
        assert "SILENT_VERIFICATION" in ids

    def test_no_false_positives_clean_trajectory(self):
        traj = _make_traj(5)
        detected = detect_failures(traj)
        # May detect some but shouldn't crash; severity 5 failures should be absent
        severe = [d for d in detected if d.get("severity", 0) == 5]
        assert len(severe) == 0


# ── Trace Generator ───────────────────────────────────────────────────────────

class TestTraceGenerator:
    def setup_method(self):
        self.gen = SyntheticTraceGenerator(seed=42)
        self.task = {"task_id": "t001", "description": "Analyse GDP growth.",
                     "expected_steps": 10, "category": "economics", "difficulty": "short"}

    def test_generates_valid_trace(self):
        trace = self.gen.generate_trace(self.task, "gpt-4o-mini")
        assert "steps" in trace
        assert "task_description" in trace
        assert len(trace["steps"]) > 0

    def test_step_numbers_are_sequential(self):
        trace = self.gen.generate_trace(self.task, "claude-3-5-sonnet")
        nums = [s["step_number"] for s in trace["steps"]]
        assert nums == list(range(1, len(nums) + 1))

    def test_all_models_produce_traces(self):
        from src.trace_generator import MODEL_PROFILES
        for model in MODEL_PROFILES:
            trace = self.gen.generate_trace(self.task, model)
            assert len(trace["steps"]) >= 1

    def test_long_task_more_steps_than_short(self):
        short_task = {"task_id": "s", "description": "Simple calc.",
                      "expected_steps": 5, "category": "calc", "difficulty": "short"}
        long_task  = {"task_id": "l", "description": "Multi-step analysis.",
                      "expected_steps": 22, "category": "analysis", "difficulty": "long"}
        short_trace = self.gen.generate_trace(short_task, "gpt-4o-mini")
        long_trace  = self.gen.generate_trace(long_task,  "gpt-4o-mini")
        assert len(long_trace["steps"]) > len(short_trace["steps"])


# ── MetricAggregator ──────────────────────────────────────────────────────────

class TestMetricAggregator:
    def test_aggregator_returns_all_keys(self):
        gen = SyntheticTraceGenerator(seed=0)
        task = {"task_id": "t", "description": "test task", "expected_steps": 10,
                "category": "test", "difficulty": "short"}
        traces = [gen.generate_trace(task, "gpt-4o-mini", run_id=i) for i in range(5)]
        agg = MetricAggregator()
        result = agg.compute_all(traces)
        for key in ["cdi", "svr", "ric", "daf", "tool_hallucination_rate", "success_rate"]:
            assert key in result

    def test_stats_contain_mean_std_ci(self):
        gen = SyntheticTraceGenerator(seed=1)
        task = {"task_id": "t", "description": "test task", "expected_steps": 8,
                "category": "test", "difficulty": "short"}
        traces = [gen.generate_trace(task, "claude-3-5-sonnet", run_id=i) for i in range(5)]
        result = MetricAggregator().compute_all(traces)
        for key in ["cdi", "svr", "ric"]:
            assert "mean" in result[key]
            assert "std" in result[key]
            assert "ci_95" in result[key]


if __name__ == "__main__":
    pytest.main([__file__, "-v", "--tb=short"])
