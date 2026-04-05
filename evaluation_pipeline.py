"""
ToolReliBench: Statistical Evaluation Pipeline
===============================================
Pairwise t-tests, Cohen's d, degradation curves, and model comparisons.
"""

import math
import numpy as np
import pandas as pd
from typing import Dict, List, Any, Tuple, Optional
from scipy import stats
from src.metrics import MetricAggregator, compute_cdi, compute_svr, compute_ric


def _cohens_d(a: List[float], b: List[float]) -> float:
    """Pooled Cohen's d effect size."""
    n1, n2 = len(a), len(b)
    if n1 < 2 or n2 < 2:
        return 0.0
    s1, s2 = np.std(a, ddof=1), np.std(b, ddof=1)
    sp = math.sqrt(((n1 - 1) * s1**2 + (n2 - 1) * s2**2) / (n1 + n2 - 2))
    if sp == 0:
        return 0.0
    return (np.mean(a) - np.mean(b)) / sp


def pairwise_comparisons(
    model_metrics: Dict[str, Dict[str, List[float]]],
    metric_keys: List[str] = ("cdi", "svr", "ric"),
) -> pd.DataFrame:
    """
    Compute pairwise Welch t-tests and Cohen's d for each metric pair of models.
    Returns a DataFrame with columns:
      model_a, model_b, metric, mean_diff, t_stat, p_value, cohens_d, significant
    """
    models = list(model_metrics.keys())
    rows = []

    for i, m_a in enumerate(models):
        for m_b in models[i + 1:]:
            for metric in metric_keys:
                a_vals = model_metrics[m_a].get(metric, [])
                b_vals = model_metrics[m_b].get(metric, [])
                if not a_vals or not b_vals:
                    continue

                t_stat, p_val = stats.ttest_ind(a_vals, b_vals, equal_var=False)
                d = _cohens_d(a_vals, b_vals)
                rows.append({
                    "model_a": m_a,
                    "model_b": m_b,
                    "metric": metric.upper(),
                    "mean_a": round(float(np.mean(a_vals)), 4),
                    "mean_b": round(float(np.mean(b_vals)), 4),
                    "mean_diff": round(float(np.mean(a_vals) - np.mean(b_vals)), 4),
                    "t_statistic": round(float(t_stat), 3),
                    "p_value": round(float(p_val), 4),
                    "cohens_d": round(float(d), 3),
                    "significant": bool(p_val < 0.05),
                })

    return pd.DataFrame(rows)


def self_reflection_analysis(
    trajectories_by_model: Dict[str, List[Dict]],
) -> pd.DataFrame:
    """
    Compute the self-reflection paradox:
    - short tasks (< 12 steps): reflection should help accuracy
    - long tasks (>= 12 steps): reflection should increase SVR

    Returns summary table.
    """
    rows = []
    for model, traces in trajectories_by_model.items():
        short = [t for t in traces if t.get("metadata", {}).get("n_steps", 0) < 12]
        long  = [t for t in traces if t.get("metadata", {}).get("n_steps", 0) >= 12]

        for label, subset in [("short (≤11 steps)", short), ("long (≥12 steps)", long)]:
            if not subset:
                continue
            svr_vals = [compute_svr(t)["svr"] for t in subset]
            success_vals = [1 if t.get("success") else 0 for t in subset]
            rows.append({
                "model": model,
                "task_length": label,
                "n_traces": len(subset),
                "mean_success_rate": round(float(np.mean(success_vals)), 3),
                "mean_svr": round(float(np.mean(svr_vals)), 3),
            })
    return pd.DataFrame(rows)


def delegation_threshold_analysis(
    trajectories: List[Dict],
) -> pd.DataFrame:
    """
    Compute failure rate at different delegation depths (approximated via DAF).
    """
    from src.metrics import compute_daf

    depth_bins = [(1.0, 1.5, "depth=1"), (1.5, 3.0, "depth=2"), (3.0, 99, "depth=3+")]
    rows = []
    for lo, hi, label in depth_bins:
        subset = []
        for t in trajectories:
            daf_val = compute_daf(t)["daf"]
            if lo <= daf_val < hi:
                subset.append(t)
        if subset:
            fail_rate = 1.0 - float(np.mean([1 if t.get("success") else 0 for t in subset]))
            n = len(subset)
            se = math.sqrt(fail_rate * (1 - fail_rate) / n) if n > 1 else 0.0
            ci_lo = max(0.0, fail_rate - 1.96 * se)
            ci_hi = min(1.0, fail_rate + 1.96 * se)
            rows.append({
                "delegation_depth": label,
                "n_trajectories": n,
                "failure_rate": round(fail_rate, 3),
                "ci_95_low": round(ci_lo, 3),
                "ci_95_high": round(ci_hi, 3),
            })
    return pd.DataFrame(rows)


def stepwise_degradation(
    trajectories: List[Dict],
    max_step: int = 30,
) -> Dict[str, List[float]]:
    """
    For each step index, compute mean CDI, SVR (cumulative), and tool hallucination.
    Returns dict of lists for plotting.
    """
    step_cdi: Dict[int, List[float]] = {s: [] for s in range(1, max_step + 1)}
    step_hall: Dict[int, List[float]] = {s: [] for s in range(1, max_step + 1)}

    valid_tools = {"calculator", "search", "data_lookup",
                   "web_search", "retrieve", "lookup"}

    for traj in trajectories:
        cdi_r = compute_cdi(traj)
        step_scores = cdi_r.get("step_scores", [])
        for idx, score in enumerate(step_scores):
            step = idx + 1
            if step <= max_step:
                step_cdi[step].append(score)

        for step_obj in traj.get("steps", []):
            sn = step_obj.get("step_number", 0)
            if sn > max_step:
                continue
            tcs = step_obj.get("tool_calls", []) or []
            if tcs:
                hall = sum(1 for tc in tcs if tc.get("name", "").lower() not in valid_tools)
                step_hall[sn].append(hall / len(tcs))
            else:
                step_hall[sn].append(0.0)

    steps_list = list(range(1, max_step + 1))
    mean_cdi = [float(np.mean(step_cdi[s])) if step_cdi[s] else 0.0 for s in steps_list]
    mean_hall = [float(np.mean(step_hall[s])) if step_hall[s] else 0.0 for s in steps_list]

    return {
        "steps": steps_list,
        "mean_cdi_per_step": mean_cdi,
        "mean_hallucination_per_step": mean_hall,
    }


def run_full_analysis(
    trajectories_by_model: Dict[str, List[Dict]],
) -> Dict[str, Any]:
    """
    Master analysis runner. Returns all statistics needed for the paper.
    """
    agg = MetricAggregator()
    model_summaries = {}
    model_raw_metrics: Dict[str, Dict[str, List[float]]] = {}

    for model, traces in trajectories_by_model.items():
        summary = agg.compute_all(traces)
        model_summaries[model] = summary

        # Extract per-trace scalar values for t-tests
        cdi_vals = [compute_cdi(t)["cdi"] for t in traces]
        svr_vals = [compute_svr(t)["svr"] for t in traces]
        ric_vals = [compute_ric(t)["ric"] for t in traces]
        model_raw_metrics[model] = {"cdi": cdi_vals, "svr": svr_vals, "ric": ric_vals}

    statistical_tests = pairwise_comparisons(model_raw_metrics)
    reflection_table = self_reflection_analysis(trajectories_by_model)

    all_traces = [t for traces in trajectories_by_model.values() for t in traces]
    delegation_table = delegation_threshold_analysis(all_traces)
    degradation_data = stepwise_degradation(all_traces)

    return {
        "model_summaries": model_summaries,
        "statistical_tests": statistical_tests,
        "reflection_analysis": reflection_table,
        "delegation_threshold": delegation_table,
        "stepwise_degradation": degradation_data,
    }
