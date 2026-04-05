"""
ToolReliBench: Core Metric Implementations
==========================================
CDI, SVR, RIC, DAF — all grounded in actual computation.

CDI uses TF-IDF vectors + symmetric KL divergence (no external model needed).
RIC uses cosine similarity for cycle detection, NOT exact string match.
SVR uses extended regex pattern matching.
DAF tracks delegation depth trees.
"""

import re
import math
import numpy as np
from typing import List, Dict, Any, Optional, Tuple
from collections import defaultdict
from scipy.stats import t as t_dist
from scipy.special import kl_div
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity


# ─────────────────────────────────────────────────────────────────────────────
# Embedding helpers (TF-IDF, no internet required)
# ─────────────────────────────────────────────────────────────────────────────

def _build_tfidf_embeddings(texts: List[str]) -> np.ndarray:
    """
    Fit a combined word-level + char-level TF-IDF representation.
    Using both levels captures:
      - word: vocabulary/topic overlap
      - char (3-5 grams): morphological/terminology overlap

    Returns L2-normalised dense matrix (n_texts × n_features).
    """
    word_vec = TfidfVectorizer(
        max_features=256, sublinear_tf=True,
        ngram_range=(1, 2), min_df=1, analyzer="word",
        token_pattern=r"(?u)\b\w+\b",
    )
    char_vec = TfidfVectorizer(
        max_features=256, sublinear_tf=True,
        ngram_range=(3, 5), min_df=1, analyzer="char_wb",
    )

    Xw = word_vec.fit_transform(texts).toarray().astype(np.float64)
    Xc = char_vec.fit_transform(texts).toarray().astype(np.float64)

    def _l2(M):
        norms = np.linalg.norm(M, axis=1, keepdims=True)
        norms[norms == 0] = 1.0
        return M / norms

    X = np.hstack([_l2(Xw), _l2(Xc)])
    norms = np.linalg.norm(X, axis=1, keepdims=True)
    norms[norms == 0] = 1.0
    return X / norms


def _softmax(x: np.ndarray, eps: float = 1e-10) -> np.ndarray:
    """Numerically stable softmax."""
    x = x - x.max()
    e = np.exp(x)
    return (e + eps) / (e.sum() + eps * len(x))


def _sym_kl(p: np.ndarray, q: np.ndarray, eps: float = 1e-10) -> float:
    """Symmetric (Jensen–Shannon-like) KL divergence between two distributions."""
    p = _softmax(p)
    q = _softmax(q)
    kl_pq = np.sum(np.where(p > eps, p * np.log((p + eps) / (q + eps)), 0.0))
    kl_qp = np.sum(np.where(q > eps, q * np.log((q + eps) / (p + eps)), 0.0))
    return float(0.5 * (kl_pq + kl_qp))


# ─────────────────────────────────────────────────────────────────────────────
# 1. Context Drift Index (CDI)
# ─────────────────────────────────────────────────────────────────────────────

def compute_cdi(trajectory: Dict[str, Any]) -> Dict[str, Any]:
    """
    Compute CDI = (1/N) Σ (1 - cosine_sim(task_emb, step_emb_i))

    Uses combined word + char TF-IDF embeddings fitted over the full trajectory
    corpus so that vocabulary is shared.  Returns per-step scores.

    Interpretation:
      CDI ≈ 0.0–0.3  → minimal drift, strong alignment
      CDI ≈ 0.3–0.6  → moderate drift
      CDI ≈ 0.6–0.9  → significant drift
      CDI ≈ 0.9–1.0  → severe drift (task semantics largely forgotten)
    """
    from sklearn.metrics.pairwise import cosine_similarity as _cos_sim

    task_desc = trajectory.get("task_description", "")
    steps = trajectory.get("steps", [])

    if not steps:
        return {"cdi": 0.0, "step_scores": [], "n_steps": 0}

    step_texts = [s.get("reasoning", "") or "" for s in steps]
    all_texts = [task_desc] + step_texts

    embeddings = _build_tfidf_embeddings(all_texts)
    task_emb = embeddings[0:1]      # shape (1, d)
    step_embs = embeddings[1:]      # shape (N, d)

    sims = _cos_sim(task_emb, step_embs)[0]
    step_scores = [float(1.0 - s) for s in sims]

    cdi = float(np.mean(step_scores)) if step_scores else 0.0

    return {
        "cdi": cdi,
        "step_scores": step_scores,
        "n_steps": len(steps),
        "max_drift_step": int(np.argmax(step_scores)) + 1 if step_scores else None,
    }


# ─────────────────────────────────────────────────────────────────────────────
# 2. Silent Verification Rate (SVR)
# ─────────────────────────────────────────────────────────────────────────────

VERIFICATION_CLAIM_PATTERNS = [
    r"\bverif(ied|y|ying)\b",
    r"\bconfirm(ed|ing)?\b",
    r"\bchecked?\b",
    r"\bi\s+searched?\b",
    r"\baccording\s+to\b",
    r"\bbased\s+on\s+(my\s+)?research\b",
    r"\blookup\b",
    r"\bvalidat(ed|ing)?\b",
    r"\bcross[- ]referenc(ed|ing)?\b",
    r"\bsource(d|s)?\b",
]

VERIFICATION_TOOL_NAMES = {
    "search", "verify", "check", "validate",
    "confirm", "lookup", "query", "web_search",
    "data_lookup", "retrieve",
}

_CLAIM_RE = re.compile("|".join(VERIFICATION_CLAIM_PATTERNS), re.IGNORECASE)


def compute_svr(trajectory: Dict[str, Any]) -> Dict[str, Any]:
    """
    SVR = |V_claimed \\ V_actual| / |V_claimed|

    V_claimed: steps where reasoning contains a verification-claim pattern.
    V_actual:  steps where a verification tool was actually invoked.
    """
    steps = trajectory.get("steps", [])
    claimed, silent_steps = 0, 0
    flagged = []

    for step in steps:
        reasoning = step.get("reasoning", "") or ""
        tool_calls = step.get("tool_calls", []) or []

        has_claim = bool(_CLAIM_RE.search(reasoning))
        has_actual = any(
            tc.get("name", "").lower() in VERIFICATION_TOOL_NAMES
            for tc in tool_calls
        )

        if has_claim:
            claimed += 1
            if not has_actual:
                silent_steps += 1
                flagged.append(step.get("step_number", -1))

    svr = silent_steps / claimed if claimed > 0 else 0.0
    return {
        "svr": svr,
        "claimed_verifications": claimed,
        "silent_verifications": silent_steps,
        "flagged_steps": flagged,
    }


# ─────────────────────────────────────────────────────────────────────────────
# 3. Recursive Instability Coefficient (RIC)
# ─────────────────────────────────────────────────────────────────────────────

def compute_ric(
    trajectory: Dict[str, Any],
    sim_threshold: float = 0.90,
    min_cycle_gap: int = 3,
) -> Dict[str, Any]:
    """
    RIC = |C| / N

    Cycle detection via cosine similarity on TF-IDF embeddings.
    A cycle is detected when two steps (separated by ≥ min_cycle_gap) have
    cosine similarity ≥ sim_threshold — indicating the agent revisited the
    same state without making progress.

    This is the correct implementation; the original used exact string match,
    which never fires on real (or realistic simulated) LLM outputs.
    """
    steps = trajectory.get("steps", [])
    n = len(steps)

    if n < min_cycle_gap + 1:
        return {"ric": 0.0, "cycles_detected": 0, "n_steps": n, "cycle_pairs": []}

    texts = [s.get("reasoning", "") or "" for s in steps]
    embs = _build_tfidf_embeddings(texts)

    sim_matrix = cosine_similarity(embs)  # n × n
    cycles = []

    for i in range(n):
        for j in range(i + min_cycle_gap, n):
            if sim_matrix[i, j] >= sim_threshold:
                cycles.append((i + 1, j + 1))  # 1-indexed step numbers

    ric = len(cycles) / n if n > 0 else 0.0

    return {
        "ric": min(ric, 1.0),
        "cycles_detected": len(cycles),
        "n_steps": n,
        "cycle_pairs": cycles[:20],   # store first 20 for inspection
        "sim_threshold": sim_threshold,
    }


# ─────────────────────────────────────────────────────────────────────────────
# 4. Delegation Amplification Factor (DAF)
# ─────────────────────────────────────────────────────────────────────────────

DELEGATION_TOOL_NAMES = {
    "delegate", "spawn_agent", "create_agent", "sub_agent",
    "assign_task", "fork", "dispatch",
}

DELEGATION_PATTERNS = [
    r"\bdelegate\b",
    r"\bsub[- ]?agent\b",
    r"\bspawn\b",
    r"\bassign\s+to\b",
]
_DELEG_RE = re.compile("|".join(DELEGATION_PATTERNS), re.IGNORECASE)


def compute_daf(trajectory: Dict[str, Any]) -> Dict[str, Any]:
    """
    DAF = |T_generated| / |T_original|

    Tracks tool-based and text-based delegation signals.
    For single-agent trajectories DAF = 1.0 (baseline).
    """
    steps = trajectory.get("steps", [])
    delegations = 0
    delegation_steps = []

    for step in steps:
        tool_calls = step.get("tool_calls", []) or []
        reasoning = step.get("reasoning", "") or ""

        via_tool = any(
            tc.get("name", "").lower() in DELEGATION_TOOL_NAMES
            for tc in tool_calls
        )
        via_text = bool(_DELEG_RE.search(reasoning))

        if via_tool or via_text:
            delegations += 1
            delegation_steps.append(step.get("step_number", -1))

    daf = (delegations + 1) / 1  # +1 for original task
    return {
        "daf": float(daf),
        "delegations": delegations,
        "delegation_steps": delegation_steps,
    }


# ─────────────────────────────────────────────────────────────────────────────
# 5. Composite Reliability Score
# ─────────────────────────────────────────────────────────────────────────────

def compute_composite_reliability(
    cdi: float,
    svr: float,
    ric: float,
    w_cdi: float = 0.30,
    w_svr: float = 0.30,
    w_ric: float = 0.40,
) -> float:
    """
    R = w1·CDI' + w2·SVR + w3·RIC'
    CDI' = min(CDI/2.0, 1.0)   (scale: CDI > 2 = fully unreliable)
    RIC' = min(RIC * 5, 1.0)
    Lower is better.
    """
    cdi_norm = min(cdi / 2.0, 1.0)
    ric_norm = min(ric * 5.0, 1.0)
    return w_cdi * cdi_norm + w_svr * svr + w_ric * ric_norm


# ─────────────────────────────────────────────────────────────────────────────
# 6. Tool Hallucination Rate
# ─────────────────────────────────────────────────────────────────────────────

def compute_tool_hallucination_rate(
    trajectory: Dict[str, Any],
    valid_tools: Optional[set] = None,
) -> Dict[str, Any]:
    """
    Fraction of tool calls that reference non-existent / invalid tools.
    """
    if valid_tools is None:
        valid_tools = {"calculator", "search", "data_lookup",
                       "web_search", "retrieve", "lookup"}

    steps = trajectory.get("steps", [])
    total, hallucinated = 0, 0
    bad_calls = []

    for step in steps:
        for tc in (step.get("tool_calls") or []):
            name = tc.get("name", "").lower()
            total += 1
            if name not in valid_tools:
                hallucinated += 1
                bad_calls.append({"step": step.get("step_number"), "tool": name})

    rate = hallucinated / total if total > 0 else 0.0
    return {
        "hallucination_rate": rate,
        "total_tool_calls": total,
        "hallucinated_calls": hallucinated,
        "bad_calls": bad_calls,
    }


# ─────────────────────────────────────────────────────────────────────────────
# 7. MetricAggregator — compute all metrics for a list of trajectories
# ─────────────────────────────────────────────────────────────────────────────

def _ci95(values: List[float]) -> Tuple[float, float]:
    """95 % confidence interval using t-distribution."""
    n = len(values)
    if n < 2:
        m = float(np.mean(values)) if values else 0.0
        return (m, m)
    m = np.mean(values)
    s = np.std(values, ddof=1)
    margin = t_dist.ppf(0.975, df=n - 1) * s / math.sqrt(n)
    return (float(m - margin), float(m + margin))


class MetricAggregator:
    """Compute and aggregate all metrics over a collection of trajectories."""

    def compute_all(self, trajectories: List[Dict[str, Any]]) -> Dict[str, Any]:
        cdi_vals, svr_vals, ric_vals, daf_vals, thr_vals = [], [], [], [], []
        success_vals = []

        for traj in trajectories:
            cdi_r = compute_cdi(traj)
            svr_r = compute_svr(traj)
            ric_r = compute_ric(traj)
            daf_r = compute_daf(traj)
            thr_r = compute_tool_hallucination_rate(traj)

            cdi_vals.append(cdi_r["cdi"])
            svr_vals.append(svr_r["svr"])
            ric_vals.append(ric_r["ric"])
            daf_vals.append(daf_r["daf"])
            thr_vals.append(thr_r["hallucination_rate"])
            success_vals.append(1 if traj.get("success", False) else 0)

        def _stats(vals):
            lo, hi = _ci95(vals)
            return {
                "mean": float(np.mean(vals)),
                "std": float(np.std(vals, ddof=1)) if len(vals) > 1 else 0.0,
                "ci_95": [lo, hi],
                "n": len(vals),
            }

        return {
            "cdi": _stats(cdi_vals),
            "svr": _stats(svr_vals),
            "ric": _stats(ric_vals),
            "daf": _stats(daf_vals),
            "tool_hallucination_rate": _stats(thr_vals),
            "success_rate": float(np.mean(success_vals)),
            "n_trajectories": len(trajectories),
        }
