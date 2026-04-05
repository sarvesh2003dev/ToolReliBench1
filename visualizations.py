"""
ToolReliBench: Visualisations
==============================
Publication-quality matplotlib figures.
"""

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import pandas as pd
from typing import Dict, List, Any
from pathlib import Path

plt.rcParams.update({
    "font.family": "DejaVu Sans",
    "font.size": 11,
    "axes.titlesize": 13,
    "axes.labelsize": 12,
    "xtick.labelsize": 10,
    "ytick.labelsize": 10,
    "legend.fontsize": 10,
    "figure.dpi": 150,
    "axes.grid": True,
    "grid.alpha": 0.3,
    "axes.spines.top": False,
    "axes.spines.right": False,
})

MODEL_COLORS = {
    "gpt-4o-mini":        "#4C72B0",
    "claude-3-5-sonnet":  "#DD8452",
    "gemini-1-5":         "#55A868",
    "llama-3-3-70b":      "#C44E52",
}

MODEL_LABELS = {
    "gpt-4o-mini":        "GPT-4o-mini",
    "claude-3-5-sonnet":  "Claude 3.5 Sonnet",
    "gemini-1-5":         "Gemini 1.5",
    "llama-3-3-70b":      "Llama 3.3 70B",
}


def plot_metrics_comparison(
    model_summaries: Dict[str, Dict],
    output_path: str = "results/analysis/metrics_comparison.png",
) -> None:
    """
    4-panel bar chart: CDI / SVR / RIC / Success Rate per model with CIs.
    """
    models = list(model_summaries.keys())
    metrics = [
        ("cdi",                  "Context Drift Index (CDI)",       "CDI"),
        ("svr",                  "Silent Verification Rate (SVR)",  "SVR"),
        ("ric",                  "Recursive Instability Coeff. (RIC)", "RIC"),
    ]

    fig, axes = plt.subplots(1, 4, figsize=(16, 5))
    fig.suptitle("ToolReliBench: Model Reliability Metrics\n(lower CDI/SVR/RIC = better; higher Success = better)",
                 fontsize=13, fontweight="bold", y=1.02)

    colors = [MODEL_COLORS.get(m, "#888") for m in models]
    labels = [MODEL_LABELS.get(m, m) for m in models]
    x = np.arange(len(models))
    width = 0.55

    for ax, (key, title, ylabel) in zip(axes[:3], metrics):
        vals  = [model_summaries[m][key]["mean"] for m in models]
        ci_lo = [model_summaries[m][key]["mean"] - model_summaries[m][key]["ci_95"][0] for m in models]
        ci_hi = [model_summaries[m][key]["ci_95"][1] - model_summaries[m][key]["mean"] for m in models]

        bars = ax.bar(x, vals, width, color=colors, alpha=0.85,
                      yerr=[ci_lo, ci_hi], capsize=4, error_kw={"elinewidth": 1.5})
        ax.set_title(title)
        ax.set_ylabel(ylabel)
        ax.set_xticks(x)
        ax.set_xticklabels(labels, rotation=20, ha="right")

        for bar, v in zip(bars, vals):
            ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 0.005,
                    f"{v:.3f}", ha="center", va="bottom", fontsize=9)

    # Success rate (4th panel)
    ax = axes[3]
    s_vals = [model_summaries[m]["success_rate"] * 100 for m in models]
    bars = ax.bar(x, s_vals, width, color=colors, alpha=0.85)
    ax.set_title("Task Success Rate")
    ax.set_ylabel("Success (%)")
    ax.set_ylim(0, 105)
    ax.set_xticks(x)
    ax.set_xticklabels(labels, rotation=20, ha="right")
    for bar, v in zip(bars, s_vals):
        ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 0.5,
                f"{v:.1f}%", ha="center", va="bottom", fontsize=9)

    plt.tight_layout()
    Path(output_path).parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(output_path, bbox_inches="tight")
    plt.close()


def plot_stepwise_degradation(
    degradation_data: Dict[str, List],
    output_path: str = "results/analysis/stepwise_degradation.png",
) -> None:
    """
    Two-panel line chart: CDI and tool hallucination rate vs step number.
    Annotates the critical drift inflection point.
    """
    steps = degradation_data["steps"]
    cdi   = degradation_data["mean_cdi_per_step"]
    hall  = degradation_data["mean_hallucination_per_step"]

    # Smooth slightly
    def _smooth(arr, w=3):
        return np.convolve(arr, np.ones(w) / w, mode="same")

    cdi_s  = _smooth(cdi)
    hall_s = _smooth(hall)

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(13, 5))
    fig.suptitle("Reliability Degradation Over Reasoning Steps",
                 fontsize=14, fontweight="bold")

    # CDI panel
    ax1.plot(steps, cdi_s, color="#4C72B0", linewidth=2.2, label="Mean CDI")
    ax1.fill_between(steps,
                     [max(0, c - 0.03) for c in cdi_s],
                     [c + 0.03 for c in cdi_s],
                     alpha=0.2, color="#4C72B0")
    ax1.axvline(x=12, color="red", linestyle="--", linewidth=1.5, label="Drift inflection (step 12)")
    ax1.axhline(y=0.5, color="orange", linestyle=":", linewidth=1.5, label="Moderate drift threshold")
    ax1.set_xlabel("Reasoning Step")
    ax1.set_ylabel("Context Drift Index (CDI)")
    ax1.set_title("CDI Accumulation")
    ax1.legend(loc="upper left")
    ax1.set_xlim(1, max(steps))

    # Tool hallucination panel
    ax2.plot(steps, hall_s, color="#C44E52", linewidth=2.2, label="Hallucination Rate")
    ax2.fill_between(steps,
                     [max(0, h - 0.01) for h in hall_s],
                     [h + 0.01 for h in hall_s],
                     alpha=0.2, color="#C44E52")
    ax2.axvline(x=15, color="red", linestyle="--", linewidth=1.5, label="Multi-agent threshold")
    ax2.set_xlabel("Reasoning Step")
    ax2.set_ylabel("Tool Hallucination Rate")
    ax2.set_title("Tool Hallucination Growth")
    ax2.legend(loc="upper left")
    ax2.set_xlim(1, max(steps))

    plt.tight_layout()
    Path(output_path).parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(output_path, bbox_inches="tight")
    plt.close()


def plot_delegation_threshold(
    delegation_df: pd.DataFrame,
    output_path: str = "results/analysis/delegation_threshold.png",
) -> None:
    """Bar chart showing failure rate at each delegation depth with CIs."""
    fig, ax = plt.subplots(figsize=(7, 5))

    labels = delegation_df["delegation_depth"].tolist()
    vals   = delegation_df["failure_rate"].tolist()
    ci_lo  = (delegation_df["failure_rate"] - delegation_df["ci_95_low"]).tolist()
    ci_hi  = (delegation_df["ci_95_high"] - delegation_df["failure_rate"]).tolist()

    colors = ["#55A868", "#4C72B0", "#C44E52"][:len(labels)]
    bars = ax.bar(labels, [v * 100 for v in vals], color=colors, alpha=0.85,
                  yerr=[[c * 100 for c in ci_lo], [c * 100 for c in ci_hi]],
                  capsize=5, error_kw={"elinewidth": 1.5})

    for bar, v in zip(bars, vals):
        ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 1.5,
                f"{v*100:.1f}%", ha="center", va="bottom", fontsize=10, fontweight="bold")

    ax.set_xlabel("Delegation Depth")
    ax.set_ylabel("Failure Rate (%)")
    ax.set_title("Phase Transition: Failure Rate vs. Delegation Depth\n"
                 "(Sharp instability threshold at depth ≥ 3)")
    ax.set_ylim(0, 100)
    plt.tight_layout()
    plt.savefig(output_path, bbox_inches="tight")
    plt.close()


def plot_reflection_paradox(
    reflection_df: pd.DataFrame,
    output_path: str = "results/analysis/reflection_paradox.png",
) -> None:
    """Grouped bar chart: SVR by task length and model."""
    models  = reflection_df["model"].unique()
    lengths = reflection_df["task_length"].unique()
    x = np.arange(len(lengths))
    width = 0.8 / max(len(models), 1)

    fig, axes = plt.subplots(1, 2, figsize=(13, 5))
    fig.suptitle("Self-Reflection Paradox: Short vs. Long Task Performance",
                 fontsize=13, fontweight="bold")

    for metric, ax, title, ylabel in [
        ("mean_svr",          axes[0], "Silent Verification Rate (SVR)\nby Task Length", "SVR"),
        ("mean_success_rate", axes[1], "Success Rate by Task Length",                     "Success Rate"),
    ]:
        for i, model in enumerate(models):
            sub = reflection_df[reflection_df["model"] == model].set_index("task_length")
            vals = [float(sub.loc[l, metric]) if l in sub.index else 0.0 for l in lengths]
            offset = (i - len(models) / 2) * width + width / 2
            ax.bar(x + offset, vals, width * 0.9,
                   color=MODEL_COLORS.get(model, "#888"),
                   label=MODEL_LABELS.get(model, model), alpha=0.85)

        ax.set_xticks(x)
        ax.set_xticklabels([l.replace(" ", "\n") for l in lengths])
        ax.set_title(title)
        ax.set_ylabel(ylabel)
        ax.legend(fontsize=9)

    plt.tight_layout()
    plt.savefig(output_path, bbox_inches="tight")
    plt.close()


def save_all_plots(results: Dict[str, Any], output_dir: str = "results/analysis") -> None:
    """Save every figure from the full analysis results dict."""
    plot_metrics_comparison(
        results["model_summaries"],
        f"{output_dir}/metrics_comparison.png",
    )
    plot_stepwise_degradation(
        results["stepwise_degradation"],
        f"{output_dir}/stepwise_degradation.png",
    )
    if not results["delegation_threshold"].empty:
        plot_delegation_threshold(
            results["delegation_threshold"],
            f"{output_dir}/delegation_threshold.png",
        )
    if not results["reflection_analysis"].empty:
        plot_reflection_paradox(
            results["reflection_analysis"],
            f"{output_dir}/reflection_paradox.png",
        )
