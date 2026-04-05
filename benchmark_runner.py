"""
ToolReliBench: Main Benchmark Runner
=====================================
Orchestrates trace generation, metric computation, statistical analysis,
and report generation.

Usage:
    python -m src.benchmark_runner                          # full run
    python -m src.benchmark_runner --runs-per-task 5       # quick run
    python -m src.benchmark_runner --models gpt-4o-mini claude-3-5-sonnet
"""

import argparse
import json
import logging
import sys
from pathlib import Path

import numpy as np
import pandas as pd

from src.trace_generator import generate_and_save_traces, MODEL_PROFILES, ARCHITECTURE_PROFILES
from src.evaluation_pipeline import run_full_analysis
from src.visualizations import save_all_plots

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s  %(levelname)-8s  %(message)s",
    datefmt="%H:%M:%S",
)
log = logging.getLogger("ToolReliBench")

# Default long-horizon task set (20 tasks)
DEFAULT_TASKS = [
    {"task_id": f"short_{i:03d}", "description": d, "expected_steps": e, "category": c, "difficulty": "short"}
    for i, (d, e, c) in enumerate([
        ("Calculate compound interest on $10,000 at 5% for 10 years.", 5, "calculation"),
        ("Verify Tokyo is the largest city. Search and confirm population.", 6, "research"),
        ("Look up S&P 500 value and compute what $5,000 would purchase.", 7, "finance"),
        ("Find capital of France and estimate distance Paris–Berlin.", 5, "geography"),
        ("Confirm Python is a high-level language. Search and verify.", 4, "verification"),
        ("Retrieve 2023 US GDP and calculate per-capita income.", 8, "economics"),
        ("Look up median global age and compute adult population share.", 7, "demographics"),
        ("Find NASDAQ / S&P 500 ratio and interpret the result.", 6, "finance"),
        ("Search for ML definition and verify against two sources.", 7, "verification"),
        ("Calculate real interest rate given 5.25% nominal and 3.4% inflation.", 5, "calculation"),
    ])
] + [
    {"task_id": f"long_{i:03d}", "description": d, "expected_steps": e, "category": c, "difficulty": "long"}
    for i, (d, e, c) in enumerate([
        ("Comprehensive financial analysis: S&P 500 / NASDAQ ratios, US GDP, inflation, unemployment. Cross-verify all indicators and summarise economic outlook.", 20, "financial_analysis"),
        ("Multi-step demographic research: world population, top-5 country shares, median age, urbanisation rate, 2030 projection. Verify consistency throughout.", 22, "demographic_research"),
        ("Chained data analysis: retrieve inflation, Fed rate, real interest rate, unemployment, GDP growth. Build economic health index and identify trends.", 20, "economic_analysis"),
        ("Recursive verification chain: machine learning definition → subtypes → deep learning → cross-reference all claims → resolve conflicts → final verified summary.", 24, "verification_chain"),
        ("Tool composition analysis: Python popularity, AI adoption %, programmer job market, language-job correlation. Verify with three independent sources.", 26, "tool_composition"),
        ("Multi-hop research on climate economics: CO2 metrics, green energy adoption rates, carbon pricing, GDP impact. Synthesise 8+ data points.", 25, "multi_hop_research"),
        ("Planning task: 10-city conference schedule with budget constraints, timezone conflicts, and venue availability. Verify each constraint satisfied.", 28, "planning"),
        ("Cross-reference synthesis: compare IMF, World Bank, and Fed data on US economic indicators. Identify discrepancies and provide reconciled estimate.", 22, "cross_reference"),
        ("Recursive verification of historical tech adoption (internet, mobile, AI) — project AI inflection point using S-curve analysis.", 24, "forecasting"),
        ("Error recovery task: start with intentionally noisy data, identify errors, correct, re-verify, produce clean final report.", 20, "error_recovery"),
    ])
]


def _save_report(results: dict, output_dir: Path) -> None:
    """Write markdown research report."""
    sums = results["model_summaries"]
    st   = results["statistical_tests"]
    ref  = results["reflection_analysis"]
    dele = results["delegation_threshold"]

    lines = ["# ToolReliBench: Experimental Results\n"]
    lines.append(f"**Total Trajectories:** {sum(v['n_trajectories'] for v in sums.values())}")
    lines.append(f"**Models Evaluated:** {len(sums)}")
    lines.append(f"**Models:** {', '.join(sums.keys())}\n")

    lines.append("## Model Summary\n")
    lines.append("| Model | Success % | CDI (↓) | SVR (↓) | RIC (↓) | Tool Hall. (↓) |")
    lines.append("|-------|-----------|---------|---------|---------|----------------|")
    for m, s in sums.items():
        lines.append(
            f"| {m} | {s['success_rate']*100:.1f}% "
            f"| {s['cdi']['mean']:.3f} ±{s['cdi']['std']:.3f} "
            f"| {s['svr']['mean']:.3f} ±{s['svr']['std']:.3f} "
            f"| {s['ric']['mean']:.3f} ±{s['ric']['std']:.3f} "
            f"| {s['tool_hallucination_rate']['mean']:.3f} |"
        )

    lines.append("\n## Statistical Comparisons\n")
    if not st.empty:
        lines.append(st.to_markdown(index=False))

    lines.append("\n## Self-Reflection Paradox\n")
    if not ref.empty:
        lines.append(ref.to_markdown(index=False))

    lines.append("\n## Delegation Threshold\n")
    if not dele.empty:
        lines.append(dele.to_markdown(index=False))

    lines.append("\n---\n*Generated by ToolReliBench*")

    report_path = output_dir / "research_report.md"
    report_path.write_text("\n".join(lines))
    log.info(f"Report saved → {report_path}")


def run(args) -> None:
    output_dir = Path(args.output_dir)
    traces_dir = output_dir / "traces"
    analysis_dir = output_dir / "analysis"
    analysis_dir.mkdir(parents=True, exist_ok=True)

    models       = args.models or list(MODEL_PROFILES.keys())
    architectures = args.architectures or list(ARCHITECTURE_PROFILES.keys())

    log.info("=" * 60)
    log.info("ToolReliBench — Benchmark Run")
    log.info(f"  Models:        {models}")
    log.info(f"  Architectures: {architectures}")
    log.info(f"  Tasks:         {len(DEFAULT_TASKS)} ({args.runs_per_task} runs each)")
    log.info("=" * 60)

    log.info("Generating traces…")
    all_traces = generate_and_save_traces(
        tasks=DEFAULT_TASKS,
        output_dir=str(traces_dir),
        models=models,
        architectures=architectures,
        runs_per_task=args.runs_per_task,
        seed=42,
    )
    total = sum(len(v) for v in all_traces.values())
    log.info(f"Generated {total} trajectories")

    log.info("Running analysis…")
    results = run_full_analysis(all_traces)

    # Save raw metrics JSON
    metrics_path = analysis_dir / "metrics.json"
    metrics_path.write_text(json.dumps(results["model_summaries"], indent=2, default=str))

    # Save CSV tables
    if not results["statistical_tests"].empty:
        results["statistical_tests"].to_csv(analysis_dir / "statistical_tests.csv", index=False)
    if not results["reflection_analysis"].empty:
        results["reflection_analysis"].to_csv(analysis_dir / "reflection_analysis.csv", index=False)
    if not results["delegation_threshold"].empty:
        results["delegation_threshold"].to_csv(analysis_dir / "delegation_threshold.csv", index=False)

    log.info("Generating plots…")
    save_all_plots(results, str(analysis_dir))

    log.info("Writing report…")
    _save_report(results, analysis_dir)

    # Print summary
    print("\n" + "=" * 60)
    print("Results Summary")
    print("=" * 60)
    for m, s in results["model_summaries"].items():
        print(f"\n{m}:")
        print(f"  Success Rate : {s['success_rate']*100:.1f}%")
        print(f"  CDI          : {s['cdi']['mean']:.3f} ± {s['cdi']['std']:.3f}")
        print(f"  SVR          : {s['svr']['mean']:.3f} ± {s['svr']['std']:.3f}")
        print(f"  RIC          : {s['ric']['mean']:.3f} ± {s['ric']['std']:.3f}")
        print(f"  Tool Hall.   : {s['tool_hallucination_rate']['mean']:.3f}")
    print(f"\nAll outputs → {output_dir}/")
    print("=" * 60)


def main():
    parser = argparse.ArgumentParser(description="ToolReliBench benchmark runner")
    parser.add_argument("--models", nargs="+", default=None,
                        choices=list(MODEL_PROFILES.keys()))
    parser.add_argument("--architectures", nargs="+", default=None,
                        choices=list(ARCHITECTURE_PROFILES.keys()))
    parser.add_argument("--runs-per-task", type=int, default=20)
    parser.add_argument("--output-dir", default="./results")
    args = parser.parse_args()
    run(args)


if __name__ == "__main__":
    main()
