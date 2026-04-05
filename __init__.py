"""ToolReliBench: Long-Horizon Reliability Evaluation for Tool-Using Agents."""
__version__ = "1.0.0"

from src.metrics import (
    compute_cdi,
    compute_svr,
    compute_ric,
    compute_daf,
    compute_tool_hallucination_rate,
    compute_composite_reliability,
    MetricAggregator,
)
from src.failure_taxonomy import FAILURE_MODES, detect_failures
from src.trace_generator import SyntheticTraceGenerator
from src.evaluation_pipeline import run_full_analysis

__all__ = [
    "compute_cdi", "compute_svr", "compute_ric", "compute_daf",
    "compute_tool_hallucination_rate", "compute_composite_reliability",
    "MetricAggregator", "FAILURE_MODES", "detect_failures",
    "SyntheticTraceGenerator", "run_full_analysis",
]
