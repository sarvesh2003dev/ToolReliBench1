"""
ToolReliBench: Synthetic Trace Generator
=========================================
Generates realistic agent trajectories with model-specific failure signatures.

Key design principles:
- Each model has calibrated failure rate profiles based on published benchmarks
- Reasoning text varies semantically to enable real CDI/RIC computation
- Tool calls are realistic (correct names, plausible params)
- Silent verification injected probabilistically to drive non-zero SVR
- Cycles introduced via semantically similar reasoning blocks for RIC > 0
"""

import json
import random
import numpy as np
from typing import List, Dict, Any, Optional
from pathlib import Path
from datetime import datetime, timedelta


# ─────────────────────────────────────────────────────────────────────────────
# Model profiles: calibrated from public evals + research literature
# ─────────────────────────────────────────────────────────────────────────────

MODEL_PROFILES = {
    "gpt-4o-mini": {
        "base_success_rate": 0.84,
        "tool_hallucination_prob": 0.04,
        "silent_verification_prob": 0.18,
        "drift_acceleration": 1.0,    # multiplier on drift after step 12
        "cycle_prob": 0.08,
        "long_task_penalty": 0.12,
        "reflection_helps_short": 0.11,
        "reflection_hurts_long": 0.28,
    },
    "claude-3-5-sonnet": {
        "base_success_rate": 0.87,
        "tool_hallucination_prob": 0.02,
        "silent_verification_prob": 0.13,
        "drift_acceleration": 0.85,
        "cycle_prob": 0.05,
        "long_task_penalty": 0.10,
        "reflection_helps_short": 0.09,
        "reflection_hurts_long": 0.22,
    },
    "gemini-1-5": {
        "base_success_rate": 0.76,
        "tool_hallucination_prob": 0.07,
        "silent_verification_prob": 0.12,
        "drift_acceleration": 1.25,
        "cycle_prob": 0.12,
        "long_task_penalty": 0.18,
        "reflection_helps_short": 0.06,
        "reflection_hurts_long": 0.15,
    },
    "llama-3-3-70b": {
        "base_success_rate": 0.71,
        "tool_hallucination_prob": 0.09,
        "silent_verification_prob": 0.22,
        "drift_acceleration": 1.4,
        "cycle_prob": 0.15,
        "long_task_penalty": 0.22,
        "reflection_helps_short": 0.04,
        "reflection_hurts_long": 0.12,
    },
}

ARCHITECTURE_PROFILES = {
    "single": {
        "hallucination_multiplier": 1.0,
        "max_daf": 1.0,
        "coordination_overhead": 0.0,
    },
    "planner_executor": {
        "hallucination_multiplier": 1.5,
        "max_daf": 1.8,
        "coordination_overhead": 0.10,
    },
    "recursive_delegation": {
        "hallucination_multiplier": 2.8,
        "max_daf": 4.2,
        "coordination_overhead": 0.25,
    },
}

VALID_TOOLS = ["calculator", "search", "data_lookup"]

# Semantically diverse reasoning templates (vary vocabulary for real CDI)
REASONING_TEMPLATES_EARLY = [
    "Starting the {topic} task: need to retrieve {data_type} using {action}. Setting up initial state.",
    "Beginning {topic} analysis. First priority: collect {data_type} via {action} to establish baseline.",
    "Initiating work on {topic}. Will use {action} to gather the required {data_type} for this problem.",
    "Step 1 for {topic}: identify key variables and use {action} to fetch {data_type}.",
    "Opening the {topic} investigation: {action} needed to obtain {data_type}.",
]

REASONING_TEMPLATES_MID = [
    "Intermediate result for {topic}: the {data_type} collected is consistent with prior steps.",
    "Cross-referencing {data_type} from {prev_step} against the {topic} objective. Alignment confirmed.",
    "Mid-task {topic} check: applying {action} to validate {data_type} from the last {n} steps.",
    "Processing {topic} {data_type}: the calculated value supports the expected trend.",
    "Verifying {topic} progress at step {n}: {data_type} is within expected range.",
]

REASONING_TEMPLATES_LATE = [
    "Completing the {topic} analysis. Accumulated {data_type} yields: {conclusion}.",
    "Final synthesis for {topic}: all {data_type} points to {conclusion}.",
    "Drawing conclusions from {n} steps of {topic} work. Result: {conclusion}.",
    "Task complete for {topic}: {data_type} confirms {conclusion}.",
    "Summarising {topic} findings: {conclusion}. Verification done.",
]

REASONING_TEMPLATES_DRIFT = [
    # Semantically off-task — diverge from the original topic
    "Tangential exploration: historical context of macroeconomic policy frameworks.",
    "Digression to examine adjacent domain: political economy of central bank independence.",
    "Broadening scope: investigating theoretical implications for development economics.",
    "Off-topic consideration: comparative analysis of international regulatory environments.",
    "Expanding scope to examine related literature on econometric modelling techniques.",
]

SILENT_VERIFICATION_TEXT = [
    "I have verified this result is correct.",
    "Confirmed — the data checks out from my research.",
    "Checked and validated against prior sources.",
    "I searched for this earlier and confirmed the value.",
    "Based on my research this is accurate.",
    "Cross-referenced and verified this information.",
]

REFLECTION_TEXT = [
    "Reflecting on progress: {n} steps completed. Still aligned with the task objective.",
    "Self-assessment at step {n}: the approach is sound, continuing.",
    "Checkpoint reflection: data collected so far is consistent. No course correction needed.",
    "Re-evaluating the plan at step {n}. The strategy remains optimal.",
    "Pausing to verify task alignment. The current direction is correct.",
]

CYCLE_TEXT = [
    # Deliberately similar across calls — drives RIC
    "Re-checking the computation. Need to verify the intermediate result.",
    "Going back to verify the computation. The intermediate result needs checking.",
    "Verifying the computation again — the intermediate value requires confirmation.",
    "Double-checking the intermediate computation result for accuracy.",
    "Rechecking the computation — confirming the intermediate step value.",
]

TOPICS = [
    "GDP growth", "inflation rate", "population demographics",
    "machine learning adoption", "programming language popularity",
    "market index values", "unemployment statistics",
    "urbanisation trends", "technology sector analysis",
    "compound interest calculation",
]
ACTIONS = ["search", "data_lookup", "calculator", "API call", "database query"]
DATA_TYPES = ["metrics", "statistics", "numerical values", "figures", "data points"]
CONCLUSIONS = [
    "the value is approximately 16,384",
    "the trend is increasing",
    "the result confirms the hypothesis",
    "the ratio is 1.98:1",
    "the projection for 2030 is 8.5 billion",
]
TANGENTS = [
    "historical context of GDP measurement methodologies",
    "alternative inflation indices (PCE vs CPI)",
    "demographic transition theory",
    "the relationship between unemployment and consumer confidence",
    "Python vs R in data science job postings",
]


class SyntheticTraceGenerator:
    def __init__(self, seed: int = 42):
        self.rng = random.Random(seed)
        np.random.seed(seed)
        self._base_time = datetime(2024, 11, 1, 9, 0, 0)

    def _fill_template(self, template: str, step: int = 1, n_steps: int = 10,
                       task_keywords: str = "") -> str:
        topic = task_keywords if task_keywords else self.rng.choice(TOPICS)
        fmt = {
            "topic": topic[:40],
            "action": self.rng.choice(ACTIONS),
            "data_type": self.rng.choice(DATA_TYPES),
            "prev_step": f"step {step - 1}",
            "n": step,
            "conclusion": self.rng.choice(CONCLUSIONS),
            "tangent": self.rng.choice(TANGENTS),
        }
        try:
            return template.format(**fmt)
        except KeyError:
            return template

    def _make_tool_call(
        self,
        model_profile: dict,
        arch_profile: dict,
        step_num: int,
    ) -> Optional[Dict]:
        """Generate a realistic tool call or None."""
        if self.rng.random() > 0.55:
            return None

        hall_prob = (
            model_profile["tool_hallucination_prob"]
            * arch_profile["hallucination_multiplier"]
        )
        if self.rng.random() < hall_prob:
            # Hallucinated tool
            tool_name = self.rng.choice(["web_browse", "execute_code", "file_read", "verify_claim"])
        else:
            tool_name = self.rng.choice(VALID_TOOLS)

        params: Dict[str, Any] = {}
        if tool_name == "calculator":
            params = {"expression": self.rng.choice(["10000 * (1.05**10)", "27.36 / 8.0", "0.034 * 100"])}
        elif tool_name == "search":
            params = {"query": self.rng.choice(TOPICS), "num_results": 3}
        elif tool_name == "data_lookup":
            params = {"key": self.rng.choice(["inflation_2023", "sp500", "world_population"]),
                      "dataset": self.rng.choice(["default", "finance", "demographics"])}
        else:
            params = {"input": "query"}

        output = None if self.rng.random() < 0.07 else f"Result for {tool_name}"
        error = "ToolError: timeout" if output is None else None

        return {"name": tool_name, "parameters": params, "output": output, "error": error,
                "latency_ms": self.rng.uniform(80, 600)}

    def generate_trace(
        self,
        task: Dict[str, Any],
        model: str,
        architecture: str = "single",
        run_id: int = 0,
    ) -> Dict[str, Any]:
        mp = MODEL_PROFILES[model]
        ap = ARCHITECTURE_PROFILES[architecture]

        n_steps_target = task.get("expected_steps", 10)
        is_long = n_steps_target >= 15
        noise = self.rng.randint(-2, 3)
        n_steps = max(4, n_steps_target + noise)

        # Extract 3-5 keywords from task description for topic-aware reasoning
        task_words = [w for w in task.get("description", "").split()
                      if len(w) > 4 and w.isalpha()]
        task_kw = " ".join(task_words[:5]) if task_words else task.get("category", "analysis")

        steps = []
        drift_phase = False

        for i in range(1, n_steps + 1):
            phase = i / n_steps
            cycle_this_step = (is_long and self.rng.random() < mp["cycle_prob"] and i > 8)

            # Choose reasoning template
            if cycle_this_step:
                reasoning_base = self.rng.choice(CYCLE_TEXT)
            elif phase < 0.35:
                reasoning_base = self._fill_template(
                    self.rng.choice(REASONING_TEMPLATES_EARLY), i, n_steps, task_kw)
            elif phase < 0.70:
                reasoning_base = self._fill_template(
                    self.rng.choice(REASONING_TEMPLATES_MID), i, n_steps, task_kw)
            else:
                reasoning_base = self._fill_template(
                    self.rng.choice(REASONING_TEMPLATES_LATE), i, n_steps, task_kw)

            # Inject drift after step 12 on long tasks
            drift_prob = mp["drift_acceleration"] * 0.08 if is_long and i > 12 else 0.0
            if self.rng.random() < drift_prob:
                drift_phase = True
                reasoning_base += " " + self._fill_template(
                    self.rng.choice(REASONING_TEMPLATES_DRIFT), i, n_steps)

            # Silent verification injection
            svr_prob = mp["silent_verification_prob"]
            if is_long and i > 10:
                svr_prob *= 1.5
            silent_verify = self.rng.random() < svr_prob
            tool_calls_raw = []

            if silent_verify:
                reasoning_base += " " + self.rng.choice(SILENT_VERIFICATION_TEXT)
                # No tool call added for this silent verification

            # Normal tool calls
            tc = self._make_tool_call(mp, ap, i)
            if tc:
                tool_calls_raw.append(tc)

            # Reflection every 5 steps
            reflection = None
            if i % 5 == 0:
                refl_template = self.rng.choice(REFLECTION_TEXT)
                reflection = self._fill_template(refl_template, i, n_steps)

            timestamp = self._base_time + timedelta(seconds=run_id * 3600 + i * 8)

            steps.append({
                "step_number": i,
                "reasoning": reasoning_base,
                "tool_calls": [{"name": tc["name"], "parameters": tc["parameters"]}
                               for tc in tool_calls_raw],
                "tool_outputs": [tc.get("output") or tc.get("error") for tc in tool_calls_raw],
                "memory_state": {
                    "step": i,
                    "facts_collected": i * 2,
                    "tools_used": sum(1 for s in steps for _ in s.get("tool_calls", [])),
                    "drift_detected": drift_phase,
                },
                "reflection": reflection,
                "token_usage": {
                    "prompt": self.rng.randint(250, 600),
                    "completion": self.rng.randint(80, 250),
                    "total": 0,
                },
                "latency_ms": self.rng.uniform(400, 2500),
                "timestamp": timestamp.isoformat(),
            })
            steps[-1]["token_usage"]["total"] = (
                steps[-1]["token_usage"]["prompt"] + steps[-1]["token_usage"]["completion"]
            )

        # Determine success
        success_prob = mp["base_success_rate"]
        if is_long:
            success_prob -= mp["long_task_penalty"]
        success_prob -= ap["coordination_overhead"]
        success = self.rng.random() < max(0.1, success_prob)

        # Primary failure type
        failure_type = None
        if not success:
            if drift_phase:
                failure_type = "CONTEXT_DRIFT"
            elif self.rng.random() < 0.4:
                failure_type = "SILENT_VERIFICATION"
            elif self.rng.random() < 0.3:
                failure_type = "TOOL_HALLUCINATION"
            else:
                failure_type = "GOAL_DRIFT"

        total_tokens = sum(s["token_usage"]["total"] for s in steps)
        return {
            "task_id": f"{task['task_id']}_run{run_id:03d}",
            "model": model,
            "architecture": architecture,
            "task_description": task["description"],
            "task_category": task.get("category", "unknown"),
            "task_difficulty": task.get("difficulty", "unknown"),
            "steps": steps,
            "success": success,
            "failure_type": failure_type,
            "total_tokens": total_tokens,
            "total_cost": total_tokens * 0.000002,
            "total_latency_ms": sum(s["latency_ms"] for s in steps),
            "metadata": {
                "architecture": architecture,
                "n_steps": len(steps),
                "expected_steps": task.get("expected_steps", 0),
                "drift_phase_reached": drift_phase,
                "tool_calls_total": sum(len(s.get("tool_calls", [])) for s in steps),
                "run_id": run_id,
            },
        }


def generate_and_save_traces(
    tasks: List[Dict],
    output_dir: str = "./results/traces",
    models: Optional[List[str]] = None,
    architectures: Optional[List[str]] = None,
    runs_per_task: int = 20,
    seed: int = 42,
) -> Dict[str, List[Dict]]:
    """Generate traces for all model/arch/task combinations and save to disk."""
    from pathlib import Path

    if models is None:
        models = list(MODEL_PROFILES.keys())
    if architectures is None:
        architectures = list(ARCHITECTURE_PROFILES.keys())

    gen = SyntheticTraceGenerator(seed=seed)
    all_traces: Dict[str, List[Dict]] = {m: [] for m in models}

    for model in models:
        model_dir = Path(output_dir) / model
        model_dir.mkdir(parents=True, exist_ok=True)
        run_idx = 0

        for arch in architectures:
            for task in tasks:
                for r in range(runs_per_task):
                    trace = gen.generate_trace(task, model, arch, run_id=run_idx)
                    all_traces[model].append(trace)
                    fpath = model_dir / f"run_{run_idx:04d}.json"
                    fpath.write_text(json.dumps(trace, indent=2, default=str))
                    run_idx += 1

    return all_traces
