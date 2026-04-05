# ToolReliBench: Experimental Design

This document describes the complete experimental protocol for ToolReliBench.

---

## 1. Task Design

### 1.1 Task Categories

#### Short Tasks (5-10 steps)
Used as baseline for comparison:
- Single-hop research queries
- Direct calculations
- Simple lookups
- Basic comparisons
- Short planning tasks

#### Long Tasks (15-30 steps)
Primary stress tests:
- Multi-hop research chains
- Structured data analysis
- Complex planning with constraints
- Recursive verification
- Cross-reference synthesis
- Delegation stress tests
- Context drift tests
- Error recovery scenarios

### 1.2 Task Properties

Each task includes:
- **Task ID**: Unique identifier
- **Category**: Type of task
- **Difficulty**: Short, medium, or long
- **Description**: Detailed instructions
- **Expected Steps**: Estimated step count
- **Constraints**: Hard and soft constraints
- **Success Criteria**: Objective evaluation criteria
- **Ground Truth**: Reference answer (where applicable)
- **Tools Required**: Necessary tool types

### 1.3 Task Generation Strategy

Tasks are generated using:
1. Template-based generation with variable substitution
2. Controlled complexity scaling
3. Explicit constraint specification
4. Multiple difficulty levels
5. Diverse domain coverage

---

## 2. Agent Architectures

### 2.1 Single Agent

**Design:**
- Unified reasoning and execution
- Single context window
- Direct tool access
- Periodic self-reflection

**Parameters:**
- Max steps: 50
- Reflection frequency: Every 5 steps
- Memory: Full trajectory

**Expected Behavior:**
- Simplest architecture
- No coordination overhead
- Context window limitations
- Potential for drift

### 2.2 Planner + Executor

**Design:**
- Separate planning and execution components
- Planner: High-level strategy, subtask generation
- Executor: Tool selection, parameter specification
- Bidirectional communication

**Parameters:**
- Max steps: 50
- Replanning threshold: 20% deviation
- Plan format: Ordered subtask list

**Expected Behavior:**
- Better separation of concerns
- Coordination overhead
- Plan-execution misalignment possible
- More structured approach

### 2.3 Recursive Delegation

**Design:**
- Agents can spawn sub-agents
- Tree-structured execution
- Parent-child communication
- Result aggregation

**Parameters:**
- Max depth: 3
- Max branching: 5
- Delegation threshold: Subtask complexity > threshold

**Expected Behavior:**
- Natural task decomposition
- Parallelization potential
- Error propagation risk
- Delegation explosion possible

---

## 3. Evaluation Protocol

### 3.1 Trajectory Logging

Each trajectory must log:

```json
{
  "task_id": "task_001",
  "model": "gpt-4o-mini",
  "architecture": "single",
  "steps": [
    {
      "step_number": 1,
      "reasoning": "...",
      "tool_calls": [...],
      "tool_outputs": [...],
      "memory_state": {...},
      "self_reflection": "...",
      "embedding": [...],
      "tokens_used": 1200,
      "cost_usd": 0.024,
      "latency_ms": 1500
    }
  ],
  "success": true,
  "failure_type": null,
  "total_tokens": 15000,
  "total_cost": 0.30,
  "total_latency_ms": 25000
}
```

### 3.2 Metric Computation

For each trajectory:
1. Compute CDI from embeddings
2. Compute SVR from reasoning and tool calls
3. Compute RIC from state cycles
4. Compute DAF from delegation events
5. Compute composite reliability score

### 3.3 Statistical Analysis

For each model-architecture combination:
1. Compute mean and std for each metric
2. Compute 95% confidence intervals
3. Perform pairwise t-tests
4. Compute Cohen's d effect sizes
5. Identify significant differences

---

## 4. Experimental Conditions

### 4.1 Baseline Conditions

- Short tasks (5-10 steps)
- Single agent architecture
- No cost constraints
- Full tool access

### 4.2 Stress Conditions

- Long tasks (15-30 steps)
- All three architectures
- Cost-optimized mode
- Limited tool access

### 4.3 Comparison Conditions

- Multi-agent vs single-agent
- With vs without self-reflection
- With vs without verification
- Different model sizes

---

## 5. Sample Size Calculation

### 5.1 Minimum Sample Sizes

Based on power analysis:
- Effect size: d = 0.5 (medium)
- Power: 0.80
- Significance: α = 0.05
- Required: n = 64 per group

**ToolReliBench uses:**
- 20 runs per task-model-architecture
- 20 tasks (10 short + 10 long)
- 3-4 models
- 3 architectures
- Total: ~1200-2400 trajectories

### 5.2 Justification

This sample size provides:
- 95% power to detect medium effects
- 80% power to detect small-medium effects
- Robust confidence intervals
- Stable variance estimates

---

## 6. Degradation Curve Analysis

### 6.1 Curve Types

1. **CDI vs Step Count**
   - X-axis: Step number
   - Y-axis: Context Drift Index
   - Shows drift accumulation

2. **Tool Hallucination vs Step Count**
   - X-axis: Step number
   - Y-axis: Hallucination rate
   - Shows degradation pattern

3. **Success Rate vs Step Count**
   - X-axis: Step number
   - Y-axis: Cumulative success rate
   - Shows reliability decay

4. **Cost vs Step Count**
   - X-axis: Step number
   - Y-axis: Cumulative cost
   - Shows cost efficiency

### 6.2 Degradation Point Detection

Algorithm:
1. Compute baseline (first 3 steps)
2. Set threshold = baseline × 1.5
3. Find first step where value > threshold
4. Verify sustained increase (3+ subsequent steps)
5. Return degradation point

---

## 7. Failure Mode Detection

### 7.1 Detection Pipeline

For each trajectory:
1. Run all 15 failure mode detectors
2. Record confidence scores
3. Identify primary failure mode
4. Aggregate across trajectories

### 7.2 Failure Distribution Analysis

For each model-architecture:
1. Count occurrences of each failure mode
2. Compute failure mode distribution
3. Identify dominant failure modes
4. Compare across conditions

---

## 8. Counterintuitive Finding Detection

### 8.1 Hypothesis Testing

Pre-registered hypotheses:
1. Self-reflection benefit decreases with step count
2. Multi-agent reliability degrades faster than single-agent
3. Delegation has a sharp instability threshold
4. Cost optimization increases silent failures

### 8.2 Detection Criteria

A finding is counterintuitive if:
1. Effect direction contradicts common assumptions
2. Statistical significance (p < 0.05)
3. Medium or large effect size (d ≥ 0.5)
4. Replicable across task types

---

## 9. Reproducibility Measures

### 9.1 Fixed Parameters

- Random seed: 42
- Model versions: Specified in config
- Task templates: Version controlled
- Metric implementations: Frozen

### 9.2 Logging Requirements

All experiments log:
- Full trajectory JSON
- Model configuration
- Tool definitions
- Random seeds
- Environment info

### 9.3 Version Control

- Code: Git repository
- Tasks: JSON files with versioning
- Results: Timestamped directories
- Analysis: Jupyter notebooks

---

## 10. Expected Outcomes

### 10.1 Primary Hypotheses

1. **H1:** CDI increases non-linearly with step count
2. **H2:** SVR increases significantly after step 15
3. **H3:** Multi-agent architectures show higher RIC
4. **H4:** DAF > 5 correlates with task failure

### 10.2 Expected Effect Sizes

| Comparison | Expected d | Direction |
|------------|-----------|-----------|
| Short vs Long CDI | 0.8 | Long > Short |
| Single vs Multi SVR | 0.6 | Multi > Single |
| Depth 2 vs Depth 3 RIC | 1.2 | Depth 3 > Depth 2 |
| With vs Without Reflection | 0.5 | Context dependent |

---

## 11. Limitations and Mitigations

### 11.1 Known Limitations

1. **Synthetic traces** may not capture all real-world failures
2. **Embedding quality** affects CDI and RIC
3. **Pattern matching** for SVR may miss nuanced cases
4. **Computational cost** of full evaluation

### 11.2 Mitigations

1. Validate synthetic traces against real agent runs
2. Use multiple embedding models
3. Manual verification of SVR detections
4. Parallel execution and caching

---

## 12. Ethical Considerations

### 12.1 Safety

- No harmful task generation
- No prompt injection attacks on real systems
- Synthetic failures only

### 12.2 Transparency

- All metrics documented
- All failures categorized
- All results reported (no cherry-picking)

### 12.3 Reproducibility

- Open-source code
- Public dataset
- Detailed documentation
