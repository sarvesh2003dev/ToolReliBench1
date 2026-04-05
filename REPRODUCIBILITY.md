# ToolReliBench: Reproducibility Checklist

This document provides a comprehensive checklist for reproducing ToolReliBench results.

---

## 1. Environment Setup

### 1.1 Hardware Requirements

- [ ] CPU: 8+ cores (16+ recommended)
- [ ] RAM: 32GB minimum (64GB recommended)
- [ ] Storage: 50GB free space
- [ ] GPU: Optional (for local model inference)

### 1.2 Software Requirements

- [ ] Python 3.9 or higher
- [ ] Git
- [ ] pip or conda

### 1.3 Dependencies

```bash
pip install -r requirements.txt
```

Required packages:
- numpy >= 1.24.0
- scipy >= 1.10.0
- sentence-transformers >= 2.2.0
- matplotlib >= 3.7.0
- openai >= 1.0.0
- anthropic >= 0.8.0

---

## 2. Random Seeds

All random processes use a fixed seed for reproducibility:

```python
SEED = 42

import numpy as np
import random

np.random.seed(SEED)
random.seed(SEED)
```

Verify seed is set:
```python
print(np.random.randn(3))  # Should match: [-0.4967,  1.1552,  0.5350]
```

---

## 3. Model Versions

### 3.1 API Models

| Model | Version | API Endpoint |
|-------|---------|--------------|
| GPT-4o-mini | gpt-4o-mini-2024-07-18 | api.openai.com |
| Claude-3.5-Sonnet | claude-3-5-sonnet-20241022 | api.anthropic.com |
| Gemini-1.5 | gemini-1.5-pro-002 | generativelanguage.googleapis.com |

### 3.2 Local Models

| Model | Version | Ollama Command |
|-------|---------|----------------|
| Llama-3.3-70B | llama3.3:70b | `ollama pull llama3.3:70b` |

### 3.3 Verification

```python
import openai
client = openai.OpenAI()
response = client.models.retrieve("gpt-4o-mini")
print(response.id)  # Should match specified version
```

---

## 4. Task Generation

### 4.1 Generate Tasks

```python
from src.task_generator import generate_benchmark_tasks

tasks = generate_benchmark_tasks(
    output_dir="./tasks",
    seed=42
)
```

### 4.2 Verify Task Count

```bash
python -c "import json; tasks = json.load(open('./tasks/benchmark_tasks.json')); print(f'Total: {len(tasks)}, Short: {sum(1 for t in tasks if t[\"difficulty\"]==\"short\")}, Long: {sum(1 for t in tasks if t[\"difficulty\"]==\"long\")}')"
```

Expected output:
```
Total: 20, Short: 10, Long: 10
```

### 4.3 Task Version

Task templates are version-controlled. Check version:
```python
from src.task_generator import LongHorizonTaskGenerator
generator = LongHorizonTaskGenerator(seed=42)
print(generator.__module__)  # Verify source
```

---

## 5. Metric Implementation

### 5.1 Verify Metric Versions

```python
from src.metrics import ContextDriftIndex, SilentVerificationRate

# Check implementations match specification
cdi = ContextDriftIndex()
print(cdi.__doc__)  # Should match METRICS.md

svr = SilentVerificationRate()
print(svr.VERIFICATION_PATTERNS)  # Should match specification
```

### 5.2 Test Metrics on Sample Data

```python
import numpy as np
from src.metrics import Trajectory, StepState

# Create test trajectory
test_traj = Trajectory(
    task_id="test",
    model="test-model",
    architecture="single",
    steps=[
        StepState(
            step_number=1,
            reasoning="Test reasoning",
            tool_calls=[],
            tool_outputs=[],
            memory_state={},
            embedding=np.random.randn(384)
        )
    ],
    task_embedding=np.random.randn(384),
    success=True,
    total_tokens=100,
    cost_usd=0.01,
    latency_seconds=1.0
)

# Compute metrics
from src.metrics import MetricAggregator
aggregator = MetricAggregator()
metrics = aggregator.compute_all(test_traj)
print(metrics)
```

---

## 6. Running the Benchmark

### 6.1 Full Benchmark

```bash
python -m src.benchmark_runner \
    --output-dir ./results \
    --seed 42 \
    --models gpt-4o-mini claude-3.5-sonnet \
    --architectures single planner_executor recursive \
    --short-tasks 10 \
    --long-tasks 10 \
    --runs-per-task 20 \
    --use-synthetic
```

### 6.2 Expected Runtime

| Configuration | Approximate Runtime |
|--------------|---------------------|
| Synthetic traces only | 5-10 minutes |
| With API calls (2 models) | 2-4 hours |
| With local models | 8-12 hours |
| Full (4 models, all arch) | 12-24 hours |

### 6.3 Progress Verification

Monitor progress:
```bash
tail -f results/analysis/evaluation_report.json
```

Check trajectory count:
```bash
find results/traces -name "*.json" | wc -l
```

Expected: 1200+ traces

---

## 7. Output Verification

### 7.1 Required Output Files

```
results/
├── tasks.json                          # Generated tasks
├── traces/
│   ├── model=gpt-4o-mini/
│   │   ├── run_001.json
│   │   └── ...
│   └── model=claude-3.5-sonnet/
│       └── ...
└── analysis/
    ├── evaluation_report.json          # Main results
    ├── final_report.json               # Complete report
    └── visualizations/
        ├── report.md                   # Markdown report
        ├── results_table.tex           # LaTeX table
        ├── degradation_cdi.csv         # CDI curves
        ├── degradation_hallucination.csv
        └── degradation_curves.png      # Plots
```

### 7.2 Verify Report Structure

```python
import json

with open('results/analysis/final_report.json') as f:
    report = json.load(f)

# Check required keys
required_keys = [
    'benchmark', 'version', 'models_evaluated',
    'individual_results', 'comparisons', 'key_findings'
]

for key in required_keys:
    assert key in report, f"Missing key: {key}"

print("Report structure verified")
```

### 7.3 Verify Statistical Summaries

```python
for model_key, result in report['individual_results'].items():
    stats = result['statistical_summaries']
    
    # Check each metric has required fields
    for metric in ['cdi', 'svr', 'ric']:
        assert 'mean' in stats[metric]
        assert 'std' in stats[metric]
        assert 'ci_95' in stats[metric]
        assert 'n' in stats[metric]
    
    print(f"{model_key}: OK")
```

---

## 8. Validation Tests

### 8.1 Metric Sanity Checks

```python
def validate_metrics(report):
    """Validate metric values are in expected ranges."""
    for model_key, result in report['individual_results'].items():
        stats = result['statistical_summaries']
        
        # CDI should be in [0, inf)
        cdi_mean = stats['cdi']['mean']
        assert 0 <= cdi_mean < 10, f"CDI out of range: {cdi_mean}"
        
        # SVR should be in [0, 1]
        svr_mean = stats['svr']['mean']
        assert 0 <= svr_mean <= 1, f"SVR out of range: {svr_mean}"
        
        # RIC should be in [0, 1]
        ric_mean = stats['ric']['mean']
        assert 0 <= ric_mean <= 1, f"RIC out of range: {ric_mean}"
    
    print("All metrics in valid ranges")

validate_metrics(report)
```

### 8.2 Statistical Test Validation

```python
def validate_comparisons(report):
    """Validate statistical comparisons."""
    for comp_key, comparison in report['comparisons'].items():
        for metric, stats in comparison.items():
            # p-value should be in [0, 1]
            p_val = stats['p_value']
            assert 0 <= p_val <= 1, f"Invalid p-value: {p_val}"
            
            # Cohen's d should be finite
            d = stats['cohens_d']
            assert abs(d) < 100, f"Suspicious Cohen's d: {d}"
    
    print("All comparisons valid")

validate_comparisons(report)
```

---

## 9. Reproducibility Checklist

### 9.1 Pre-Experiment

- [ ] Hardware meets minimum requirements
- [ ] Python version 3.9+
- [ ] All dependencies installed
- [ ] Random seed set to 42
- [ ] Model versions verified
- [ ] Task generation successful (20 tasks)
- [ ] Metric implementations verified

### 9.2 During Experiment

- [ ] No errors in trajectory generation
- [ ] Progress logging active
- [ ] Trajectory count increasing
- [ ] No out-of-memory errors
- [ ] API rate limits respected

### 9.3 Post-Experiment

- [ ] All output files present
- [ ] Report structure valid
- [ ] Metric values in expected ranges
- [ ] Statistical tests computed
- [ ] Visualizations generated
- [ ] Key findings documented

### 9.4 Validation

- [ ] Results match expected trends
- [ ] Degradation curves monotonic (where expected)
- [ ] Statistical significance thresholds met
- [ ] Effect sizes reasonable
- [ ] Counterintuitive findings replicated

---

## 10. Troubleshooting

### 10.1 Common Issues

**Issue: Out of memory**
- Solution: Reduce batch size, use smaller embedding model

**Issue: API rate limits**
- Solution: Add delays between requests, use exponential backoff

**Issue: Missing dependencies**
- Solution: `pip install -r requirements.txt --upgrade`

**Issue: Different results**
- Check: Random seed, model version, task version

### 10.2 Getting Help

- Check documentation: https://toolrelibench.readthedocs.io
- Open issue: https://github.com/your-org/toolrelibench/issues
- Contact: toolrelibench@research.org

---

## 11. Citation

If you reproduce ToolReliBench results, please cite:

```bibtex
@misc{toolrelibench2024,
  title={ToolReliBench: Long-Horizon Reliability Evaluation for Tool-Using Agents},
  author={ToolReliBench Research Team},
  year={2024},
  howpublished={\url{https://github.com/your-org/toolrelibench}}
}
```

---

## 12. Version History

| Version | Date | Changes |
|---------|------|---------|
| 1.0.0 | 2024-11-01 | Initial release |

---

## 13. Acknowledgments

This reproducibility protocol was developed with reference to:
- ML Reproducibility Checklist (Pineau et al., 2021)
- NeurIPS Reproducibility Guidelines
- ACM Artifact Review Badging
