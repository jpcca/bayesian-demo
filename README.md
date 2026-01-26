# Bayesian Height/Weight Prediction Calibration Study

A research project evaluating LLM uncertainty calibration for anthropometric predictions. Compares three prompting strategies using real NHANES population data.

## Research Question

> **Are LLM uncertainty estimates well-calibrated?**
>
> When an LLM predicts "height ~ Normal(175cm, σ=6cm)", does the 90% prediction interval actually contain the true value 90% of the time?

## Overview

This project evaluates whether structured prompting strategies (web search, Bayesian reasoning) improve the **calibration** of LLM-generated uncertainty estimates.

### The Three Approaches

| Approach | Tools | Description |
|----------|-------|-------------|
| **Baseline** | None | Pure Claude synthesis without tools |
| **Web Search** | DuckDuckGo | Claude with real web search for population statistics |
| **Probabilistic** | Web search + PyMC | Bayesian reasoning with actual MCMC execution |

### Why Calibration?

Traditional accuracy metrics (MAE, RMSE) ignore uncertainty. We focus on:

- **Coverage probability**: Does the 90% interval contain 90% of true values?
- **Interval score**: A proper scoring rule rewarding calibration + sharpness
- **Sharpness**: Narrower intervals are better *if* calibrated

## Key Features

- **Real data**: NHANES (CDC) anthropometric measurements, not synthetic ground truth
- **Actual web search**: DuckDuckGo integration for population statistics
- **PyMC execution**: Generated Bayesian models are actually run, posteriors extracted
- **Rigorous statistics**: Friedman test, Wilcoxon signed-rank with Bonferroni correction
- **Power-analyzed**: n=50 subjects for detecting medium effects (d=0.5)

## Quick Start

### Prerequisites

- Python 3.10+
- Claude Code CLI authenticated (`claude` in terminal)

### Setup

```bash
cd bayesian-demo

# Create virtual environment and install dependencies
python3 -m venv venv
source venv/bin/activate
pip install -e .

# Optional: Install PyMC for probabilistic approach
pip install -e ".[pymc]"

# Optional: Install web search for real searches
pip install -e ".[dev]"
```

### Run Experiment

```bash
cd src
python example_runner.py
```

This will:
1. Download NHANES data (cached locally)
2. Generate 50 subject vignettes with stratified demographics
3. Run all 3 approaches (150 total predictions)
4. Calculate calibration metrics
5. Run statistical tests
6. Save results to `results/`

Expected runtime: 15-20 minutes for 150 predictions.

## Project Structure

```
bayesian-demo/
├── README.md                  # This file
├── pyproject.toml             # Dependencies
│
├── data/
│   ├── nhanes/                # Cached NHANES data
│   └── subjects_template.json # Legacy template
│
├── src/
│   ├── example_runner.py      # Main experiment runner
│   │
│   ├── data/
│   │   └── nhanes_loader.py   # NHANES download, vignette generation
│   │
│   ├── models/
│   │   └── schemas.py         # Pydantic data models
│   │       ├── Subject              # NHANES subject with actual measurements
│   │       ├── PredictionResult     # LLM prediction output
│   │       ├── CalibrationMetrics   # Per-prediction calibration
│   │       └── AggregatedCalibrationMetrics
│   │
│   ├── evaluation/
│   │   ├── metrics.py         # KL divergence, Wasserstein (legacy)
│   │   ├── calibration.py     # Coverage, interval score
│   │   └── statistical_tests.py  # Friedman, Wilcoxon, McNemar
│   │
│   ├── tools/
│   │   ├── web_search.py      # DuckDuckGo integration
│   │   └── pymc_executor.py   # Sandboxed PyMC execution
│   │
│   └── prompts/
│       └── probabilistic_agent_prompt.md  # Bayesian reasoning prompt
│
├── tests/
│   ├── test_metrics.py        # Unit tests for evaluation
│   └── conftest.py            # Pytest configuration
│
└── results/                   # Generated output
    ├── calibration_results.md
    ├── statistical_report.md
    └── all_results.json
```

## Ground Truth: NHANES Data

We use real measurements from the CDC's National Health and Nutrition Examination Survey:

```python
from data.nhanes_loader import load_subjects

subjects = load_subjects(n=50, cycle='2017-2018', seed=42)

# Each subject has:
# - subject_id: str
# - text_description: str (generated vignette)
# - actual_height_cm: float (real measurement)
# - actual_weight_kg: float (real measurement)
# - age, gender, ethnicity (for analysis)
```

**Vignette example**:
> "Alex is a 34-year-old Mexican American man who works in construction. 
> He describes himself as moderately active and enjoys playing soccer on weekends."

**Key**: Vignettes never contain actual height/weight values.

## Evaluation Metrics

### Primary: Calibration

| Metric | Description | Target |
|--------|-------------|--------|
| **Coverage@90** | % of true values in 90% interval | 90% |
| **Calibration Error** | \|observed - nominal\| | 0% |
| **Interval Score** | Proper scoring rule | Lower is better |

### Secondary: Sharpness & Accuracy

| Metric | Description |
|--------|-------------|
| **Mean Interval Width** | Narrower is better (if calibrated) |
| **MAE(μ)** | Mean accuracy of point prediction |

## Statistical Analysis

### Tests

| Test | Use Case |
|------|----------|
| **Friedman** | Overall difference among 3 approaches |
| **Wilcoxon** | Pairwise comparisons with Bonferroni |
| **McNemar** | Compare coverage rates |

### Power Analysis

With n=50 subjects:
- Can detect medium effects (Cohen's d = 0.5)
- Can detect 10% coverage differences
- 95% CI on coverage ≈ ±10%

## Results Format

### Calibration Table (`results/calibration_results.md`)

```markdown
| Approach      | Coverage@90 (Height) | Coverage@90 (Weight) | Calibration Error | Mean Interval Score |
|---------------|----------------------|----------------------|-------------------|---------------------|
| baseline      | 78%                  | 82%                  | 12%               | 15.3                |
| web_search    | 85%                  | 88%                  | 5%                | 12.1                |
| probabilistic | 91%                  | 89%                  | 1%                | 10.8                |
```

### Statistical Report (`results/statistical_report.md`)

```markdown
## Friedman Test
Statistic: 12.4, p = 0.002 (significant)

## Pairwise Comparisons (Bonferroni-corrected)
| Comparison                    | p-value | Effect Size | Significant |
|-------------------------------|---------|-------------|-------------|
| baseline vs web_search        | 0.023   | 0.42        | Yes         |
| baseline vs probabilistic     | 0.001   | 0.68        | Yes         |
| web_search vs probabilistic   | 0.156   | 0.24        | No          |
```

## How It Works

### 1. Data Loading

```python
from data.nhanes_loader import load_subjects

# Downloads NHANES, generates diverse vignettes
subjects = load_subjects(n=50)
```

### 2. Prediction

```python
predictor = ClaudePredictor(approach="probabilistic")
result = await predictor.predict(subject.text_description)

# Returns:
# - height_distribution: Normal(mu, sigma)
# - weight_distribution: Normal(mu, sigma)
# - pymc_code: str (for probabilistic approach)
```

### 3. PyMC Execution (Probabilistic Approach)

```python
from tools.pymc_executor import execute_pymc_code

if result.pymc_code:
    posterior = execute_pymc_code(result.pymc_code, timeout=60)
    # Updates distributions with actual MCMC posterior
```

### 4. Calibration Evaluation

```python
from evaluation.calibration import evaluate_calibration

metrics = evaluate_calibration(prediction, subject)
# Checks: Is true value in predicted 50/80/90/95% intervals?
```

### 5. Statistical Testing

```python
from evaluation.statistical_tests import friedman_test, all_pairwise_comparisons

friedman = friedman_test({'baseline': scores_b, 'web': scores_w, 'prob': scores_p})
pairwise = all_pairwise_comparisons(results_by_approach)
```

## Configuration

### Change Sample Size

```python
# In main():
subjects = load_calibration_subjects(n=100)  # More subjects
```

### Change NHANES Cycle

```python
subjects = load_subjects(n=50, cycle='2015-2016')  # Different year
```

### Disable PyMC Execution

```python
# In ClaudePredictor.predict():
# Comment out the pymc execution block
```

## Dependencies

**Core**:
- `claude-agent-sdk` - Claude AI integration
- `pydantic>=2.0.0` - Data validation
- `numpy>=1.24.0` - Numerical operations
- `scipy>=1.10.0` - Statistical functions
- `pandas>=2.0.0` - Data analysis

**Optional [pymc]**:
- `pymc>=5.10.0` - Bayesian inference
- `arviz>=0.17.0` - Posterior analysis

**Optional [dev]**:
- `duckduckgo-search>=6.0.0` - Web search
- `pytest>=9.0.1` - Testing

## Running Tests

```bash
pytest tests/ -v
```

Tests cover:
- KL divergence properties
- Wasserstein distance properties  
- Calibration metric calculations
- Statistical test behavior

## Troubleshooting

### NHANES Download Fails

```
HTTPError: 403 Forbidden
```

**Solution**: CDC servers may be temporarily unavailable. Wait and retry, or use cached data.

### PyMC Timeout

```
TimeoutError: PyMC execution exceeded 60 seconds
```

**Solution**: Increase timeout or skip PyMC execution for faster runs.

### Web Search Rate Limited

```
RateLimitExceeded: Maximum 3 searches per prediction
```

**Solution**: This is intentional to prevent abuse. Falls back to simulated data.

### Authentication Errors

```
Could not resolve authentication method
```

**Solution**: Run `claude` in terminal to authenticate.

## Citation

If using this code or methodology, please cite:

```bibtex
@software{bayesian_calibration_demo,
  title = {Bayesian Height/Weight Prediction Calibration Study},
  year = {2025},
  note = {Evaluating LLM uncertainty calibration with NHANES data}
}
```

## Data Sources

- **NHANES**: CDC National Health and Nutrition Examination Survey
  - Public domain, no license restrictions
  - https://wwwn.cdc.gov/nchs/nhanes/

## References

- Gneiting, T., & Raftery, A. E. (2007). Strictly proper scoring rules, prediction, and estimation. *JASA*.
- NHANES: https://wwwn.cdc.gov/nchs/nhanes/
- PyMC: https://www.pymc.io/

## License

Research demo project. NHANES data is public domain.
