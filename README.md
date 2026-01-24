# Extracted Components for New Project

This directory contains essential components extracted from the "transcribe" project that can be reused for the new height/weight prediction evaluation project.

## Overview

The new project will:
1. Take 50 text paragraphs describing people
2. Predict their height/weight using 3 approaches:
   - **Baseline**: Claude Agent SDK (no tools)
   - **Web Search**: Claude Agent SDK + web search
   - **Probabilistic**: Claude Agent SDK + web search + PyMC prompting
3. Evaluate using 2 metrics:
   - Distribution error (KL divergence, Wasserstein distance)
   - Invalid output rate
4. Output research paper style table

## Directory Structure

```
extracted_for_reuse/
├── README.md                          # This file
├── prompts/
│   └── probabilistic_agent_prompt.md  # System prompt for probabilistic approach
├── models/
│   └── schemas.py                     # Pydantic data models
├── evaluation/
│   └── metrics.py                     # Evaluation metrics implementation
└── example_runner.py                  # Complete example script
```

## How to Use

### 1. Copy to Your New Project

```bash
# Create new project
mkdir height_weight_eval
cd height_weight_eval

# Copy extracted components
cp -r /path/to/transcribe/extracted_for_reuse/* ./src/
```

### 2. Set Up Environment

Create `pyproject.toml`:

```toml
[project]
name = "height-weight-eval"
version = "0.1.0"
dependencies = [
    "anthropic>=0.40.0",           # Claude API
    "pydantic>=2.0.0",             # Data validation
    "numpy>=1.24.0",               # Numerical operations
    "scipy>=1.10.0",               # Statistical functions
    "pandas>=2.0.0",               # Data analysis
    "pymc>=5.25.1",                # For probabilistic approach
    "arviz>=0.22.0",               # PyMC visualization
]
```

Install:
```bash
pip install -e .
```

### 3. Create Test Data

Create `data/subjects.json` with 50 subjects:

```json
[
  {
    "subject_id": "001",
    "text_description": "Sarah is a 32-year-old Norwegian woman who works as a software engineer. She mentioned playing volleyball in college and still plays recreationally on weekends. She describes herself as taller than most of her female friends and maintains an active lifestyle with regular gym sessions.",
    "height": {
      "distribution_type": "normal",
      "mu": 178.0,
      "sigma": 4.0,
      "unit": "cm"
    },
    "weight": {
      "distribution_type": "normal",
      "mu": 72.0,
      "sigma": 6.0,
      "unit": "kg"
    }
  },
  {
    "subject_id": "002",
    "text_description": "...",
    "height": {...},
    "weight": {...}
  }
  // ... 48 more subjects
]
```

**Tips for creating test data:**
- Use real or realistic demographic distributions
- Vary ages, nationalities, activities, occupations
- Include edge cases (very tall/short, athletes, children)
- Ground truth can be based on:
  - Real data if available
  - Synthetic data from known distributions
  - Expert annotations with uncertainty

### 4. Configure API Key

```bash
export ANTHROPIC_API_KEY="your-api-key-here"
```

### 5. Run Experiments

```bash
python example_runner.py
```

This will:
1. Load 50 subjects from `data/subjects.json`
2. Run all 3 approaches (baseline, web_search, probabilistic)
3. Calculate metrics for each prediction
4. Aggregate results across all subjects
5. Save results to `results/`

### 6. View Results

Results are saved in multiple formats:

**Markdown Table** (`results/experiment_results.md`):
```markdown
| Approach | N Valid | Invalid Rate (%) | KL Div (Height) | KL Div (Weight) | ...
|----------|---------|------------------|-----------------|-----------------|...
| baseline | 45/50   | 10.0             | 0.523           | 0.612           | ...
| web_search | 48/50 | 4.0              | 0.412           | 0.501           | ...
| probabilistic | 47/50 | 6.0           | 0.298           | 0.387           | ...
```

**CSV** (`results/experiment_results.csv`):
- Importable into Excel/R/Python
- All metrics included

**JSON** (`results/all_results.json`):
- Complete results for every subject
- Includes predictions, ground truth, metrics
- Useful for debugging and detailed analysis

## Key Components Explained

### 1. Prompts

**File**: `prompts/probabilistic_agent_prompt.md`

This is the system prompt for the probabilistic approach. It:
- Instructs Claude to use Bayesian reasoning
- Encourages web search for population statistics
- Specifies output format (JSON with distributions)
- Provides examples and guidelines

**Customization**:
- Modify the prompt to emphasize different aspects
- Add domain-specific knowledge
- Adjust sigma ranges based on your data

### 2. Data Models

**File**: `models/schemas.py`

Key classes:

- `DistributionParams`: Parameters for a probability distribution (mu, sigma, type)
- `PredictionResult`: Output from a prediction agent
- `GroundTruth`: Ground truth data for a subject
- `EvaluationMetrics`: Metrics comparing prediction to ground truth
- `ExperimentResult`: Result for one subject in one approach
- `AggregatedMetrics`: Aggregated metrics across all subjects

**Validation**:
- Pydantic automatically validates data types
- Custom validators ensure sigma > 0
- `is_valid` property checks if prediction succeeded

### 3. Evaluation Metrics

**File**: `evaluation/metrics.py`

Implements:

**Distribution Metrics**:
- `calculate_kl_divergence_normal()`: KL divergence for normal distributions
  - Closed-form solution
  - Lower is better (0 = perfect match)
- `calculate_wasserstein_distance_normal()`: Wasserstein-2 distance
  - Geometric distance between distributions
  - Interpretable in original units

**Evaluation Functions**:
- `evaluate_prediction()`: Compare one prediction to ground truth
- `aggregate_results()`: Average metrics across all subjects
- `format_results_table()`: Format as markdown table

**Adding Custom Metrics**:
```python
def custom_metric(pred, truth):
    # Your implementation
    return score

# Add to calculate_distribution_error():
return {
    "kl_divergence": kl_div,
    "wasserstein_distance": wasserstein,
    "custom_metric": custom_metric(pred, truth),
}
```

### 4. Example Runner

**File**: `example_runner.py`

Main components:

**ClaudePredictor**:
- Wraps Claude API calls
- Handles different system prompts per approach
- Implements retry logic (from transcribe project)
- Parses JSON responses with error handling

**ExperimentRunner**:
- Orchestrates full experiment
- Runs 3 approaches × 50 subjects = 150 predictions
- Saves intermediate results (crash recovery)
- Aggregates and formats results

**Customization**:
```python
# Change model
self.model = "claude-opus-4-5-20251101"  # Use Opus for better performance

# Adjust retries
prediction = await predictor.predict(text, max_retries=5)

# Run subset
subjects_subset = subjects[:10]  # Test with 10 subjects first
```

## Integration with Claude Agent SDK

The current `example_runner.py` uses the basic Claude API. To use the Agent SDK with tools:

### Install Agent SDK

```bash
pip install anthropic-agent-sdk  # Replace with actual package name
```

### Modify ClaudePredictor

```python
from anthropic_agent_sdk import Agent, WebSearchTool

class ClaudePredictor:
    def __init__(self, approach):
        self.approach = approach

        if approach == "baseline":
            self.agent = Agent(
                model="claude-sonnet-4-5-20250929",
                system_prompt=self._load_prompt()
            )
        elif approach in ["web_search", "probabilistic"]:
            self.agent = Agent(
                model="claude-sonnet-4-5-20250929",
                system_prompt=self._load_prompt(),
                tools=[WebSearchTool()]
            )

    async def predict(self, person_description):
        response = await self.agent.run(person_description)
        # Parse response...
```

## PyMC Integration (Approach C)

For the probabilistic approach, you may want to actually execute the PyMC code:

### Add PyMC Executor

Create `models/pymc_executor.py`:

```python
import subprocess
import json
from pathlib import Path

def execute_pymc_code(pymc_code: str, timeout: int = 30) -> dict:
    """
    Execute PyMC code and return distribution parameters.

    Adapted from transcribe/api/main.py WebSocket endpoint.
    """
    # Write code to temp file
    temp_file = Path(f"temp_model_{os.getpid()}.py")

    with open(temp_file, "w") as f:
        f.write(pymc_code)
        f.write("""
import pymc as pm
import arviz as az
import numpy as np

# Sample the model
with model:
    trace = pm.sample(1000, tune=1000, chains=1, progressbar=False)

# Extract posterior statistics
height_samples = trace.posterior['height'].values.flatten()
weight_samples = trace.posterior['weight'].values.flatten()

output = {
    "height": {
        "mu": float(np.mean(height_samples)),
        "sigma": float(np.std(height_samples))
    },
    "weight": {
        "mu": float(np.mean(weight_samples)),
        "sigma": float(np.std(weight_samples))
    }
}

import json
with open("pymc_output.json", "w") as f:
    json.dump(output, f)
""")

    # Execute
    try:
        result = subprocess.run(
            ["python", str(temp_file)],
            capture_output=True,
            text=True,
            timeout=timeout
        )

        if result.returncode != 0:
            raise Exception(f"PyMC execution failed: {result.stderr}")

        # Read output
        with open("pymc_output.json", "r") as f:
            output = json.load(f)

        return output

    finally:
        # Cleanup
        temp_file.unlink(missing_ok=True)
        Path("pymc_output.json").unlink(missing_ok=True)
```

### Use in Predictor

```python
async def predict(self, person_description):
    # Get Claude response with PyMC code
    response = await self.agent.run(person_description)
    result = PredictionResult(**parse_response(response))

    if self.approach == "probabilistic" and result.pymc_code:
        # Execute PyMC code to refine distributions
        try:
            pymc_output = execute_pymc_code(result.pymc_code)

            # Update distributions with sampled values
            result.height_distribution.mu = pymc_output["height"]["mu"]
            result.height_distribution.sigma = pymc_output["height"]["sigma"]
            result.weight_distribution.mu = pymc_output["weight"]["mu"]
            result.weight_distribution.sigma = pymc_output["weight"]["sigma"]
        except Exception as e:
            print(f"PyMC execution failed: {e}")
            # Fall back to LLM-provided distributions

    return result
```

## Tips and Best Practices

### 1. Start Small
```bash
# Test with 5 subjects first
subjects_test = subjects[:5]
results = await runner.run_single_experiment("baseline", subjects_test)
```

### 2. Monitor Costs
- Claude API costs money
- 50 subjects × 3 approaches = 150 API calls
- Each call may be ~1000-2000 tokens
- Estimate: ~$0.50-2.00 total for Sonnet (check current pricing)

### 3. Handle Failures
- Intermediate results are saved automatically
- Check `results/intermediate/` for partial results
- Resume from checkpoint if needed

### 4. Validate Test Data
```python
# Check ground truth distributions are reasonable
for subject in subjects:
    assert 140 <= subject.height.mu <= 220, f"Invalid height for {subject.subject_id}"
    assert 40 <= subject.weight.mu <= 150, f"Invalid weight for {subject.subject_id}"
    assert subject.height.sigma >= 1.0
    assert subject.weight.sigma >= 1.0
```

### 5. Statistical Significance
With 50 subjects, check if differences are significant:
```python
from scipy import stats

# Compare two approaches
baseline_kl = [r.metrics.kl_divergence_height for r in baseline_results if r.is_success]
web_kl = [r.metrics.kl_divergence_height for r in web_results if r.is_success]

t_stat, p_value = stats.ttest_ind(baseline_kl, web_kl)
print(f"T-test: t={t_stat:.3f}, p={p_value:.4f}")
```

## Troubleshooting

### Issue: All predictions are invalid

**Possible causes:**
- Wrong JSON schema in prompt
- Claude not following instructions
- Pydantic validation too strict

**Solutions:**
1. Check a few example responses manually
2. Print `response_text` before parsing
3. Adjust schema or validation rules
4. Improve prompt clarity

### Issue: KL divergence is negative

**Cause:** Mathematical error in implementation

**Solution:** Check that you're using `log(σ_true/σ_pred)` not `log(σ_pred/σ_true)`

### Issue: Web search not working

**Solutions:**
1. Verify Agent SDK installation
2. Check WebSearchTool configuration
3. Ensure API has web search permissions
4. Fallback to basic API + manual search simulation

### Issue: PyMC execution times out

**Solutions:**
1. Reduce `draws` and `tune` (e.g., 500/500 instead of 1000/1000)
2. Increase timeout (e.g., 60s)
3. Use faster sampler (e.g., `pm.sample(sampler="jax")`)
4. Skip PyMC execution, use LLM-provided distributions

## Next Steps

1. **Create test dataset** (`data/subjects.json`)
2. **Run pilot test** (5-10 subjects to validate pipeline)
3. **Tune prompts** based on pilot results
4. **Run full experiment** (all 50 subjects × 3 approaches)
5. **Analyze results** and create visualizations
6. **Write paper** using the results table

## Additional Resources

- **Original transcribe project**: See `../` for full implementation
- **Reusable components guide**: See `../REUSABLE_COMPONENTS.md`
- **Claude Agent SDK docs**: [Link when available]
- **PyMC documentation**: https://www.pymc.io/
- **ArviZ documentation**: https://arviz-devs.github.io/arviz/

## Questions?

Refer back to the original transcribe project:
- WebSocket loop: `../api/main.py:35` (error handling, retry logic)
- LLM calling: `../api/utils.py` (multi-backend support)
- Data models: `../api/models.py` (FHIR integration, validation)
- System prompt: `../api/prompts/height_weight_system.md` (Bayesian reasoning)
