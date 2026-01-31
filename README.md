# Bayesian Demo

A research project comparing three approaches to predicting human height and weight as probability distributions using Claude AI and Bayesian reasoning.

## Overview

This project evaluates different methods for making probabilistic predictions (distributions, not point estimates) about people's physical characteristics based on textual descriptions.

### The Three Approaches

| Approach | Tools | Description |
|----------|-------|-------------|
| **Baseline** | None | Pure Claude synthesis without web search |
| **Web Search** | Web search | Claude with access to population statistics |
| **Probabilistic** | Web search + PyMC | Bayesian reasoning with explicit uncertainty quantification |

### Why Distributions?

Instead of predicting "175cm tall", we predict "height ~ Normal(175cm, σ=6cm)" to:
- Explicitly model uncertainty
- Enable rigorous evaluation (KL divergence, Wasserstein distance)
- Compare how well different approaches quantify confidence

## Quick Start

### Prerequisites

- Python 3.10+
- Claude Code CLI authenticated (run `claude` in terminal to verify)

### Setup

```bash
# Clone/navigate to project
cd bayesian-demo

# Create virtual environment and install dependencies
uv venv venv
source .venv/bin/activate
uv pip install -e .

# Verify test data exists
ls data/subjects.json  # Should show 5 template subjects
```

### Run Pilot Test

```bash
python src/example_runner.py
```

This will:
- Load 5 subjects from `data/subjects.json`
- Run all 3 approaches (15 total predictions)
- Calculate evaluation metrics
- Save results to `results/`
- Print summary table

Expected runtime: 2-3 minutes for 15 predictions.

## Project Structure

```
bayesian-demo/
├── README.md                  # This file
├── pyproject.toml             # Dependencies (claude-agent-sdk, pydantic, etc.)
│
├── data/
│   └── subjects.json          # Test subjects with ground truth
│
├── src/
│   ├── example_runner.py      # Main experiment runner
│   │   ├── ClaudePredictor    # Wrapper for Claude Agent SDK
│   │   └── ExperimentRunner   # Orchestrates experiments
│   │
│   ├── models/
│   │   └── schemas.py         # Pydantic data models
│   │       ├── DistributionParams
│   │       ├── PredictionResult
│   │       ├── GroundTruth
│   │       ├── EvaluationMetrics
│   │       └── AggregatedMetrics
│   │
│   ├── evaluation/
│   │   └── metrics.py         # Evaluation functions
│   │       ├── calculate_kl_divergence_normal()
│   │       ├── calculate_wasserstein_distance_normal()
│   │       ├── evaluate_prediction()
│   │       └── aggregate_results()
│   │
│   └── prompts/
│       └── probabilistic_agent_prompt.md  # System prompt for approach C
│
└── results/                   # Generated output
    ├── experiment_results.md  # Markdown table
    ├── experiment_results.csv # CSV format
    ├── all_results.json       # Complete results
    └── intermediate/          # Crash recovery checkpoints
```

## Test Data Format

Each subject in `data/subjects.json` has:

```json
{
  "subject_id": "001",
  "text_description": "Sarah is a 32-year-old Norwegian woman who works as a software engineer...",
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
}
```

**Current dataset**: 5 template subjects (for pilot testing)
**Target**: 50 subjects for full experiment

## How It Works

### 1. ClaudePredictor

Wrapper around Claude Agent SDK's `query()` function:

```python
from claude_agent_sdk import query, ClaudeAgentOptions, AssistantMessage

predictor = ClaudePredictor(approach="baseline")
result = await predictor.predict(person_description)
```

Key features:
- Approach-specific system prompts
- Retry logic (3 attempts)
- JSON parsing with markdown code block handling
- Pydantic validation

### 2. Evaluation Metrics

**KL Divergence** (Kullback-Leibler):
- Measures difference between probability distributions
- Lower is better (0 = perfect match)
- Closed-form solution for normal distributions

**Wasserstein Distance**:
- L2 distance between distributions
- Interpretable in original units (cm or kg)
- Geometric interpretation

**Invalid Rate**:
- Percentage of predictions that failed to produce valid distributions
- Important for comparing reliability across approaches

### 3. System Prompts

**Baseline**: Simple instructions to output distributions
**Web Search**: Encourages using web search for population statistics
**Probabilistic**: Full Bayesian reasoning prompt with PyMC code generation

See `src/prompts/probabilistic_agent_prompt.md` for the complete probabilistic prompt.

## Authentication

This project uses **Claude Code authentication** (not API keys):

```bash
# Authenticate Claude Code (one time)
claude

# Verify authentication
claude --version
```

The `claude-agent-sdk` automatically uses Claude Code's authentication when available.

**No API key needed!** The SDK will use your Claude Code session.

## Running Experiments

### Pilot Test (5 subjects)

```bash
cd src
python example_runner.py
```

### Full Experiment (50 subjects)

1. Create 50 subjects in `data/subjects.json`
2. Run the same command
3. Wait ~15-20 minutes for 150 predictions

### Single Prediction Test

```bash
python test_single.py  # Tests SDK integration
```

## Results

After running, check:

**Markdown table** (`results/experiment_results.md`):
```markdown
| Approach      | N Valid | Invalid Rate (%) | KL Div (Height) | KL Div (Weight) | ...
|---------------|---------|------------------|-----------------|-----------------|-----
| baseline      | 5/5     | 0.0              | 0.523           | 0.612           | ...
| web_search    | 5/5     | 0.0              | 0.412           | 0.501           | ...
| probabilistic | 5/5     | 0.0              | 0.298           | 0.387           | ...
```

**CSV** (`results/experiment_results.csv`):
- Import into Excel, R, Python for analysis
- All metrics included

**JSON** (`results/all_results.json`):
- Complete results for every prediction
- Includes reasoning, predictions, ground truth, metrics
- Useful for debugging and detailed analysis

## Evaluation Metrics Explained

For each prediction, we calculate:

| Metric | Description | Formula (Normal distributions) |
|--------|-------------|-------------------------------|
| **KL Divergence** | Information-theoretic distance | `log(σ_true/σ_pred) + (σ_pred² + (μ_pred - μ_true)²)/(2σ_true²) - 0.5` |
| **Wasserstein-2** | Geometric L2 distance | `√((μ_pred - μ_true)² + (σ_pred - σ_true)²)` |
| **MAE (mu)** | Absolute error on mean | `|μ_pred - μ_true|` |
| **MAE (sigma)** | Absolute error on std dev | `|σ_pred - σ_true|` |

Aggregated across all subjects:
- Mean of each metric
- Standard deviation
- Count of valid vs invalid predictions

## Customization

### Change Model

In `src/example_runner.py`:

```python
# Use Opus for better performance
self.options = ClaudeAgentOptions(model="claude-opus-4-5-20251101")
```

### Adjust Retries

```python
prediction = await predictor.predict(description, max_retries=5)
```

### Add Custom Metrics

In `src/evaluation/metrics.py`:

```python
def custom_metric(pred_dist, true_dist):
    # Your implementation
    return score

# Add to evaluate_prediction() function
metrics.custom_score = custom_metric(prediction.height_distribution, ground_truth.height)
```

### Modify Prompts

Edit the prompts in `src/example_runner.py` (baseline, web_search) or `src/prompts/probabilistic_agent_prompt.md` (probabilistic).

## Troubleshooting

### JSON Parsing Errors

```
JSONDecodeError: Expecting value: line 1 column 1 (char 0)
```

**Solution**: Check if Claude is returning markdown code blocks. The code handles this with:
```python
if "```json" in response_text:
    response_text = response_text.split("```json")[1].split("```")[0]
```

### Authentication Errors

```
Could not resolve authentication method
```

**Solution**: Authenticate Claude Code:
```bash
claude
```

### Import Errors

```
ModuleNotFoundError: No module named 'claude_agent_sdk'
```

**Solution**:
```bash
source venv/bin/activate
pip install -e .
```

### Missing Test Data

```
FileNotFoundError: data/subjects.json
```

**Solution**: The template is already copied to `data/subjects.json`. If missing:
```bash
cp data/subjects_template.json data/subjects.json
```

## Development

### Debug SDK Integration

```bash
python debug_sdk.py  # Inspect message types and structure
```

### Test Single Prediction

```bash
python test_single.py  # Quick validation without full experiment
```

### Check Intermediate Results

During long runs, results are saved immediately:
```bash
ls results/intermediate/
# Shows: baseline_001.json, baseline_002.json, etc.
```

## Future Enhancements

1. **Web Search Tool Integration**: Currently simulated in prompts, could use actual web search tool
2. **PyMC Execution**: Execute generated PyMC code to refine distributions (currently uses LLM-provided params)
3. **Extended Metrics**: Add coverage probability, calibration metrics
4. **Visualization**: Plot predicted vs true distributions
5. **Dataset Expansion**: 50+ subjects with diverse demographics

## Technical Details

### Claude Agent SDK Usage

```python
from claude_agent_sdk import query, ClaudeAgentOptions, AssistantMessage

async for message in query(prompt=prompt, options=ClaudeAgentOptions()):
    if isinstance(message, AssistantMessage):
        for block in message.content:
            if hasattr(block, "text"):
                response_text += block.text
```

**Important**: Use `isinstance(message, AssistantMessage)` not `message.type == 'assistant'` because SDK messages don't have a `.type` attribute.

### Dependencies

Core:
- `claude-agent-sdk` - Claude AI integration with Code authentication
- `pydantic>=2.0.0` - Data validation
- `numpy>=1.24.0` - Numerical operations
- `scipy>=1.10.0` - Statistical functions
- `pandas>=2.0.0` - Data analysis

Optional:
- `pymc>=5.25.1` - For executing generated probabilistic models
- `arviz>=0.22.0` - PyMC visualization
- `matplotlib>=3.7.0` - Plotting
- `seaborn>=0.12.0` - Statistical visualization

### Cost Estimate

Claude Code + Claude Agent SDK pricing:
- Pilot test (5 subjects × 3 approaches = 15 predictions): ~2-3 minutes
- Full experiment (50 subjects × 3 approaches = 150 predictions): ~15-20 minutes

No direct API costs when using Claude Code authentication.

## References

- **Claude Agent SDK**: https://platform.claude.com/docs/en/agent-sdk/quickstart
- **PyMC Documentation**: https://www.pymc.io/
- **KL Divergence**: https://en.wikipedia.org/wiki/Kullback%E2%80%93Leibler_divergence
- **Wasserstein Distance**: https://en.wikipedia.org/wiki/Wasserstein_metric

## License

This is a research demo project.

## Contributing

This is a research prototype. For issues or questions, refer to the codebase documentation.
