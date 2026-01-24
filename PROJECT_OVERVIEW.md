# Bayesian Demo - Project Overview

Quick reference guide for the bayesian-demo research project.

## What This Project Does

Compares three approaches for predicting human height and weight as **probability distributions** (not point estimates) using Claude AI.

### Input
Text descriptions of people:
```
"Sarah is a 32-year-old Norwegian woman who works as a software engineer.
She mentioned playing volleyball in college and still plays recreationally
on weekends..."
```

### Output
Probability distributions with uncertainty:
```json
{
  "height_distribution": {
    "distribution_type": "normal",
    "mu": 178,
    "sigma": 5,
    "unit": "cm"
  },
  "weight_distribution": {
    "distribution_type": "normal",
    "mu": 72,
    "sigma": 7,
    "unit": "kg"
  }
}
```

## The Three Approaches

| Approach | Description | Tools |
|----------|-------------|-------|
| **Baseline** | Pure Claude synthesis | None |
| **Web Search** | Claude with population statistics | Web search (simulated in prompt) |
| **Probabilistic** | Bayesian reasoning with PyMC | Web search + PyMC code generation |

## Key Innovation

**Probabilistic predictions instead of point estimates**:
- Traditional: "Height = 175cm" (no uncertainty)
- This project: "Height ~ Normal(175cm, σ=6cm)" (explicit confidence)

This enables:
- Rigorous evaluation (KL divergence, Wasserstein distance)
- Uncertainty quantification
- Comparison of how well approaches model confidence

## Project Status

✅ **Completed**:
- Project structure created
- Claude Agent SDK integrated
- Evaluation metrics implemented (KL divergence, Wasserstein)
- System prompts adapted for all three approaches
- Test data (5 subjects) ready
- Single prediction test passing

⏳ **In Progress**:
- Pilot test with 5 subjects
- Full experiment with 50 subjects (need to create dataset)

🔮 **Future Enhancements**:
- Integrate actual web search tool (currently simulated)
- Execute generated PyMC code (currently use LLM params)
- Expand to 50+ diverse subjects

## Quick Start

```bash
# Setup
cd bayesian-demo
python3 -m venv venv
source venv/bin/activate
pip install -e .

# Run pilot test (5 subjects)
cd src
python example_runner.py

# View results
cat ../results/experiment_results.md
```

## File Structure

```
bayesian-demo/
├── src/
│   ├── example_runner.py          # Main script ← RUN THIS
│   ├── models/schemas.py          # Data models
│   ├── evaluation/metrics.py      # KL divergence, Wasserstein
│   └── prompts/                   # System prompts
│
├── data/
│   └── subjects.json              # 5 test subjects
│
├── results/                       # Generated after running
│   ├── experiment_results.md      # Summary table
│   └── all_results.json           # Complete data
│
└── README.md                      # Full documentation
```

## Expected Results

After running the experiment:

| Approach      | N Valid | Invalid Rate (%) | Mean KL (Height) | Mean KL (Weight) |
|---------------|---------|------------------|------------------|------------------|
| baseline      | 5/5     | 0.0              | 0.52             | 0.61             |
| web_search    | 5/5     | 0.0              | 0.41             | 0.50             |
| probabilistic | 5/5     | 0.0              | 0.30             | 0.39             |

*Actual numbers will vary - these are hypothetical*

## Key Metrics

**KL Divergence**: Information-theoretic distance between distributions
- Lower is better (0 = perfect match)
- Measures how well predicted distribution matches ground truth

**Wasserstein Distance**: Geometric L2 distance
- Interpretable in original units (cm or kg)
- Robust to outliers

**Invalid Rate**: % of predictions that failed to produce valid distributions
- Tests reliability across approaches

## Authentication

Uses **Claude Code** (not API keys):
```bash
# One-time authentication
claude

# Then SDK automatically uses your session
python example_runner.py  # No API key needed
```

## Runtime & Cost

**Pilot test** (5 subjects × 3 approaches = 15 predictions):
- Runtime: ~2-3 minutes
- Cost: Free with Claude Code

**Full experiment** (50 subjects):
- Runtime: ~15-20 minutes
- Cost: Free with Claude Code

## Next Steps

1. ✅ Verify pilot test works (5 subjects)
2. Create 50-subject dataset with diverse demographics
3. Run full experiment
4. Analyze results
5. Generate research paper table

## Technical Highlights

**Claude Agent SDK Integration**:
```python
from claude_agent_sdk import query, ClaudeAgentOptions, AssistantMessage

async for message in query(prompt=prompt, options=ClaudeAgentOptions()):
    if isinstance(message, AssistantMessage):
        # Process response
```

**Evaluation**:
```python
metrics = evaluate_prediction(prediction, ground_truth)
# Returns: kl_divergence_height, kl_divergence_weight,
#          wasserstein_height, wasserstein_weight, mae_mu, mae_sigma
```

**Experiment Runner**:
```python
runner = ExperimentRunner()
results = await runner.run_all_experiments(subjects)
runner.save_results(results)
```

## Documentation

- `README.md` - Complete documentation with API details
- `PROJECT_OVERVIEW.md` - This file (quick reference)
- `src/prompts/probabilistic_agent_prompt.md` - Bayesian reasoning prompt
- `debug_sdk.py` - SDK integration debugging
- `test_single.py` - Single prediction validation

## Common Issues

**Import error**: Activate venv with `source venv/bin/activate`

**No module claude_agent_sdk**: Run `pip install -e .`

**Auth error**: Run `claude` to authenticate Claude Code

**JSON parse error**: Check `debug_sdk.py` to verify SDK integration

## Questions?

See `README.md` for detailed documentation including:
- Complete setup instructions
- Test data format
- Customization options
- Troubleshooting guide
- Technical implementation details
