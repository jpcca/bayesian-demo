# Bayesian Demo - Project Overview

## Quick Summary

Evaluation framework comparing 3 approaches to predicting height/weight as probability distributions using Claude AI.

## The Three Approaches

| Approach | Tools | Description |
|----------|-------|-------------|
| **Baseline** | None | Pure Claude synthesis, no web search |
| **Web Search** | Web search | Claude + population statistics from web |
| **Probabilistic (Ours)** | Web search + PyMC prompting | Bayesian reasoning with explicit uncertainty quantification |

## What Makes This Different

- **Not point estimates** - Outputs probability distributions (e.g., height ~ N(175cm, σ=6cm))
- **Uncertainty quantification** - Explicitly models confidence via sigma parameter
- **Bayesian reasoning** - Updates beliefs based on evidence (demographics, activities, etc.)
- **Rigorous evaluation** - Uses KL divergence and Wasserstein distance, not just MAE

## Setup (5 minutes)

```bash
# 1. Install
cd bayesian-demo
pip install -e .

# 2. Set API key
export ANTHROPIC_API_KEY="your-key"

# 3. Create test data (or use template)
cp data/subjects_template.json data/subjects.json
# Edit to add 45 more subjects (50 total needed)

# 4. Run pilot test (5 subjects)
cd src
python example_runner.py
```

## Project Status

**Current state:**
- ✅ Code structure extracted from transcribe project
- ✅ Evaluation metrics implemented (KL divergence, Wasserstein)
- ✅ System prompts adapted for Claude Agent SDK
- ✅ Example runner with retry logic and error handling
- ⏳ Need to create 50-subject test dataset
- ⏳ Need to integrate Claude Agent SDK web search tool

**Next steps:**
1. Create/collect 50 diverse subject descriptions
2. Determine ground truth distributions
3. Run pilot test with 5 subjects
4. Tune prompts based on pilot results
5. Run full experiment
6. Generate results table for paper

## Key Files

- `src/example_runner.py` - Main experiment script
- `src/prompts/probabilistic_agent_prompt.md` - System prompt for approach C
- `src/models/schemas.py` - Data models (PredictionResult, GroundTruth, etc.)
- `src/evaluation/metrics.py` - KL divergence, Wasserstein distance, aggregation
- `data/subjects_template.json` - Example test data format (5 subjects)
- `README.md` - Full documentation from extraction

## Expected Results Table

After running the experiment, you'll get:

| Approach | N Valid | Invalid Rate (%) | KL Div (Height) | KL Div (Weight) | MAE Height (cm) |
|----------|---------|------------------|-----------------|-----------------|-----------------|
| baseline | 45/50   | 10.0             | 0.523           | 0.612           | 8.2             |
| web_search | 48/50 | 4.0              | 0.412           | 0.501           | 6.5             |
| probabilistic | 47/50 | 6.0           | 0.298           | 0.387           | 4.8             |

*Numbers are hypothetical - run experiment to get real results*

## Cost & Time

- **Development time**: 4-6 hours (mostly creating test data)
- **Experiment runtime**: 1-2 hours (150 API calls)
- **API cost**: ~$1-2.50 (Claude Sonnet)

## Questions?

See `README.md` for full documentation including:
- Detailed setup instructions
- Test data format
- Claude Agent SDK integration
- PyMC execution
- Troubleshooting
