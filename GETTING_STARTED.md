# Getting Started with Bayesian Demo

## Quick Start (5 minutes)

### 1. Run Setup Script
```bash
cd /Users/palin/PycharmProjects/bayesian-demo
./setup.sh
```

This will:
- Check Python version (requires 3.10+)
- Create virtual environment
- Install dependencies
- Initialize git repository

### 2. Activate Virtual Environment
```bash
source venv/bin/activate
```

### 3. Set API Key
```bash
export ANTHROPIC_API_KEY="sk-ant-..."
```

### 4. Create Test Data
```bash
# Copy template
cp data/subjects_template.json data/subjects.json

# Edit to add more subjects
# Template has 5 subjects, you need 50 total for full experiment
# For pilot testing, 5 subjects is fine
```

### 5. Run Pilot Test
```bash
cd src
python example_runner.py
```

## What Happens During the Experiment

The runner will:
1. Load subjects from `data/subjects.json`
2. For each of 3 approaches (baseline, web_search, probabilistic):
   - Make predictions for each subject
   - Calculate metrics (KL divergence, Wasserstein, etc.)
   - Save intermediate results to `results/intermediate/`
3. Aggregate results across all subjects
4. Generate output files:
   - `results/experiment_results.md` - Markdown table
   - `results/experiment_results.csv` - CSV format
   - `results/all_results.json` - Complete data

## Expected Output

```
===================================================================
Running experiment: baseline
===================================================================

[baseline] Processing subject 1/5...
[baseline] Processing subject 2/5...
...

baseline completed:
  Valid: 5/5
  Invalid rate: 0.0%
  Mean KL (height): 0.423
  Mean KL (weight): 0.512

===================================================================
Running experiment: web_search
===================================================================
...
```

## Cost Estimate

- **Pilot test (5 subjects)**: ~$0.10
- **Full test (50 subjects)**: ~$1-2.50

## Troubleshooting

### Import Errors
```bash
# Make sure you're in the src/ directory when running
cd src
python example_runner.py
```

### API Key Not Found
```bash
# Check if set
echo $ANTHROPIC_API_KEY

# Set it
export ANTHROPIC_API_KEY="your-key"
```

### Test Data Not Found
```bash
# Check if file exists
ls -l data/subjects.json

# If not, copy template
cp data/subjects_template.json data/subjects.json
```

### Module Import Errors
```bash
# Reinstall in development mode
pip install -e .
```

## File Structure Overview

```
bayesian-demo/
├── setup.sh                    # Setup script
├── PROJECT_OVERVIEW.md         # High-level overview
├── GETTING_STARTED.md          # This file
├── README.md                   # Full documentation
├── pyproject.toml              # Dependencies
│
├── data/
│   ├── subjects_template.json  # Example data (5 subjects)
│   └── subjects.json           # Your test data (create this)
│
├── src/
│   ├── example_runner.py       # Main script - RUN THIS
│   ├── models/schemas.py       # Data models
│   ├── evaluation/metrics.py   # Evaluation metrics
│   └── prompts/probabilistic_agent_prompt.md  # System prompt
│
└── results/                    # Output directory
    ├── experiment_results.md   # Generated table
    ├── experiment_results.csv  # Generated CSV
    ├── all_results.json        # Complete results
    └── intermediate/           # Crash recovery checkpoints
```

## Next Steps

1. **Run pilot test** with 5 subjects (template data)
2. **Review output** in `results/`
3. **Create full dataset** (50 subjects)
4. **Run full experiment**
5. **Analyze results** for your paper

## Need Help?

- **Setup issues**: See `PROJECT_OVERVIEW.md`
- **Code details**: See `README.md`
- **Architecture**: See `/Users/palin/PycharmProjects/transcribe/REUSABLE_COMPONENTS.md`

## Tips

- Start with the 5-subject template to validate everything works
- Check intermediate results in `results/intermediate/` during long runs
- API calls are retried 3 times automatically on failure
- Each subject prediction is saved immediately (crash-safe)
