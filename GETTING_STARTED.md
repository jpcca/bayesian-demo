# Getting Started with Bayesian Demo

Step-by-step guide to run your first experiment.

## Prerequisites

Before starting, ensure you have:

- ✅ Python 3.10 or higher
- ✅ Claude Code CLI installed and authenticated
- ✅ Terminal access

### Verify Claude Code Authentication

```bash
# Check if Claude Code is authenticated
claude

# If not authenticated, follow the prompts to sign in
# You only need to do this once
```

## Installation (5 minutes)

### 1. Navigate to Project

```bash
cd /Users/palin/PycharmProjects/bayesian-demo
```

### 2. Create Virtual Environment

```bash
python3 -m venv venv
source venv/bin/activate
```

You should see `(venv)` in your terminal prompt.

### 3. Install Dependencies

```bash
pip install -e .
```

This installs:
- `claude-agent-sdk` - Claude AI integration
- `pydantic` - Data validation
- `numpy`, `scipy` - Statistical functions
- `pandas` - Data analysis

### 4. Verify Installation

```bash
python test_single.py
```

Expected output:
```
Sending query...
Got text: ```json...
Full response:
{
  "reasoning": "Sarah is a Norwegian woman...",
  "height_distribution": {...},
  "weight_distribution": {...}
}

Parsed successfully:
{...}
```

If this works, you're ready to run experiments!

## Running Your First Experiment

### Pilot Test (5 subjects)

```bash
cd src
cd src
python claude_runner.py

```

### What Happens

The script will:

1. **Load test data** from `data/subjects.json` (5 subjects)

2. **Run baseline approach**:
   ```
   [baseline] Processing subject 1/5...
   [baseline] Processing subject 2/5...
   ...
   baseline completed:
     Valid: 5/5
     Invalid rate: 0.0%
     Mean KL (height): 0.423
   ```

3. **Run web_search approach**:
   ```
   [web_search] Processing subject 1/5...
   ...
   ```

4. **Run probabilistic approach**:
   ```
   [probabilistic] Processing subject 1/5...
   ...
   ```

5. **Save results**:
   ```
   ✓ Results saved to ../results/
   ```

6. **Print summary table**:
   ```
   FINAL RESULTS
   ============================================================

   | Approach      | N Valid | Invalid Rate (%) | KL Div (Height) | ...
   |---------------|---------|------------------|-----------------|...
   | baseline      | 5/5     | 0.0              | 0.523           | ...
   | web_search    | 5/5     | 0.0              | 0.412           | ...
   | probabilistic | 5/5     | 0.0              | 0.298           | ...
   ```

### Expected Runtime

- **5 subjects × 3 approaches = 15 predictions**
- **~6-10 seconds per prediction**
- **Total: 2-3 minutes**

## Viewing Results

After running, check the `results/` directory:

```bash
cd ../results
ls -l
```

You should see:

### 1. Markdown Table (`experiment_results.md`)

```bash
cat experiment_results.md
```

Ready-to-paste table for research papers.

### 2. CSV File (`experiment_results.csv`)

```bash
head experiment_results.csv
```

Import into Excel, R, or Python for analysis.

### 3. Complete Results (`all_results.json`)

```bash
cat all_results.json | python -m json.tool | head -50
```

Every prediction with reasoning, distributions, and metrics.

### 4. Intermediate Checkpoints (`intermediate/`)

```bash
ls intermediate/
```

Individual result files (for crash recovery):
```
baseline_001.json
baseline_002.json
web_search_001.json
...
```

## Understanding the Output

### Metrics Explained

**KL Divergence** (Lower is better):
- `0.0` = Perfect match
- `0.3` = Good match
- `1.0+` = Poor match

**Wasserstein Distance** (In cm or kg):
- Geometric distance between distributions
- Interpretable: "Off by 5cm on average"

**Invalid Rate**:
- Percentage of predictions that failed
- Target: < 5%

### Example Result

```json
{
  "subject_id": "001",
  "approach": "baseline",
  "prediction": {
    "reasoning": "Sarah is Norwegian (tall population), plays volleyball...",
    "height_distribution": {"mu": 178, "sigma": 5, "unit": "cm"},
    "weight_distribution": {"mu": 72, "sigma": 7, "unit": "kg"}
  },
  "ground_truth": {
    "height": {"mu": 178, "sigma": 4, "unit": "cm"},
    "weight": {"mu": 72, "sigma": 6, "unit": "kg"}
  },
  "metrics": {
    "kl_divergence_height": 0.125,
    "kl_divergence_weight": 0.085,
    "wasserstein_height": 1.0,
    "wasserstein_weight": 1.0,
    "mae_mu_height": 0.0,
    "mae_mu_weight": 0.0
  }
}
```

## Next Steps

### 1. Analyze Pilot Results

Review the results to see which approach performs best on the 5 test subjects.

### 2. Create Full Dataset (50 subjects)

Edit `data/subjects.json` to add 45 more subjects:

```json
[
  {
    "subject_id": "001",
    "text_description": "...",
    "height": {"distribution_type": "normal", "mu": 178, "sigma": 4, "unit": "cm"},
    "weight": {"distribution_type": "normal", "mu": 72, "sigma": 6, "unit": "kg"}
  },
  // ... add 49 more
]
```

**Tips for creating subjects**:
- Vary demographics (age, nationality, gender)
- Include different activities (athlete, sedentary, etc.)
- Mix obvious cases and edge cases
- Use realistic ground truth values

### 3. Run Full Experiment

```bash
cd src
cd src
python claude_runner.py

```

Runtime: ~15-20 minutes for 150 predictions.

### 4. Analyze Results

Import CSV into your analysis tool:

```python
import pandas as pd
results = pd.read_csv('../results/experiment_results.csv')
print(results)
```

## Troubleshooting

### Problem: `ModuleNotFoundError: No module named 'claude_agent_sdk'`

**Solution**: Activate virtual environment and install
```bash
source venv/bin/activate
pip install -e .
```

### Problem: `FileNotFoundError: data/subjects.json`

**Solution**: Verify you're in the `src/` directory
```bash
cd src
cd src
python claude_runner.py

```

### Problem: `Could not resolve authentication method`

**Solution**: Authenticate Claude Code
```bash
claude
# Follow prompts to sign in
```

### Problem: JSON parsing error

**Solution**: Run debug script to check SDK integration
```bash
python debug_sdk.py
```

### Problem: Intermediate results not saved

**Solution**: Check that `results/intermediate/` directory exists
```bash
mkdir -p results/intermediate
```

## Command Reference

```bash
# Setup
python3 -m venv venv
source venv/bin/activate
pip install -e .

# Run tests
python test_single.py           # Test SDK integration
python debug_sdk.py             # Debug SDK messages

# Run experiments
cd src
python claude_runner.py        # Run full experiment with Claude
python ollama_runner.py        # Run full experiment with Ollama


# View results
cat ../results/experiment_results.md
cat ../results/all_results.json | python -m json.tool
ls ../results/intermediate/
```

## File Locations

```
bayesian-demo/
├── test_single.py              # Quick SDK test
├── debug_sdk.py                # SDK debugging
│
├── data/
│   └── subjects.json           # Test data (5 subjects)
│
├── src/
│   ├── claude_runner.py        # Claude experiment script
│   ├── ollama_runner.py        # Ollama experiment script
│   ├── experiment_core.py      # Shared logic
│
└── results/                    # Generated output
    ├── experiment_results.md
    ├── experiment_results.csv
    ├── all_results.json
    └── intermediate/
```

## Tips

1. **Start small**: Validate with 5 subjects before creating 50
2. **Check intermediate results**: Monitor `results/intermediate/` during long runs
3. **Use debug script**: If errors occur, run `debug_sdk.py` to inspect SDK behavior
4. **Save your work**: Results are crash-safe (saved immediately after each prediction)

## Getting Help

- **Setup issues**: See `PROJECT_OVERVIEW.md`
- **Technical details**: See `README.md`
- **SDK issues**: Run `debug_sdk.py` and check Claude Agent SDK docs

## What's Next?

After successful pilot test:
1. ✅ Verify all 3 approaches work
2. Create 50-subject dataset
3. Run full experiment
4. Analyze results for research paper
5. Consider enhancements (actual web search, PyMC execution)

Happy experimenting!
