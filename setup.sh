#!/bin/bash
# Setup script for bayesian-demo project

set -e  # Exit on error

echo "==================================="
echo "Bayesian Demo - Setup Script"
echo "==================================="
echo ""

# Check Python version
echo "Checking Python version..."
python_version=$(python3 --version 2>&1 | grep -oE '[0-9]+\.[0-9]+' | head -1)
required_version="3.10"

if [ "$(printf '%s\n' "$required_version" "$python_version" | sort -V | head -n1)" != "$required_version" ]; then
    echo "❌ Error: Python $required_version or higher required (found $python_version)"
    exit 1
fi
echo "✓ Python $python_version"

# Create virtual environment
echo ""
echo "Creating virtual environment..."
if [ ! -d "venv" ]; then
    python3 -m venv venv
    echo "✓ Virtual environment created"
else
    echo "✓ Virtual environment already exists"
fi

# Activate virtual environment
echo ""
echo "Activating virtual environment..."
source venv/bin/activate

# Install dependencies
echo ""
echo "Installing dependencies..."
pip install --upgrade pip
pip install -e .
echo "✓ Dependencies installed"

# Check for API key
echo ""
echo "Checking for ANTHROPIC_API_KEY..."
if [ -z "$ANTHROPIC_API_KEY" ]; then
    echo "⚠️  Warning: ANTHROPIC_API_KEY not set"
    echo "   Set it with: export ANTHROPIC_API_KEY='your-key-here'"
else
    echo "✓ ANTHROPIC_API_KEY is set"
fi

# Check for test data
echo ""
echo "Checking for test data..."
if [ ! -f "data/subjects.json" ]; then
    echo "⚠️  Warning: data/subjects.json not found"
    echo "   Template available at: data/subjects_template.json"
    echo "   You need to create 50 subjects to run the full experiment"
else
    subject_count=$(cat data/subjects.json | python3 -c "import sys, json; print(len(json.load(sys.stdin)))")
    echo "✓ Found data/subjects.json with $subject_count subjects"
    if [ "$subject_count" -lt 50 ]; then
        echo "   ⚠️  Warning: Need 50 subjects (currently have $subject_count)"
    fi
fi

# Initialize git if not already
echo ""
if [ ! -d ".git" ]; then
    echo "Initializing git repository..."
    git init
    git add .
    git commit -m "Initial commit: Bayesian demo project setup"
    echo "✓ Git repository initialized"
else
    echo "✓ Git repository already initialized"
fi

echo ""
echo "==================================="
echo "Setup Complete!"
echo "==================================="
echo ""
echo "Next steps:"
echo "1. Activate the virtual environment:"
echo "   source venv/bin/activate"
echo ""
echo "2. Set your API key (if not already set):"
echo "   export ANTHROPIC_API_KEY='your-key-here'"
echo ""
echo "3. Create test data:"
echo "   cp data/subjects_template.json data/subjects.json"
echo "   # Edit to add 45 more subjects (50 total needed)"
echo ""
echo "4. Run pilot test (5 subjects):"
echo "   cd src"
echo "   python example_runner.py"
echo ""
echo "See PROJECT_OVERVIEW.md for more details."
echo ""
