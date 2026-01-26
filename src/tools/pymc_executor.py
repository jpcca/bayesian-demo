"""PyMC Code Executor - Safely execute PyMC code and extract posterior summaries.

This module provides sandboxed execution of PyMC probabilistic models generated
by Claude, with automatic sampling and posterior extraction.

Example:
    >>> code = '''
    ... import pymc as pm
    ... import numpy as np
    ... with pm.Model() as model:
    ...     height = pm.Normal('height', mu=170, sigma=10)
    ...     weight = pm.Normal('weight', mu=70, sigma=15)
    ... '''
    >>> result = execute_pymc_code(code, timeout=60)
    >>> if result.success:
    ...     print(f"Height: {result.height_mu:.1f} ± {result.height_sigma:.1f}")
"""

from dataclasses import dataclass
import json
import re
import subprocess
import tempfile
from pathlib import Path

__all__ = [
    "PyMCExecutionResult",
    "execute_pymc_code",
    "prepare_pymc_code",
    "validate_pymc_code",
]


# Dangerous patterns that should not appear in user code
DANGEROUS_PATTERNS: list[str] = [
    "import os",
    "import sys",
    "import subprocess",
    "from os",
    "from sys",
    "from subprocess",
    "open(",
    "exec(",
    "eval(",
    "__import__",
    "os.system",
    "os.popen",
    "shutil",
    "socket",
    "urllib",
    "requests",
    "http.client",
    "pickle",
    "marshal",
    "compile(",
    "globals(",
    "locals(",
    "getattr(",
    "setattr(",
    "delattr(",
    "breakpoint(",
]


# Code to append for sampling and result extraction
SAMPLING_CODE = """
# ==== Auto-appended sampling code ====
import json as _json
import warnings as _warnings

_warnings.filterwarnings("ignore")

# Sample from posterior
with model:
    _trace = pm.sample(
        draws=500,
        tune=200,
        cores=1,
        progressbar=False,
        return_inferencedata=False,
        random_seed=42,
    )

# Extract summaries
_result = {
    "height_mu": float(_trace["height"].mean()),
    "height_sigma": float(_trace["height"].std()),
    "weight_mu": float(_trace["weight"].mean()),
    "weight_sigma": float(_trace["weight"].std()),
}
print("PYMC_RESULT:" + _json.dumps(_result))
"""


# Required imports to prepend if missing
REQUIRED_IMPORTS = """import pymc as pm
import numpy as np
import arviz as az
"""


@dataclass
class PyMCExecutionResult:
    """Result of PyMC code execution.

    Attributes:
        success: Whether execution completed successfully
        height_mu: Posterior mean for height parameter
        height_sigma: Posterior std for height parameter
        weight_mu: Posterior mean for weight parameter
        weight_sigma: Posterior std for weight parameter
        error: Error message if execution failed
        raw_output: Raw stdout from the subprocess
    """

    success: bool
    height_mu: float | None = None
    height_sigma: float | None = None
    weight_mu: float | None = None
    weight_sigma: float | None = None
    error: str | None = None
    raw_output: str | None = None

    @classmethod
    def from_error(cls, error: str, raw_output: str | None = None) -> "PyMCExecutionResult":
        """Create a failed result from an error message."""
        return cls(success=False, error=error, raw_output=raw_output)

    @classmethod
    def from_posteriors(
        cls,
        height_mu: float,
        height_sigma: float,
        weight_mu: float,
        weight_sigma: float,
        raw_output: str | None = None,
    ) -> "PyMCExecutionResult":
        """Create a successful result from posterior values."""
        return cls(
            success=True,
            height_mu=height_mu,
            height_sigma=height_sigma,
            weight_mu=weight_mu,
            weight_sigma=weight_sigma,
            raw_output=raw_output,
        )


def validate_pymc_code(code: str) -> tuple[bool, str]:
    """Validate PyMC code before execution.

    Performs security and structural validation:
    1. Checks for dangerous imports/patterns that could be exploited
    2. Verifies the code contains a PyMC model definition
    3. Verifies the model defines 'height' and 'weight' variables

    Args:
        code: PyMC code string to validate

    Returns:
        Tuple of (is_valid, error_message).
        If valid, error_message is empty string.

    Examples:
        >>> code = "import pymc as pm; with pm.Model(): height = pm.Normal('height', 0, 1)"
        >>> valid, msg = validate_pymc_code(code)
        >>> # valid is False because 'weight' is missing

        >>> dangerous = "import os; os.system('rm -rf /')"
        >>> valid, msg = validate_pymc_code(dangerous)
        >>> valid
        False
    """
    # Check for dangerous patterns
    code_lower = code.lower()
    for pattern in DANGEROUS_PATTERNS:
        if pattern.lower() in code_lower:
            return False, f"Dangerous pattern detected: '{pattern}'"

    # Check for PyMC model definition
    has_model = bool(re.search(r"pm\.Model\s*\(", code) or re.search(r"pymc\.Model\s*\(", code))
    if not has_model:
        return False, "Code must contain 'pm.Model()' or 'pymc.Model()' context"

    # Check for 'height' variable definition
    has_height = bool(re.search(r"['\"]height['\"]\s*[,)]", code) or re.search(r"height\s*=", code))
    if not has_height:
        return False, "Code must define a 'height' random variable"

    # Check for 'weight' variable definition
    has_weight = bool(re.search(r"['\"]weight['\"]\s*[,)]", code) or re.search(r"weight\s*=", code))
    if not has_weight:
        return False, "Code must define a 'weight' random variable"

    return True, ""


def prepare_pymc_code(raw_code: str) -> str:
    """Prepare PyMC code for execution.

    Adds necessary components if missing:
    1. Import statements (pymc, numpy, arviz)
    2. Sampling code to draw from the posterior
    3. Summary extraction code that outputs JSON

    Args:
        raw_code: PyMC model code (should define a `model` context)

    Returns:
        Modified code ready for subprocess execution

    Example:
        >>> code = '''with pm.Model() as model:
        ...     height = pm.Normal('height', mu=170, sigma=10)
        ...     weight = pm.Normal('weight', mu=70, sigma=15)
        ... '''
        >>> prepared = prepare_pymc_code(code)
        >>> 'pm.sample' in prepared
        True
    """
    lines = []

    # Add imports if missing
    has_pymc_import = "import pymc" in raw_code or "from pymc" in raw_code
    has_numpy_import = "import numpy" in raw_code or "from numpy" in raw_code

    if not has_pymc_import:
        lines.append("import pymc as pm")
    if not has_numpy_import:
        lines.append("import numpy as np")

    if lines:
        lines.append("")  # blank line after imports

    # Add the user's model code
    lines.append(raw_code.strip())

    # Add sampling and extraction code
    lines.append(SAMPLING_CODE)

    return "\n".join(lines)


def _parse_result(output: str) -> dict[str, float] | None:
    """Parse the PYMC_RESULT JSON from subprocess output.

    Args:
        output: Raw stdout from subprocess

    Returns:
        Parsed result dict or None if not found
    """
    marker = "PYMC_RESULT:"
    idx = output.find(marker)
    if idx == -1:
        return None

    json_start = idx + len(marker)
    # Find end of JSON (next newline or end of string)
    json_end = output.find("\n", json_start)
    if json_end == -1:
        json_str = output[json_start:]
    else:
        json_str = output[json_start:json_end]

    try:
        return json.loads(json_str.strip())
    except json.JSONDecodeError:
        return None


def execute_pymc_code(code: str, timeout: int = 60) -> PyMCExecutionResult:
    """Execute PyMC code in a sandboxed subprocess.

    The execution flow:
    1. Validate the code for security and structure
    2. Prepare the code by adding imports and sampling
    3. Write to a temporary file and execute with subprocess
    4. Parse the output to extract posterior summaries

    Args:
        code: PyMC code string (should define a `model` context
            with 'height' and 'weight' random variables)
        timeout: Maximum execution time in seconds (default: 60)

    Returns:
        PyMCExecutionResult with:
        - success=True and posterior means/stds if successful
        - success=False and error message if failed

    Example:
        >>> code = '''
        ... import pymc as pm
        ... with pm.Model() as model:
        ...     height = pm.Normal('height', mu=170, sigma=10)
        ...     weight = pm.Normal('weight', mu=70, sigma=15)
        ... '''
        >>> result = execute_pymc_code(code, timeout=120)
        >>> if result.success:
        ...     print(f"Height: {result.height_mu:.1f}")

    Raises:
        No exceptions are raised; errors are captured in the result.
    """
    # Step 1: Validate
    is_valid, error_msg = validate_pymc_code(code)
    if not is_valid:
        return PyMCExecutionResult.from_error(f"Validation failed: {error_msg}")

    # Step 2: Prepare
    prepared_code = prepare_pymc_code(code)

    # Step 3: Execute in subprocess
    try:
        # Use a temporary file for more reliable execution
        with tempfile.NamedTemporaryFile(mode="w", suffix=".py", delete=False) as temp_file:
            temp_file.write(prepared_code)
            temp_path = temp_file.name

        try:
            result = subprocess.run(
                ["python", temp_path],
                capture_output=True,
                timeout=timeout,
                text=True,
                env=None,  # Use current environment for PyMC access
            )
        finally:
            # Clean up temp file
            Path(temp_path).unlink(missing_ok=True)

    except subprocess.TimeoutExpired:
        return PyMCExecutionResult.from_error(f"Execution timed out after {timeout} seconds")
    except FileNotFoundError:
        return PyMCExecutionResult.from_error("Python interpreter not found")
    except Exception as e:
        return PyMCExecutionResult.from_error(
            f"Subprocess execution failed: {type(e).__name__}: {e}"
        )

    # Combine stdout and stderr for full output
    raw_output = result.stdout
    if result.stderr:
        raw_output += "\n--- stderr ---\n" + result.stderr

    # Check for execution errors
    if result.returncode != 0:
        error_summary = result.stderr[:500] if result.stderr else "Unknown error"
        return PyMCExecutionResult.from_error(
            f"Execution failed (exit code {result.returncode}): {error_summary}",
            raw_output=raw_output,
        )

    # Step 4: Parse results
    parsed = _parse_result(result.stdout)
    if parsed is None:
        return PyMCExecutionResult.from_error(
            "Failed to parse PYMC_RESULT from output",
            raw_output=raw_output,
        )

    # Validate required keys
    required_keys = ["height_mu", "height_sigma", "weight_mu", "weight_sigma"]
    missing_keys = [k for k in required_keys if k not in parsed]
    if missing_keys:
        return PyMCExecutionResult.from_error(
            f"Missing keys in result: {missing_keys}",
            raw_output=raw_output,
        )

    return PyMCExecutionResult.from_posteriors(
        height_mu=parsed["height_mu"],
        height_sigma=parsed["height_sigma"],
        weight_mu=parsed["weight_mu"],
        weight_sigma=parsed["weight_sigma"],
        raw_output=raw_output,
    )
