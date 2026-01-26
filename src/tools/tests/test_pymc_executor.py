"""Tests for PyMC executor module.

These tests verify the contract behavior without requiring PyMC installation.
For full integration tests, install the [pymc] extras.
"""

from src.tools.pymc_executor import (
    PyMCExecutionResult,
    prepare_pymc_code,
    validate_pymc_code,
)


class TestPyMCExecutionResult:
    """Tests for PyMCExecutionResult dataclass."""

    def test_from_error_creates_failed_result(self) -> None:
        """Error results should have success=False."""
        result = PyMCExecutionResult.from_error("Test error")
        assert result.success is False
        assert result.error == "Test error"
        assert result.height_mu is None
        assert result.weight_mu is None

    def test_from_error_with_raw_output(self) -> None:
        """Error results can include raw output."""
        result = PyMCExecutionResult.from_error("Test error", raw_output="stderr")
        assert result.raw_output == "stderr"

    def test_from_posteriors_creates_successful_result(self) -> None:
        """Posterior results should have success=True."""
        result = PyMCExecutionResult.from_posteriors(
            height_mu=170.0,
            height_sigma=5.0,
            weight_mu=70.0,
            weight_sigma=8.0,
        )
        assert result.success is True
        assert result.height_mu == 170.0
        assert result.height_sigma == 5.0
        assert result.weight_mu == 70.0
        assert result.weight_sigma == 8.0
        assert result.error is None


class TestValidatePyMCCode:
    """Tests for validate_pymc_code function."""

    def test_valid_code_passes(self) -> None:
        """Valid PyMC code should pass validation."""
        code = """
import pymc as pm
with pm.Model() as model:
    height = pm.Normal('height', mu=170, sigma=10)
    weight = pm.Normal('weight', mu=70, sigma=15)
"""
        is_valid, msg = validate_pymc_code(code)
        assert is_valid is True
        assert msg == ""

    def test_rejects_dangerous_imports(self) -> None:
        """Code with dangerous imports should be rejected."""
        dangerous_codes = [
            "import os\nwith pm.Model(): height = pm.Normal('height', 0, 1); weight = pm.Normal('weight', 0, 1)",
            "import sys\nwith pm.Model(): height = pm.Normal('height', 0, 1); weight = pm.Normal('weight', 0, 1)",
            "import subprocess\nwith pm.Model(): height = pm.Normal('height', 0, 1); weight = pm.Normal('weight', 0, 1)",
            "open('/etc/passwd')\nwith pm.Model(): height = pm.Normal('height', 0, 1); weight = pm.Normal('weight', 0, 1)",
            "eval('1+1')\nwith pm.Model(): height = pm.Normal('height', 0, 1); weight = pm.Normal('weight', 0, 1)",
        ]
        for code in dangerous_codes:
            is_valid, msg = validate_pymc_code(code)
            assert is_valid is False
            assert "Dangerous pattern" in msg

    def test_requires_model_context(self) -> None:
        """Code must contain pm.Model() context."""
        code = "height = pm.Normal('height', 0, 1); weight = pm.Normal('weight', 0, 1)"
        is_valid, msg = validate_pymc_code(code)
        assert is_valid is False
        assert "pm.Model()" in msg

    def test_requires_height_variable(self) -> None:
        """Code must define a 'height' random variable."""
        code = """
with pm.Model() as model:
    weight = pm.Normal('weight', mu=70, sigma=15)
"""
        is_valid, msg = validate_pymc_code(code)
        assert is_valid is False
        assert "height" in msg.lower()

    def test_requires_weight_variable(self) -> None:
        """Code must define a 'weight' random variable."""
        code = """
with pm.Model() as model:
    height = pm.Normal('height', mu=170, sigma=10)
"""
        is_valid, msg = validate_pymc_code(code)
        assert is_valid is False
        assert "weight" in msg.lower()


class TestPreparePyMCCode:
    """Tests for prepare_pymc_code function."""

    def test_adds_imports_when_missing(self) -> None:
        """Should add imports if not present."""
        code = """
with pm.Model() as model:
    height = pm.Normal('height', mu=170, sigma=10)
    weight = pm.Normal('weight', mu=70, sigma=15)
"""
        prepared = prepare_pymc_code(code)
        assert "import pymc as pm" in prepared
        assert "import numpy as np" in prepared

    def test_preserves_existing_imports(self) -> None:
        """Should not duplicate existing imports."""
        code = """
import pymc as pm
import numpy as np
with pm.Model() as model:
    height = pm.Normal('height', mu=170, sigma=10)
    weight = pm.Normal('weight', mu=70, sigma=15)
"""
        prepared = prepare_pymc_code(code)
        # Should not have duplicate imports
        assert prepared.count("import pymc") == 1
        assert prepared.count("import numpy") == 1

    def test_adds_sampling_code(self) -> None:
        """Should add sampling and result extraction code."""
        code = """
with pm.Model() as model:
    height = pm.Normal('height', mu=170, sigma=10)
    weight = pm.Normal('weight', mu=70, sigma=15)
"""
        prepared = prepare_pymc_code(code)
        assert "pm.sample" in prepared
        assert "PYMC_RESULT:" in prepared
        assert "json.dumps" in prepared or "_json.dumps" in prepared

    def test_includes_original_model(self) -> None:
        """Should include the original model code."""
        code = """
with pm.Model() as model:
    height = pm.Normal('height', mu=170, sigma=10)
    weight = pm.Normal('weight', mu=70, sigma=15)
"""
        prepared = prepare_pymc_code(code)
        assert "pm.Normal('height'" in prepared
        assert "pm.Normal('weight'" in prepared
