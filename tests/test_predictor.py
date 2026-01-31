"""
Unit tests for example_runner.py ClaudePredictor class.

Tests the predictor with mocked Claude Agent SDK to avoid API calls.
"""

import pytest
from unittest.mock import patch, MagicMock
import json

import sys

sys.path.insert(0, "src")

from models.schemas import PredictionResult


class TestClaudePredictorInit:
    """Tests for ClaudePredictor initialization."""

    def test_baseline_approach_no_web_search(self):
        """Baseline approach should not enable WebSearch."""
        with patch("example_runner.ClaudeAgentOptions") as MockOptions:
            from example_runner import ClaudePredictor

            ClaudePredictor(approach="baseline")  # Instantiate to trigger __init__

            # Check that WebSearch is NOT in allowed_tools
            call_kwargs = MockOptions.call_args[1]
            assert "allowed_tools" not in call_kwargs or call_kwargs.get("allowed_tools") is None

    def test_web_search_approach_enables_web_search(self):
        """Web search approach should enable WebSearch tool."""
        with patch("example_runner.ClaudeAgentOptions") as MockOptions:
            from example_runner import ClaudePredictor

            ClaudePredictor(approach="web_search")  # Instantiate to trigger __init__

            call_kwargs = MockOptions.call_args[1]
            assert "allowed_tools" in call_kwargs
            assert "WebSearch" in call_kwargs["allowed_tools"]

    def test_probabilistic_approach_enables_web_search(self):
        """Probabilistic approach should enable WebSearch tool."""
        with patch("example_runner.ClaudeAgentOptions") as MockOptions:
            from example_runner import ClaudePredictor

            ClaudePredictor(approach="probabilistic")  # Instantiate to trigger __init__

            call_kwargs = MockOptions.call_args[1]
            assert "allowed_tools" in call_kwargs
            assert "WebSearch" in call_kwargs["allowed_tools"]

    def test_system_prompt_passed_to_options(self):
        """System prompt should be passed to ClaudeAgentOptions."""
        with patch("example_runner.ClaudeAgentOptions") as MockOptions:
            from example_runner import ClaudePredictor

            ClaudePredictor(approach="baseline")  # Instantiate to trigger __init__

            call_kwargs = MockOptions.call_args[1]
            assert "system_prompt" in call_kwargs
            assert len(call_kwargs["system_prompt"]) > 0


class TestClaudePredictorPredict:
    """Tests for ClaudePredictor.predict() method."""

    @pytest.mark.asyncio
    async def test_successful_prediction(self, sample_prediction_json):
        """Test successful prediction parsing."""
        # Create mock that matches AssistantMessage check
        from example_runner import AssistantMessage

        class MockTextBlock:
            def __init__(self, text):
                self.text = text

        class MockAssistantMsg(AssistantMessage):
            def __init__(self, content):
                self.content = content

        async def mock_query(*args, **kwargs):
            # Create a mock that passes isinstance check
            msg = MagicMock(spec=AssistantMessage)
            msg.content = [MockTextBlock(json.dumps(sample_prediction_json))]
            yield msg

        with patch("example_runner.query", mock_query):
            with patch("example_runner.ClaudeAgentOptions"):
                from example_runner import ClaudePredictor

                predictor = ClaudePredictor(approach="baseline")
                result = await predictor.predict("Test person description")

                assert isinstance(result, PredictionResult)
                assert result.is_valid is True
                assert result.height_distribution is not None
                assert result.weight_distribution is not None

    @pytest.mark.asyncio
    async def test_prediction_with_markdown_wrapper(self, sample_prediction_json):
        """Test parsing JSON wrapped in markdown code blocks."""
        from example_runner import AssistantMessage

        class MockTextBlock:
            def __init__(self, text):
                self.text = text

        wrapped_json = f"```json\n{json.dumps(sample_prediction_json, indent=2)}\n```"

        async def mock_query(*args, **kwargs):
            msg = MagicMock(spec=AssistantMessage)
            msg.content = [MockTextBlock(wrapped_json)]
            yield msg

        with patch("example_runner.query", mock_query):
            with patch("example_runner.ClaudeAgentOptions"):
                from example_runner import ClaudePredictor

                predictor = ClaudePredictor(approach="baseline")
                result = await predictor.predict("Test person description")

                assert isinstance(result, PredictionResult)
                assert result.is_valid is True

    @pytest.mark.asyncio
    async def test_prediction_error_handling(self):
        """Test handling of invalid responses."""
        from example_runner import AssistantMessage

        class MockTextBlock:
            def __init__(self, text):
                self.text = text

        async def mock_query(*args, **kwargs):
            msg = MagicMock(spec=AssistantMessage)
            msg.content = [MockTextBlock("I cannot process this request.")]
            yield msg

        with patch("example_runner.query", mock_query):
            with patch("example_runner.ClaudeAgentOptions"):
                from example_runner import ClaudePredictor

                predictor = ClaudePredictor(approach="baseline")
                result = await predictor.predict("Test person description")

                assert isinstance(result, PredictionResult)
                assert result.is_valid is False
                assert result.error is not None

    @pytest.mark.asyncio
    async def test_retry_on_failure(self):
        """Test that predictor retries on failure."""
        from example_runner import AssistantMessage

        class MockTextBlock:
            def __init__(self, text):
                self.text = text

        call_count = 0

        async def failing_then_success(*args, **kwargs):
            nonlocal call_count
            call_count += 1
            if call_count < 3:
                raise Exception("Temporary failure")
            # Success on third try
            msg = MagicMock(spec=AssistantMessage)
            msg.content = [
                MockTextBlock(
                    '{"reasoning": "test", "height_distribution": {"distribution_type": "normal", "mu": 175, "sigma": 6, "unit": "cm"}, "weight_distribution": {"distribution_type": "normal", "mu": 70, "sigma": 8, "unit": "kg"}}'
                )
            ]
            yield msg

        with patch("example_runner.query", failing_then_success):
            with patch("example_runner.ClaudeAgentOptions"):
                from example_runner import ClaudePredictor

                predictor = ClaudePredictor(approach="baseline")
                result = await predictor.predict("Test person description", max_retries=3)

                assert call_count == 3
                assert result.is_valid is True

    @pytest.mark.asyncio
    async def test_max_retries_exceeded(self):
        """Test that predictor gives up after max retries."""

        async def always_fail(*args, **kwargs):
            raise Exception("Persistent failure")
            yield  # Make it a generator

        with patch("example_runner.query", always_fail):
            with patch("example_runner.ClaudeAgentOptions"):
                from example_runner import ClaudePredictor

                predictor = ClaudePredictor(approach="baseline")
                result = await predictor.predict("Test person description", max_retries=2)

                assert result.is_valid is False
                assert "Failed after 2 attempts" in result.reasoning


class TestClaudePredictorPrompts:
    """Tests for prompt loading."""

    def test_baseline_prompt_content(self):
        """Test baseline prompt contains key instructions."""
        with patch("example_runner.ClaudeAgentOptions"):
            from example_runner import ClaudePredictor

            predictor = ClaudePredictor(approach="baseline")

            assert "height" in predictor.system_prompt.lower()
            assert "weight" in predictor.system_prompt.lower()
            assert "json" in predictor.system_prompt.lower()

    def test_web_search_prompt_mentions_search(self):
        """Test web_search prompt mentions search capabilities."""
        with patch("example_runner.ClaudeAgentOptions"):
            from example_runner import ClaudePredictor

            predictor = ClaudePredictor(approach="web_search")

            assert "search" in predictor.system_prompt.lower()

    def test_probabilistic_prompt_mentions_bayesian(self):
        """Test probabilistic prompt contains Bayesian concepts."""
        with patch("example_runner.ClaudeAgentOptions"):
            with patch("os.path.exists", return_value=True):
                with patch(
                    "builtins.open",
                    MagicMock(
                        return_value=MagicMock(
                            __enter__=lambda s: s,
                            __exit__=lambda s, *args: None,
                            read=lambda: "Bayesian reasoning PyMC probabilistic",
                        )
                    ),
                ):
                    from example_runner import ClaudePredictor

                    predictor = ClaudePredictor(approach="probabilistic")

                    # Either loads from file or uses fallback
                    prompt_lower = predictor.system_prompt.lower()
                    assert (
                        "bayesian" in prompt_lower
                        or "probabilistic" in prompt_lower
                        or "pymc" in prompt_lower
                    )
