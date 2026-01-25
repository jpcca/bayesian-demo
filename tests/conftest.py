"""
Pytest configuration and fixtures for bayesian-demo tests.

Provides mock fixtures for Claude Agent SDK to enable unit testing
without making actual API calls.
"""

import pytest
from dataclasses import dataclass
from typing import List, Optional


# Mock Claude Agent SDK types
@dataclass
class MockTextBlock:
    """Mock for claude_agent_sdk.types.TextBlock"""

    text: str


@dataclass
class MockAssistantMessage:
    """Mock for claude_agent_sdk.types.AssistantMessage"""

    content: List[MockTextBlock]
    error: Optional[str] = None
    model: Optional[str] = "claude-sonnet-4-20250514"
    parent_tool_use_id: Optional[str] = None


@dataclass
class MockResultMessage:
    """Mock for claude_agent_sdk.types.ResultMessage"""

    duration_ms: int = 1000
    duration_api_ms: int = 800
    is_error: bool = False
    num_turns: int = 1
    result: Optional[str] = None
    session_id: str = "test-session"
    subtype: str = "success"
    total_cost_usd: float = 0.01
    usage: Optional[dict] = None


@pytest.fixture
def sample_prediction_json():
    """Sample valid prediction JSON response from Claude."""
    return {
        "reasoning": "Based on Norwegian female averages and volleyball background.",
        "web_searches_performed": ["average height Norwegian women"],
        "height_distribution": {
            "distribution_type": "normal",
            "mu": 178.0,
            "sigma": 5.0,
            "unit": "cm",
        },
        "weight_distribution": {
            "distribution_type": "normal",
            "mu": 72.0,
            "sigma": 7.0,
            "unit": "kg",
        },
        "pymc_code": "import pymc as pm\n\nwith pm.Model() as model:\n    height = pm.Normal('height', mu=178, sigma=5)",
    }


@pytest.fixture
def sample_ground_truth():
    """Sample ground truth data for a subject."""
    return {
        "subject_id": "001",
        "text_description": "Sarah is a 32-year-old Norwegian woman who works as a software engineer.",
        "height": {"distribution_type": "normal", "mu": 178.0, "sigma": 4.0, "unit": "cm"},
        "weight": {"distribution_type": "normal", "mu": 72.0, "sigma": 6.0, "unit": "kg"},
    }


@pytest.fixture
def mock_claude_response(sample_prediction_json):
    """
    Create a mock async generator that simulates Claude Agent SDK query() response.
    """
    import json

    async def mock_query(*args, **kwargs):
        # Yield an AssistantMessage with the JSON response
        yield MockAssistantMessage(content=[MockTextBlock(text=json.dumps(sample_prediction_json))])
        # Yield a ResultMessage
        yield MockResultMessage()

    return mock_query


@pytest.fixture
def mock_claude_response_with_markdown(sample_prediction_json):
    """
    Mock response wrapped in markdown code blocks (common LLM output format).
    """
    import json

    async def mock_query(*args, **kwargs):
        wrapped_json = f"```json\n{json.dumps(sample_prediction_json, indent=2)}\n```"
        yield MockAssistantMessage(content=[MockTextBlock(text=wrapped_json)])
        yield MockResultMessage()

    return mock_query


@pytest.fixture
def mock_claude_error_response():
    """
    Mock response that simulates an error/invalid response.
    """

    async def mock_query(*args, **kwargs):
        yield MockAssistantMessage(content=[MockTextBlock(text="I cannot process this request.")])
        yield MockResultMessage(is_error=True, subtype="error")

    return mock_query
