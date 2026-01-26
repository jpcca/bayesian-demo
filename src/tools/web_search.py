"""Web search tool integration for Claude Agent SDK.

This module provides web search capability that Claude can invoke during predictions.
Uses DuckDuckGo search (no API key required) or falls back to simulated results.

Basic Usage:
    >>> from tools.web_search import web_search, get_web_search_tool_definition
    >>> tool_def = get_web_search_tool_definition()
    >>> results = web_search("average human height")
"""

import json
import time
from typing import TYPE_CHECKING, Any

# Try to import DuckDuckGo search library
HAS_DDGS = False
try:
    from duckduckgo_search import DDGS  # type: ignore[import-not-found]

    HAS_DDGS = True
except ImportError:
    pass

if TYPE_CHECKING:
    from duckduckgo_search import DDGS  # type: ignore[import-not-found]  # noqa: F811

# Rate limiting: track search calls per prediction
_search_count = 0
_last_reset_time = time.time()
MAX_SEARCHES_PER_PREDICTION = 3
RESET_INTERVAL_SECONDS = 60  # Reset counter every minute

# Simulated data for fallback when DDGS is not available
SIMULATED_DATA: dict[str, dict[str, Any]] = {
    "average height": {
        "results": [
            {
                "title": "Average Human Height by Country",
                "snippet": (
                    "Global average adult height: Men 171cm, Women 159cm. "
                    "Varies by region: Netherlands (men 183cm), Guatemala (men 163cm)."
                ),
                "url": "https://example.com/height-stats",
            },
            {
                "title": "Human Height - Wikipedia",
                "snippet": (
                    "The average height of adults varies significantly across populations. "
                    "Genetics, nutrition, and health factors all contribute."
                ),
                "url": "https://en.wikipedia.org/wiki/Human_height",
            },
        ]
    },
    "average weight": {
        "results": [
            {
                "title": "Average Body Weight by Country",
                "snippet": (
                    "Global average adult weight: Men 70kg, Women 60kg. "
                    "Varies significantly by region and lifestyle factors."
                ),
                "url": "https://example.com/weight-stats",
            },
        ]
    },
    "population statistics": {
        "results": [
            {
                "title": "World Population Data",
                "snippet": (
                    "Current world population: ~8 billion. "
                    "Annual growth rate approximately 0.9%. Median age 30.4 years."
                ),
                "url": "https://example.com/population-data",
            },
        ]
    },
    "demographic data": {
        "results": [
            {
                "title": "Global Demographics Overview",
                "snippet": (
                    "Key demographic indicators: life expectancy 72.6 years globally, "
                    "fertility rate 2.3 children per woman, urbanization rate 56%."
                ),
                "url": "https://example.com/demographics",
            },
        ]
    },
}

# Default fallback when no pattern matches
DEFAULT_SIMULATED_RESULT: dict[str, Any] = {
    "results": [
        {
            "title": "Search Results",
            "snippet": (
                "No specific data available for this query. "
                "Consider using more specific search terms."
            ),
            "url": "https://example.com/search",
        },
    ]
}


def _reset_rate_limit_if_needed() -> None:
    """Reset the rate limit counter if the interval has passed."""
    global _search_count, _last_reset_time
    current_time = time.time()
    if current_time - _last_reset_time >= RESET_INTERVAL_SECONDS:
        _search_count = 0
        _last_reset_time = current_time


def _check_rate_limit() -> bool:
    """Check if we're within rate limits.

    Returns:
        True if search is allowed, False if rate limited.
    """
    global _search_count
    _reset_rate_limit_if_needed()
    return _search_count < MAX_SEARCHES_PER_PREDICTION


def _increment_search_count() -> None:
    """Increment the search counter."""
    global _search_count
    _search_count += 1


def _search_with_ddgs(query: str, max_results: int = 5) -> list[dict[str, str]]:
    """Execute search using DuckDuckGo.

    Args:
        query: Search query string.
        max_results: Maximum number of results to return.

    Returns:
        List of search results with title, snippet, and url.

    Raises:
        RuntimeError: If DDGS is not available.
    """
    if not HAS_DDGS:
        raise RuntimeError("DuckDuckGo search is not available")

    results = []
    with DDGS() as ddgs:
        for r in ddgs.text(query, max_results=max_results):
            results.append(
                {
                    "title": r.get("title", ""),
                    "snippet": r.get("body", ""),
                    "url": r.get("href", ""),
                }
            )
    return results


def _search_simulated(query: str) -> list[dict[str, str]]:
    """Return simulated search results based on query patterns.

    Args:
        query: Search query string.

    Returns:
        List of simulated search results.
    """
    query_lower = query.lower()

    # Check for matching patterns in simulated data
    for pattern, data in SIMULATED_DATA.items():
        if pattern in query_lower:
            return data["results"]

    # Return default result if no pattern matches
    return DEFAULT_SIMULATED_RESULT["results"]


def web_search(query: str) -> dict[str, Any]:
    """Execute a web search and return results.

    Uses DuckDuckGo search (no API key required) or falls back to simulated results.

    Args:
        query: Search query string.

    Returns:
        Dictionary containing:
            - query: Original query string
            - results: List of result dicts with title, snippet, url
            - source: "duckduckgo" | "simulated" | "rate_limited"

    Examples:
        >>> result = web_search("average human height")
        >>> assert "results" in result
        >>> assert result["source"] in ["duckduckgo", "simulated", "rate_limited"]
    """
    # Check rate limiting
    if not _check_rate_limit():
        return {
            "query": query,
            "results": [
                {
                    "title": "Rate Limited",
                    "snippet": (
                        f"Search rate limit exceeded ({MAX_SEARCHES_PER_PREDICTION} searches "
                        "per prediction). Please wait before searching again."
                    ),
                    "url": "",
                }
            ],
            "source": "rate_limited",
        }

    _increment_search_count()

    # Try DuckDuckGo first if available
    if HAS_DDGS:
        try:
            results = _search_with_ddgs(query)
            return {
                "query": query,
                "results": results,
                "source": "duckduckgo",
            }
        except Exception:
            # Fall through to simulated on any error
            pass

    # Fall back to simulated results
    results = _search_simulated(query)
    return {
        "query": query,
        "results": results,
        "source": "simulated",
    }


def get_web_search_tool_definition() -> dict[str, Any]:
    """Return the tool definition for Claude Agent SDK.

    This definition follows the format expected by Claude's tool use feature.

    Returns:
        Tool definition dictionary with name, description, and input_schema.

    Examples:
        >>> tool_def = get_web_search_tool_definition()
        >>> assert tool_def["name"] == "web_search"
        >>> assert "input_schema" in tool_def
    """
    return {
        "name": "web_search",
        "description": (
            "Search the web for information. Use this to find population statistics, "
            "demographic data, and domain knowledge."
        ),
        "input_schema": {
            "type": "object",
            "properties": {
                "query": {
                    "type": "string",
                    "description": "The search query",
                }
            },
            "required": ["query"],
        },
    }


def handle_tool_call(tool_name: str, tool_input: dict[str, Any]) -> str:
    """Handle a tool call from Claude.

    Args:
        tool_name: Name of the tool being called.
        tool_input: Arguments passed to the tool.

    Returns:
        JSON string with tool results.

    Raises:
        ValueError: If tool_name is not recognized.

    Examples:
        >>> result = handle_tool_call("web_search", {"query": "test"})
        >>> data = json.loads(result)
        >>> assert "results" in data
    """
    if tool_name != "web_search":
        error_result = {
            "error": f"Unknown tool: {tool_name}",
            "available_tools": ["web_search"],
        }
        return json.dumps(error_result)

    query = tool_input.get("query", "")
    if not query:
        error_result = {
            "error": "Missing required parameter: query",
        }
        return json.dumps(error_result)

    result = web_search(query)
    return json.dumps(result)


def reset_rate_limit() -> None:
    """Reset the rate limit counter.

    Useful for testing or starting a new prediction session.
    """
    global _search_count, _last_reset_time
    _search_count = 0
    _last_reset_time = time.time()
