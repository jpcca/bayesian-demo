"""
Runner script for the height/weight prediction evaluation project using Claude Agent SDK.
"""

import asyncio
import json
import os
from typing import Literal

from claude_agent_sdk import query, ClaudeAgentOptions, AssistantMessage

from models.schemas import PredictionResult, sanitize_nulls
from evaluation.metrics import format_results_table
from experiment_core import ExperimentRunner, load_test_data


class ClaudePredictor:
    """
    Wrapper for Claude Agent SDK predictions.
    Uses the query() function for simple one-shot predictions.
    """

    def __init__(
        self,
        approach: Literal["baseline", "web_search", "probabilistic"],
    ):
        self.approach = approach

        # Load appropriate prompt
        self.system_prompt = self._load_prompt()

        # Configure SDK options based on approach
        if approach in ("web_search", "probabilistic"):
            # Enable WebSearch tool for approaches that need it
            self.options = ClaudeAgentOptions(
                allowed_tools=["WebSearch"],
                system_prompt=self.system_prompt,
            )
        else:
            # Baseline: no web search
            self.options = ClaudeAgentOptions(
                system_prompt=self.system_prompt,
            )

    def _extract_json(self, response_text: str) -> str:
        """
        Extract JSON from response text, handling various formats.
        """
        import re

        text = response_text.strip()

        # Try to find ```json ... ``` block first
        json_block_match = re.search(r"```json\s*([\s\S]*?)```", text)
        if json_block_match:
            return json_block_match.group(1).strip()

        # Try to find any ``` ... ``` block that looks like JSON
        code_blocks = re.findall(r"```\s*([\s\S]*?)```", text)
        for block in code_blocks:
            block = block.strip()
            if block.startswith("{") or block.startswith("["):
                return block

        # No code blocks found, try to extract JSON directly
        # Look for JSON object pattern
        json_match = re.search(r"(\{[\s\S]*\})", text)
        if json_match:
            return json_match.group(1).strip()

        # Return original text and let json.loads handle the error
        return text

    def _load_prompt(self) -> str:
        """Load the appropriate system prompt based on approach."""
        if self.approach == "baseline":
            return """You are a height and weight prediction system.
Given a text description of a person, predict their height (cm) and weight (kg)
as probability distributions (NOT point estimates).

Output JSON format:
{
  "reasoning": "explanation",
  "height_distribution": {"distribution_type": "normal", "mu": 175, "sigma": 6, "unit": "cm"},
  "weight_distribution": {"distribution_type": "normal", "mu": 70, "sigma": 8, "unit": "kg"}
}
"""
        elif self.approach == "web_search":
            # Note: You'll need to implement web search tool integration with Agent SDK
            return """You are a height and weight prediction system with web search capabilities.
Use web search to gather population statistics and domain knowledge.

Output JSON format:
{
  "reasoning": "explanation including search findings",
  "web_searches_performed": ["query1", "query2"],
  "height_distribution": {"distribution_type": "normal", "mu": 175, "sigma": 6, "unit": "cm"},
  "weight_distribution": {"distribution_type": "normal", "mu": 70, "sigma": 8, "unit": "kg"}
}
"""
        else:  # probabilistic
            # Load the full probabilistic prompt using absolute path
            script_dir = os.path.dirname(os.path.abspath(__file__))
            prompt_path = os.path.join(script_dir, "prompts", "probabilistic_agent_prompt.md")
            if os.path.exists(prompt_path):
                with open(prompt_path, "r") as f:
                    return f.read()
            else:
                return """Use Bayesian reasoning and PyMC to create probability distributions.
Include web search and generate PyMC code.
See probabilistic_agent_prompt.md for full instructions."""

    async def predict(self, person_description: str, max_retries: int = 3) -> "PredictionResult":
        """
        Make a prediction for a person based on their description.

        Uses Claude Agent SDK query() function for one-shot predictions.
        """
        for attempt in range(max_retries):
            try:
                # Build user prompt (system prompt is passed via ClaudeAgentOptions)
                user_prompt = f"""USER INPUT:
{person_description}

Please respond with ONLY the JSON object, no additional text."""

                # Call Claude Agent SDK
                response_text = ""
                async for message in query(prompt=user_prompt, options=self.options):
                    # Collect assistant messages
                    if isinstance(message, AssistantMessage):
                        for block in message.content:
                            if hasattr(block, "text"):
                                response_text += block.text

                # Check for empty response
                if not response_text.strip():
                    raise ValueError("Empty response received from Claude")

                # Extract JSON from response, handling markdown code blocks
                json_text = self._extract_json(response_text)

                # Parse and sanitize
                data = json.loads(json_text)
                data = sanitize_nulls(data)

                # Validate and construct
                result = PredictionResult(**data)
                return result

            except Exception as e:
                print(f"Attempt {attempt + 1}/{max_retries} failed for {self.approach}: {e}")
                if attempt == max_retries - 1:
                    # Return invalid result
                    return PredictionResult(
                        reasoning=f"Failed after {max_retries} attempts: {str(e)}",
                        error=str(e),
                    )
                continue


async def main():
    """Main entry point."""
    print("Height/Weight Prediction Evaluation - Claude Runner")
    print("=" * 60)

    # Load test data
    print("Loading test subjects...")
    subjects = load_test_data()
    print(f"Loaded {len(subjects)} subjects")

    # Run experiments using ClaudePredictor
    runner = ExperimentRunner(predictor_class=ClaudePredictor)
    aggregated_metrics = await runner.run_all_experiments(subjects)

    # Save results
    runner.save_results(aggregated_metrics)

    # Print summary table
    print("\n" + "=" * 60)
    print("FINAL RESULTS")
    print("=" * 60 + "\n")
    print(format_results_table(aggregated_metrics))


if __name__ == "__main__":
    # Run with Claude Agent SDK (uses Claude Code authentication)
    asyncio.run(main())
