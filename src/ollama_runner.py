"""
Runner script for the height/weight prediction evaluation project using Ollama (local LLM).
"""

import asyncio
import json
import os
from typing import Literal

try:
    import ollama
except ImportError:
    ollama = None

from models.schemas import PredictionResult, sanitize_nulls
from evaluation.metrics import format_results_table
from experiment_core import ExperimentRunner, load_test_data


class OllamaPredictor:
    """
    Predictor using local Ollama model (e.g., phi4-mini).
    """

    def __init__(
        self,
        approach: Literal["baseline", "web_search", "probabilistic"],
        model_name: str = "phi4-mini:latest",
    ):
        self.approach = approach
        self.model_name = model_name
        
        if ollama is None:
            raise ImportError("ollama library not installed. Please install it with `pip install ollama`.")

        self.system_prompt = self._load_prompt()

    def _extract_json(self, response_text: str) -> str:
        """
        Extract JSON from response text.
        Reused logic from ClaudePredictor.
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

        return text

    def _load_prompt(self) -> str:
        """Load the appropriate system prompt based on approach."""
        # For now, reusing the same prompts as Claude, but might need tuning for smaller models
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
            # NOTE: Web search not implemented for Ollama in this iteration
            # Returning baseline prompt but expecting web search structure if needed
             return """You are a height and weight prediction system.
Output JSON format:
{
  "reasoning": "explanation",
  "web_searches_performed": [],
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
                return "System prompt not found."

    async def predict(self, person_description: str, max_retries: int = 3) -> "PredictionResult":
        """
        Make a prediction using Ollama.
        """
        for attempt in range(max_retries):
            try:
                user_prompt = f"""USER INPUT:
{person_description}

Please respond with ONLY the JSON object, no additional text."""

                messages = [
                    {'role': 'system', 'content': self.system_prompt},
                    {'role': 'user', 'content': user_prompt},
                ]

                # Use ollama.chat directly (synchronous call, but wrapping in async function)
                response = ollama.chat(model=self.model_name, messages=messages)
                response_text = response['message']['content']

                if not response_text.strip():
                     raise ValueError("Empty response received from Ollama")

                json_text = self._extract_json(response_text)
                
                # Parse and sanitize
                data = json.loads(json_text)
                data = sanitize_nulls(data)

                # Validate and construct
                result = PredictionResult(**data)
                return result

            except Exception as e:
                print(f"Attempt {attempt + 1}/{max_retries} failed for {self.approach} (Ollama): {e}")
                if attempt == max_retries - 1:
                    return PredictionResult(
                        reasoning=f"Failed after {max_retries} attempts: {str(e)}",
                        error=str(e),
                    )
                continue


async def main():
    """Main entry point."""
    print("Height/Weight Prediction Evaluation - Ollama Runner")
    print("=" * 60)

    # Load test data
    print("Loading test subjects...")
    subjects = load_test_data()
    print(f"Loaded {len(subjects)} subjects")

    # Run experiments using OllamaPredictor
    runner = ExperimentRunner(predictor_class=OllamaPredictor, model_name="phi4-mini:latest")
    aggregated_metrics = await runner.run_all_experiments(subjects)

    # Save results
    runner.save_results(aggregated_metrics)

    # Print summary table
    print("\n" + "=" * 60)
    print("FINAL RESULTS")
    print("=" * 60 + "\n")
    print(format_results_table(aggregated_metrics))


if __name__ == "__main__":
    asyncio.run(main())
