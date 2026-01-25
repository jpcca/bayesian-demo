"""
Example runner script for the height/weight prediction evaluation project.

This demonstrates how to use the extracted components from the transcribe project
to run experiments with the three approaches:
1. Baseline: Claude Agent SDK with no web search
2. Web Search: Claude Agent SDK with web search tool
3. Probabilistic: Claude Agent SDK with web search + PyMC prompting

Adapted from the transcribe project's api/main.py WebSocket loop.
"""

import asyncio
import json
import os
import subprocess
from typing import List, Literal

from claude_agent_sdk import query, ClaudeAgentOptions, AssistantMessage, ResultMessage

from models.schemas import (
    PredictionResult,
    GroundTruth,
    ExperimentResult,
    AggregatedMetrics,
    sanitize_nulls,
)
from evaluation.metrics import evaluate_prediction, aggregate_results, format_results_table


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
            # Load the full probabilistic prompt
            prompt_path = "prompts/probabilistic_agent_prompt.md"
            if os.path.exists(prompt_path):
                with open(prompt_path, "r") as f:
                    return f.read()
            else:
                return """Use Bayesian reasoning and PyMC to create probability distributions.
Include web search and generate PyMC code.
See probabilistic_agent_prompt.md for full instructions."""

    async def predict(
        self, person_description: str, max_retries: int = 3
    ) -> "PredictionResult":
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

                # Try to parse as JSON
                # Handle markdown code blocks if present
                if "```json" in response_text:
                    response_text = response_text.split("```json")[1].split("```")[0]
                elif "```" in response_text:
                    response_text = response_text.split("```")[1].split("```")[0]

                # Parse and sanitize
                data = json.loads(response_text.strip())
                data = sanitize_nulls(data)

                # Validate and construct
                result = PredictionResult(**data)
                return result

            except Exception as e:
                print(
                    f"Attempt {attempt + 1}/{max_retries} failed for {self.approach}: {e}"
                )
                if attempt == max_retries - 1:
                    # Return invalid result
                    return PredictionResult(
                        reasoning=f"Failed after {max_retries} attempts: {str(e)}",
                        error=str(e),
                    )
                continue


class ExperimentRunner:
    """
    Runs the full experiment across all subjects and approaches.

    Adapts the pattern from REUSABLE_COMPONENTS.md section 6.
    """

    def __init__(self):
        self.results: List["ExperimentResult"] = []
        # Get project root directory
        self.project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
        self.results_dir = os.path.join(self.project_root, "results")

    async def run_single_experiment(
        self,
        approach: Literal["baseline", "web_search", "probabilistic"],
        subjects: List["GroundTruth"],
    ) -> List["ExperimentResult"]:
        """
        Run experiment for one approach across all subjects.

        Args:
            approach: Which approach to test
            subjects: List of GroundTruth objects with descriptions

        Returns:
            List of ExperimentResult objects
        """
        predictor = ClaudePredictor(approach=approach)
        results = []

        for i, subject in enumerate(subjects):
            print(f"[{approach}] Processing subject {i+1}/{len(subjects)}...")

            # Make prediction
            prediction = await predictor.predict(subject.text_description)

            # Evaluate
            metrics = evaluate_prediction(prediction, subject)

            # Store result
            result = ExperimentResult(
                subject_id=subject.subject_id,
                approach=approach,
                prediction=prediction,
                ground_truth=subject,
                metrics=metrics,
            )
            results.append(result)

            # Save intermediate results (in case of crash)
            self._save_intermediate(result)

        return results

    def _save_intermediate(self, result: "ExperimentResult"):
        """Save intermediate results to avoid data loss."""
        intermediate_dir = os.path.join(self.results_dir, "intermediate")
        os.makedirs(intermediate_dir, exist_ok=True)
        filename = os.path.join(intermediate_dir, f"{result.approach}_{result.subject_id}.json")
        with open(filename, "w") as f:
            json.dump(result.model_dump(), f, indent=2)

    async def run_all_experiments(
        self, subjects: List["GroundTruth"]
    ) -> List["AggregatedMetrics"]:
        """
        Run all three approaches and aggregate results.

        Args:
            subjects: List of 50 GroundTruth objects

        Returns:
            List of AggregatedMetrics for each approach
        """
        approaches = ["baseline", "web_search", "probabilistic"]
        all_aggregated = []

        for approach in approaches:
            print(f"\n{'='*60}")
            print(f"Running experiment: {approach}")
            print(f"{'='*60}\n")

            results = await self.run_single_experiment(approach, subjects)
            self.results.extend(results)

            # Aggregate
            aggregated = aggregate_results(results)
            all_aggregated.append(aggregated)

            print(f"\n{approach} completed:")
            print(f"  Valid: {aggregated.n_valid}/{aggregated.n_total}")
            print(f"  Invalid rate: {aggregated.invalid_rate_percent:.1f}%")
            if aggregated.n_valid > 0:
                print(
                    f"  Mean KL (height): {aggregated.mean_kl_divergence_height:.3f}"
                )
                print(
                    f"  Mean KL (weight): {aggregated.mean_kl_divergence_weight:.3f}"
                )

        return all_aggregated

    def save_results(self, aggregated_metrics: List["AggregatedMetrics"]):
        """Save final results to CSV and markdown table."""
        import pandas as pd

        os.makedirs(self.results_dir, exist_ok=True)

        # Save as markdown table
        table = format_results_table(aggregated_metrics)
        with open(os.path.join(self.results_dir, "experiment_results.md"), "w") as f:
            f.write("# Experiment Results\n\n")
            f.write(table)
            f.write("\n\n## Metric Descriptions\n\n")
            f.write("- **KL Divergence**: Lower is better (0 = perfect match)\n")
            f.write("- **Wasserstein Distance**: L2 distance between distributions\n")
            f.write("- **MAE**: Mean Absolute Error on distribution mean (mu)\n")
            f.write("- **Invalid Rate**: % of outputs that failed to produce valid distributions\n")

        # Save as CSV
        df = pd.DataFrame([m.model_dump() for m in aggregated_metrics])
        df.to_csv(os.path.join(self.results_dir, "experiment_results.csv"), index=False)

        # Save detailed results
        with open(os.path.join(self.results_dir, "all_results.json"), "w") as f:
            json.dump([r.model_dump() for r in self.results], f, indent=2)

        print(f"\n✓ Results saved to {self.results_dir}/")


def load_test_data() -> List["GroundTruth"]:
    """
    Load test subjects and ground truth.

    You'll need to create this data file with 50 subjects.
    Format:
    [
      {
        "subject_id": "001",
        "text_description": "John is a 28-year-old...",
        "height": {"distribution_type": "normal", "mu": 175, "sigma": 6, "unit": "cm"},
        "weight": {"distribution_type": "normal", "mu": 72, "sigma": 8, "unit": "kg"}
      },
      ...
    ]
    """
    # Get the project root directory (parent of src/)
    project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    data_path = os.path.join(project_root, "data", "subjects.json")

    with open(data_path, "r") as f:
        data = json.load(f)

    return [GroundTruth(**item) for item in data]


async def main():
    """Main entry point."""
    print("Height/Weight Prediction Evaluation")
    print("=" * 60)

    # Load test data
    print("Loading test subjects...")
    subjects = load_test_data()
    print(f"Loaded {len(subjects)} subjects")

    # Run experiments
    runner = ExperimentRunner()
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
