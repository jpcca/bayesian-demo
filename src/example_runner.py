"""Example runner script for the height/weight prediction calibration study.

This demonstrates how to use the extracted components from the transcribe project
to run experiments with the three approaches:
1. Baseline: Claude Agent SDK with no web search
2. Web Search: Claude Agent SDK with web search tool
3. Probabilistic: Claude Agent SDK with web search + PyMC prompting

Updated for calibration-focused evaluation using NHANES real measurement data.
"""

import asyncio
import json
import os
from typing import Dict, List, Literal, Optional, Tuple

from claude_agent_sdk import (
    AssistantMessage,
    ClaudeAgentOptions,
    ToolUseBlock,
    query,
)

from data.nhanes_loader import Subject, load_subjects
from evaluation.calibration import (
    aggregate_calibration_results,
    evaluate_calibration,
)
from evaluation.statistical_tests import (
    FriedmanResult,
    WilcoxonResult,
    all_pairwise_comparisons,
    format_statistical_report,
    friedman_test,
)
from models.schemas import (
    AggregatedCalibrationMetrics,
    ExperimentResult,
    GroundTruth,
    PredictionResult,
    sanitize_nulls,
)
from tools.pymc_executor import execute_pymc_code
from tools.web_search import (
    get_web_search_tool_definition,
    handle_tool_call,
    reset_rate_limit,
)


class ClaudePredictor:
    """
    Wrapper for Claude Agent SDK predictions.
    Uses the query() function with optional tool support.
    """

    def __init__(
        self,
        approach: Literal["baseline", "web_search", "probabilistic"],
    ):
        self.approach = approach

        # Load appropriate prompt
        self.system_prompt = self._load_prompt()

        # Configure SDK options
        self.options = ClaudeAgentOptions()

        # Configure tools for web_search and probabilistic approaches
        if approach in ["web_search", "probabilistic"]:
            self.tools = [get_web_search_tool_definition()]
        else:
            self.tools = []

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
            return """You are a height and weight prediction system with web search capabilities.
Use the web_search tool to gather population statistics and domain knowledge
before making predictions.

STEPS:
1. Search for relevant demographic/anthropometric data
2. Use findings to inform your prediction
3. Provide probability distributions (NOT point estimates)

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
                return """You are a Bayesian prediction system. Use web search to gather data,
then generate PyMC code to create informed probability distributions.

STEPS:
1. Search for relevant demographic/anthropometric data
2. Build a Bayesian model using PyMC
3. Provide both the PyMC code and initial distribution estimates

Output JSON format:
{
  "reasoning": "explanation of Bayesian approach",
  "web_searches_performed": ["query1", "query2"],
  "height_distribution": {"distribution_type": "normal", "mu": 175, "sigma": 6, "unit": "cm"},
  "weight_distribution": {"distribution_type": "normal", "mu": 70, "sigma": 8, "unit": "kg"},
  "pymc_code": "import pymc as pm\nwith pm.Model() as model:\n    height = pm.Normal('height', mu=175, sigma=6)\n    weight = pm.Normal('weight', mu=70, sigma=8)"
}"""

    async def predict(self, person_description: str, max_retries: int = 3) -> PredictionResult:
        """
        Make a prediction for a person based on their description.

        Uses Claude Agent SDK query() function with optional tool support.
        """
        # Reset rate limit for each prediction
        reset_rate_limit()

        for attempt in range(max_retries):
            try:
                # Build full prompt with system instructions
                full_prompt = f"""{self.system_prompt}

USER INPUT:
{person_description}

Please respond with ONLY the JSON object, no additional text."""

                # Prepare options with tools if applicable
                options = self.options
                if self.tools:
                    # Note: Tool handling depends on Claude Agent SDK implementation
                    # This is a simplified approach
                    pass

                # Call Claude Agent SDK with tool handling
                response_text = ""
                searches_performed = []

                async for message in query(prompt=full_prompt, options=options):
                    # Handle tool calls
                    if isinstance(message, AssistantMessage):
                        for block in message.content:
                            if hasattr(block, "text"):
                                response_text += block.text
                            elif isinstance(block, ToolUseBlock):
                                # Handle tool call
                                if block.name == "web_search":
                                    _tool_result = handle_tool_call(block.name, block.input)
                                    searches_performed.append(block.input.get("query", ""))
                                    # The SDK handles tool result; _tool_result kept for debugging

                # Try to parse as JSON
                # Handle markdown code blocks if present
                if "```json" in response_text:
                    response_text = response_text.split("```json")[1].split("```")[0]
                elif "```" in response_text:
                    response_text = response_text.split("```")[1].split("```")[0]

                # Parse and sanitize
                data = json.loads(response_text.strip())
                data = sanitize_nulls(data)

                # Add searches if not already present
                if searches_performed and "web_searches_performed" not in data:
                    data["web_searches_performed"] = searches_performed

                # Validate and construct
                result = PredictionResult(**data)

                # For probabilistic approach, execute PyMC code if present
                if self.approach == "probabilistic" and result.pymc_code:
                    pymc_result = execute_pymc_code(result.pymc_code, timeout=120)
                    if pymc_result.success:
                        # Update distribution parameters with posterior summaries
                        if (
                            pymc_result.height_mu is not None
                            and pymc_result.height_sigma is not None
                            and result.height_distribution is not None
                        ):
                            result.height_distribution.mu = pymc_result.height_mu
                            result.height_distribution.sigma = pymc_result.height_sigma
                        if (
                            pymc_result.weight_mu is not None
                            and pymc_result.weight_sigma is not None
                            and result.weight_distribution is not None
                        ):
                            result.weight_distribution.mu = pymc_result.weight_mu
                            result.weight_distribution.sigma = pymc_result.weight_sigma
                        print(
                            f"    [PyMC] Updated with posteriors: "
                            f"height={pymc_result.height_mu:.1f}±{pymc_result.height_sigma:.1f}, "
                            f"weight={pymc_result.weight_mu:.1f}±{pymc_result.weight_sigma:.1f}"
                        )
                    else:
                        print(f"    [PyMC] Execution failed: {pymc_result.error}")

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

        # Should not reach here, but return error just in case
        return PredictionResult(
            reasoning="Unexpected error",
            error="Unexpected control flow",
        )


class ExperimentRunner:
    """
    Runs the full experiment across all subjects and approaches.

    Updated for calibration-focused evaluation using NHANES data.
    """

    def __init__(self):
        self.results: List[ExperimentResult] = []
        # Get project root directory
        self.project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
        self.results_dir = os.path.join(self.project_root, "results")

    async def run_single_experiment(
        self,
        approach: Literal["baseline", "web_search", "probabilistic"],
        subjects: List[Subject],
    ) -> List[ExperimentResult]:
        """
        Run experiment for one approach across all subjects.

        Args:
            approach: Which approach to test
            subjects: List of Subject objects with actual measurements

        Returns:
            List of ExperimentResult objects with calibration metrics
        """
        predictor = ClaudePredictor(approach=approach)
        results = []

        for i, subject in enumerate(subjects):
            print(f"[{approach}] Processing subject {i + 1}/{len(subjects)}...")

            # Make prediction
            prediction = await predictor.predict(subject.text_description)

            # Evaluate calibration
            calibration = evaluate_calibration(prediction, subject)

            # Store result
            result = ExperimentResult(
                subject_id=subject.subject_id,
                approach=approach,
                prediction=prediction,
                subject=subject,
                calibration_metrics=calibration,
            )
            results.append(result)

            # Save intermediate results (in case of crash)
            self._save_intermediate(result)

        return results

    def _save_intermediate(self, result: ExperimentResult):
        """Save intermediate results to avoid data loss."""
        intermediate_dir = os.path.join(self.results_dir, "intermediate")
        os.makedirs(intermediate_dir, exist_ok=True)
        filename = os.path.join(intermediate_dir, f"{result.approach}_{result.subject_id}.json")
        with open(filename, "w") as f:
            json.dump(result.model_dump(), f, indent=2)

    async def run_all_experiments(
        self, subjects: List[Subject]
    ) -> Tuple[
        List[AggregatedCalibrationMetrics],
        Optional[FriedmanResult],
        Dict[Tuple[str, str], WilcoxonResult],
    ]:
        """
        Run all three approaches and aggregate calibration results.

        Args:
            subjects: List of Subject objects with actual measurements

        Returns:
            Tuple of:
            - List of AggregatedCalibrationMetrics for each approach
            - FriedmanResult for overall comparison (or None if not enough data)
            - Dict of pairwise WilcoxonResults
        """
        approaches: List[Literal["baseline", "web_search", "probabilistic"]] = [
            "baseline",
            "web_search",
            "probabilistic",
        ]
        all_aggregated: List[AggregatedCalibrationMetrics] = []
        results_by_approach: Dict[str, List[ExperimentResult]] = {}

        for approach in approaches:
            print(f"\n{'=' * 60}")
            print(f"Running experiment: {approach}")
            print(f"{'=' * 60}\n")

            results = await self.run_single_experiment(approach, subjects)
            self.results.extend(results)
            results_by_approach[approach] = results

            # Aggregate calibration metrics
            aggregated = aggregate_calibration_results(results)
            aggregated.approach = approach
            all_aggregated.append(aggregated)

            print(f"\n{approach} completed:")
            print(f"  Valid predictions: {aggregated.n_predictions}/{len(subjects)}")
            if aggregated.mean_height_interval_score is not None:
                print(f"  Mean height interval score: {aggregated.mean_height_interval_score:.2f}")
            if aggregated.mean_weight_interval_score is not None:
                print(f"  Mean weight interval score: {aggregated.mean_weight_interval_score:.2f}")
            if 0.90 in aggregated.height_coverage_rates:
                print(
                    f"  Height 90% coverage: {aggregated.height_coverage_rates[0.90]:.1%} "
                    f"(target: 90%)"
                )
            if 0.90 in aggregated.weight_coverage_rates:
                print(
                    f"  Weight 90% coverage: {aggregated.weight_coverage_rates[0.90]:.1%} "
                    f"(target: 90%)"
                )

        # Run statistical analysis
        print(f"\n{'=' * 60}")
        print("Running Statistical Analysis")
        print(f"{'=' * 60}\n")

        friedman_result = None
        pairwise_results: Dict[Tuple[str, str], WilcoxonResult] = {}

        try:
            # Collect interval scores by approach for statistical tests
            scores_by_approach = self._collect_interval_scores(results_by_approach)

            if scores_by_approach:
                # Run Friedman test on combined interval scores
                friedman_result = friedman_test(scores_by_approach)
                print(
                    f"Friedman test: χ²={friedman_result.statistic:.2f}, "
                    f"p={friedman_result.p_value:.4f}"
                )

                # Run pairwise comparisons
                pairwise_results = all_pairwise_comparisons(scores_by_approach)
                for (a, b), result in pairwise_results.items():
                    sig = "*" if result.significant else ""
                    print(
                        f"  {a} vs {b}: p={result.p_value_corrected:.4f}{sig}, "
                        f"effect size r={result.effect_size:.3f}"
                    )
        except Exception as e:
            print(f"Statistical analysis failed: {e}")

        return all_aggregated, friedman_result, pairwise_results

    def _collect_interval_scores(
        self, results_by_approach: Dict[str, List[ExperimentResult]]
    ) -> Dict[str, List[float]]:
        """
        Collect interval scores for statistical testing.

        Uses combined height + weight interval scores for each subject.
        """
        scores: Dict[str, List[float]] = {}

        for approach, results in results_by_approach.items():
            approach_scores = []
            for r in results:
                if r.calibration_metrics is not None:
                    # Combine height and weight interval scores
                    h_score = r.calibration_metrics.height_interval_score
                    w_score = r.calibration_metrics.weight_interval_score
                    if h_score is not None and w_score is not None:
                        approach_scores.append(h_score + w_score)
                    elif h_score is not None:
                        approach_scores.append(h_score)
                    elif w_score is not None:
                        approach_scores.append(w_score)
                    else:
                        approach_scores.append(float("nan"))
                else:
                    approach_scores.append(float("nan"))
            scores[approach] = approach_scores

        return scores

    def save_results(
        self,
        aggregated_metrics: List[AggregatedCalibrationMetrics],
        friedman_result: Optional[FriedmanResult] = None,
        pairwise_results: Optional[Dict[Tuple[str, str], WilcoxonResult]] = None,
    ):
        """Save final calibration results to files."""
        import pandas as pd

        os.makedirs(self.results_dir, exist_ok=True)

        # Save as markdown table
        table = self._format_calibration_table(aggregated_metrics)
        with open(os.path.join(self.results_dir, "calibration_results.md"), "w") as f:
            f.write("# Calibration Study Results\n\n")
            f.write(table)
            f.write("\n\n## Metric Descriptions\n\n")
            f.write(
                "- **Coverage Rate**: Proportion of true values within predicted interval "
                "(should match nominal level if well-calibrated)\n"
            )
            f.write(
                "- **Calibration Error**: |Observed coverage - Nominal coverage| "
                "(lower is better)\n"
            )
            f.write("- **Interval Score**: Proper scoring rule (lower is better)\n")
            f.write(
                "- **Interval Width**: Average 90% interval width (narrower is better IF calibrated)\n"
            )
            f.write("- **MAE**: Mean Absolute Error on predicted mean (lower is better)\n")

            # Add statistical analysis if available
            if friedman_result is not None and pairwise_results is not None:
                f.write("\n")
                f.write(format_statistical_report(friedman_result, pairwise_results))

        # Save as CSV
        rows = []
        for m in aggregated_metrics:
            row = {
                "approach": m.approach,
                "n_predictions": m.n_predictions,
                "height_coverage_50": m.height_coverage_rates.get(0.50),
                "height_coverage_80": m.height_coverage_rates.get(0.80),
                "height_coverage_90": m.height_coverage_rates.get(0.90),
                "height_coverage_95": m.height_coverage_rates.get(0.95),
                "weight_coverage_50": m.weight_coverage_rates.get(0.50),
                "weight_coverage_80": m.weight_coverage_rates.get(0.80),
                "weight_coverage_90": m.weight_coverage_rates.get(0.90),
                "weight_coverage_95": m.weight_coverage_rates.get(0.95),
                "height_cal_error_90": m.height_calibration_error.get(0.90),
                "weight_cal_error_90": m.weight_calibration_error.get(0.90),
                "mean_height_interval_score": m.mean_height_interval_score,
                "mean_weight_interval_score": m.mean_weight_interval_score,
                "mean_height_interval_width_90": m.mean_height_interval_width_90,
                "mean_weight_interval_width_90": m.mean_weight_interval_width_90,
                "mean_height_mae": m.mean_height_mae,
                "mean_weight_mae": m.mean_weight_mae,
            }
            rows.append(row)

        df = pd.DataFrame(rows)
        df.to_csv(os.path.join(self.results_dir, "calibration_results.csv"), index=False)

        # Save detailed results
        with open(os.path.join(self.results_dir, "all_results.json"), "w") as f:
            json.dump([r.model_dump() for r in self.results], f, indent=2)

        print(f"\n✓ Results saved to {self.results_dir}/")

    def _format_calibration_table(self, metrics: List[AggregatedCalibrationMetrics]) -> str:
        """Format calibration metrics as a markdown table."""
        lines = [
            "## Coverage Rates (should match nominal levels)\n",
            "| Approach | N | 50% Height | 80% Height | 90% Height | 95% Height | 50% Weight | 80% Weight | 90% Weight | 95% Weight |",
            "|----------|---|------------|------------|------------|------------|------------|------------|------------|------------|",
        ]

        for m in metrics:
            h50 = m.height_coverage_rates.get(0.50, 0)
            h80 = m.height_coverage_rates.get(0.80, 0)
            h90 = m.height_coverage_rates.get(0.90, 0)
            h95 = m.height_coverage_rates.get(0.95, 0)
            w50 = m.weight_coverage_rates.get(0.50, 0)
            w80 = m.weight_coverage_rates.get(0.80, 0)
            w90 = m.weight_coverage_rates.get(0.90, 0)
            w95 = m.weight_coverage_rates.get(0.95, 0)
            lines.append(
                f"| {m.approach} | {m.n_predictions} | "
                f"{h50:.1%} | {h80:.1%} | {h90:.1%} | {h95:.1%} | "
                f"{w50:.1%} | {w80:.1%} | {w90:.1%} | {w95:.1%} |"
            )

        lines.append("| *Target* | - | 50% | 80% | 90% | 95% | 50% | 80% | 90% | 95% |")

        lines.extend(
            [
                "\n## Calibration & Sharpness\n",
                "| Approach | Height Cal Error (90%) | Weight Cal Error (90%) | Height Width (90%) | Weight Width (90%) | Height IS | Weight IS | Height MAE | Weight MAE |",
                "|----------|------------------------|------------------------|--------------------|--------------------|-----------|-----------|------------|------------|",
            ]
        )

        for m in metrics:
            h_err = m.height_calibration_error.get(0.90, 0)
            w_err = m.weight_calibration_error.get(0.90, 0)
            h_width = m.mean_height_interval_width_90 or 0
            w_width = m.mean_weight_interval_width_90 or 0
            h_is = m.mean_height_interval_score or 0
            w_is = m.mean_weight_interval_score or 0
            h_mae = m.mean_height_mae or 0
            w_mae = m.mean_weight_mae or 0
            lines.append(
                f"| {m.approach} | {h_err:.3f} | {w_err:.3f} | "
                f"{h_width:.1f} cm | {w_width:.1f} kg | "
                f"{h_is:.1f} | {w_is:.1f} | "
                f"{h_mae:.1f} cm | {w_mae:.1f} kg |"
            )

        return "\n".join(lines)


# Legacy functions for backwards compatibility


def load_test_data() -> List[GroundTruth]:
    """
    Load test subjects and ground truth (legacy function).

    For new calibration studies, use load_subjects() instead.
    """
    # Get the project root directory (parent of src/)
    project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    data_path = os.path.join(project_root, "data", "subjects.json")

    with open(data_path, "r") as f:
        data = json.load(f)

    return [GroundTruth(**item) for item in data]


def load_calibration_subjects(n: int = 50, seed: int = 42) -> List[Subject]:
    """Load subjects from NHANES data for calibration study."""
    return load_subjects(n=n, seed=seed)


async def main():
    """Main entry point for calibration study."""
    print("Height/Weight Prediction Calibration Study")
    print("=" * 60)

    # Load test data from NHANES
    print("Loading subjects from NHANES...")
    subjects = load_calibration_subjects(n=50, seed=42)
    print(f"Loaded {len(subjects)} subjects")

    # Show sample subject
    sample = subjects[0]
    print("\nSample subject:")
    print(f"  ID: {sample.subject_id}")
    print(f"  Description: {sample.text_description[:100]}...")
    print(f"  Actual height: {sample.actual_height_cm:.1f} cm")
    print(f"  Actual weight: {sample.actual_weight_kg:.1f} kg")

    # Run experiments
    runner = ExperimentRunner()
    aggregated_metrics, friedman_result, pairwise_results = await runner.run_all_experiments(
        subjects
    )

    # Save results
    runner.save_results(aggregated_metrics, friedman_result, pairwise_results)

    # Print summary
    print("\n" + "=" * 60)
    print("CALIBRATION STUDY RESULTS")
    print("=" * 60 + "\n")

    print("Coverage at 90% confidence level (target: 90%):")
    print("-" * 50)
    for m in aggregated_metrics:
        h90 = m.height_coverage_rates.get(0.90, 0)
        w90 = m.weight_coverage_rates.get(0.90, 0)
        print(f"  {m.approach:15s}: Height={h90:.1%}, Weight={w90:.1%}")

    print("\nCalibration Error at 90% level (lower is better):")
    print("-" * 50)
    for m in aggregated_metrics:
        h_err = m.height_calibration_error.get(0.90, 0)
        w_err = m.weight_calibration_error.get(0.90, 0)
        print(f"  {m.approach:15s}: Height={h_err:.3f}, Weight={w_err:.3f}")

    print("\nMean Interval Score (lower is better):")
    print("-" * 50)
    for m in aggregated_metrics:
        h_is = m.mean_height_interval_score or 0
        w_is = m.mean_weight_interval_score or 0
        print(f"  {m.approach:15s}: Height={h_is:.1f}, Weight={w_is:.1f}")

    # Statistical significance
    if friedman_result is not None:
        print("\nStatistical Significance:")
        print("-" * 50)
        print(
            f"  Friedman test: χ²={friedman_result.statistic:.2f}, p={friedman_result.p_value:.4f}"
        )
        if friedman_result.significant:
            print("  → Significant differences among approaches detected")

            # Report significant pairwise differences
            sig_pairs = [(a, b) for (a, b), r in pairwise_results.items() if r.significant]
            if sig_pairs:
                print("  Significant pairwise differences:")
                for a, b in sig_pairs:
                    r = pairwise_results[(a, b)]
                    print(
                        f"    • {a} vs {b}: p={r.p_value_corrected:.4f}, "
                        f"effect size r={r.effect_size:.3f}"
                    )
        else:
            print("  → No significant differences detected")


if __name__ == "__main__":
    # Run with Claude Agent SDK (uses Claude Code authentication)
    asyncio.run(main())
