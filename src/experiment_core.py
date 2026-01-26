"""
Core experiment runner logic shared between different backends (Claude, Ollama).
"""

import os
import json
from typing import List, Literal, Any, Type, Protocol

from models.schemas import (
    PredictionResult,
    GroundTruth,
    ExperimentResult,
    AggregatedMetrics,
)
from evaluation.metrics import evaluate_prediction, aggregate_results, format_results_table


class Predictor(Protocol):
    """Protocol that all predictors must implement."""
    
    async def predict(self, person_description: str, max_retries: int = 3) -> PredictionResult:
        ...


class ExperimentRunner:
    """
    Runs the full experiment across all subjects and approaches.
    """

    def __init__(self, predictor_class: Type, **predictor_kwargs):
        """
        Initialize runner with a specific predictor class.
        
        Args:
            predictor_class: Class to instantiate for predictions
            **predictor_kwargs: Arguments to pass to predictor constructor
        """
        self.results: List[ExperimentResult] = []
        # Get project root directory
        self.project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
        self.results_dir = os.path.join(self.project_root, "results")
        self.predictor_class = predictor_class
        self.predictor_kwargs = predictor_kwargs

    async def run_single_experiment(
        self,
        approach: Literal["baseline", "web_search", "probabilistic"],
        subjects: List[GroundTruth],
    ) -> List[ExperimentResult]:
        """
        Run experiment for one approach across all subjects.

        Args:
            approach: Which approach to test
            subjects: List of GroundTruth objects with descriptions

        Returns:
            List of ExperimentResult objects
        """
        # Instantiate predictor with specific approach and any other kwargs
        predictor = self.predictor_class(approach=approach, **self.predictor_kwargs)
        results = []

        for i, subject in enumerate(subjects):
            print(f"[{approach}] Processing subject {i + 1}/{len(subjects)}...")

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

    def _save_intermediate(self, result: ExperimentResult):
        """Save intermediate results to avoid data loss."""
        intermediate_dir = os.path.join(self.results_dir, "intermediate")
        os.makedirs(intermediate_dir, exist_ok=True)
        filename = os.path.join(intermediate_dir, f"{result.approach}_{result.subject_id}.json")
        with open(filename, "w") as f:
            json.dump(result.model_dump(), f, indent=2)

    async def run_all_experiments(self, subjects: List[GroundTruth]) -> List[AggregatedMetrics]:
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
            print(f"\n{'=' * 60}")
            print(f"Running experiment: {approach}")
            print(f"{'=' * 60}\n")

            results = await self.run_single_experiment(approach, subjects)
            self.results.extend(results)

            # Aggregate
            aggregated = aggregate_results(results)
            all_aggregated.append(aggregated)

            print(f"\n{approach} completed:")
            print(f"  Valid: {aggregated.n_valid}/{aggregated.n_total}")
            print(f"  Invalid rate: {aggregated.invalid_rate_percent:.1f}%")
            if aggregated.n_valid > 0:
                print(f"  Mean KL (height): {aggregated.mean_kl_divergence_height:.3f}")
                print(f"  Mean KL (weight): {aggregated.mean_kl_divergence_weight:.3f}")

        return all_aggregated

    def save_results(self, aggregated_metrics: List[AggregatedMetrics]):
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


def load_test_data() -> List[GroundTruth]:
    """
    Load test subjects and ground truth.
    """
    # Get the project root directory (parent of src/)
    project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    data_path = os.path.join(project_root, "data", "subjects.json")

    with open(data_path, "r") as f:
        data = json.load(f)

    return [GroundTruth(**item) for item in data]
