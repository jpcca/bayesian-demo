"""
Experiment runner with distribution-to-distribution evaluation.

Extends the original example_runner.py by adding comparison of predicted
distributions against population-level ground truth distributions from NHANES,
in addition to the existing point-based metrics.

Usage:
    cd src
    python distribution_runner.py                        # run all approaches
    python distribution_runner.py --approach baseline    # run one approach
    python distribution_runner.py --approach baseline web_search  # run two

Resume behaviour: if results/intermediate/{approach}_{subject_id}.json already
exists the subject is skipped automatically, so re-running the script after an
interruption picks up where it left off.
"""

import argparse
import asyncio
import json
import os
from typing import List, Literal

from example_runner import ClaudePredictor
from models.schemas import (
    AggregatedMetrics,
    ExperimentResult,
    GroundTruth,
    SubjectDemographics,
)
from evaluation.metrics import (
    evaluate_prediction,
    aggregate_results,
    format_results_table,
    evaluate_against_distribution,
    aggregate_distribution_results,
    format_distribution_results_table,
)
from evaluation.matching import (
    load_ground_truth_distributions,
    match_subject_to_distribution,
)


class DistributionExperimentRunner:
    """
    Runs experiments with both point-based and distribution-based evaluation.

    Reuses ClaudePredictor from example_runner.py for predictions,
    then evaluates against both:
    1. Individual ground truth measurements (point metrics: NLL, z-score, etc.)
    2. Population ground truth distributions (distribution metrics: KL, Wasserstein, etc.)
    """

    def __init__(self):
        self.results: List[ExperimentResult] = []
        self.project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
        self.results_dir = os.path.join(self.project_root, "results")

        # Load population distributions
        gt_path = os.path.join(self.project_root, "data", "processed", "nhanes_ground_truth.json")
        self.distributions = load_ground_truth_distributions(gt_path)
        print(f"Loaded {len(self.distributions)} population distributions")

    # Delays (seconds) to wait after each consecutive rate-limit hit before retrying.
    _RATE_LIMIT_DELAYS = [60, 120, 300]

    async def run_single_experiment(
        self,
        approach: Literal["baseline", "web_search", "probabilistic"],
        subjects: List[GroundTruth],
    ) -> List[ExperimentResult]:
        """Run experiment for one approach across all subjects.

        Subjects whose intermediate result file already exists are skipped so
        that the run can be resumed after an interruption without re-processing
        completed subjects.

        When a rate-limit error is detected in the prediction result the method
        waits progressively longer (60 s → 120 s → 300 s) before retrying the
        same subject.
        """
        predictor = ClaudePredictor(approach=approach)
        intermediate_dir = os.path.join(self.results_dir, "intermediate")
        results = []

        for i, subject in enumerate(subjects):
            # ── Resume: load cached result if available ──────────────────────
            cached_file = os.path.join(
                intermediate_dir, f"{approach}_{subject.subject_id}.json"
            )
            if os.path.exists(cached_file):
                print(
                    f"[{approach}] Subject {i + 1}/{len(subjects)} "
                    f"(id={subject.subject_id}) already done — loading cache"
                )
                with open(cached_file) as f:
                    result = ExperimentResult(**json.load(f))
                results.append(result)
                continue

            print(f"[{approach}] Processing subject {i + 1}/{len(subjects)}...")

            # ── Predict with rate-limit outer retry ───────────────────────────
            prediction, token_usage = await self._predict_with_rate_limit_retry(
                predictor, subject.text_description, approach
            )

            # ── Point-based evaluation ────────────────────────────────────────
            point_metrics = evaluate_prediction(prediction, subject)

            # ── Distribution-based evaluation ─────────────────────────────────
            pop_gt = None
            dist_metrics = None
            if subject.demographics is not None:
                pop_gt = match_subject_to_distribution(
                    subject.demographics, self.distributions
                )
                dist_metrics = evaluate_against_distribution(prediction, pop_gt)

                if pop_gt:
                    print(
                        f"  Matched: {pop_gt.distribution_key} "
                        f"({pop_gt.n_variables_matched} vars, n={pop_gt.n})"
                    )

            # ── Store & persist ───────────────────────────────────────────────
            result = ExperimentResult(
                subject_id=subject.subject_id,
                approach=approach,
                prediction=prediction,
                ground_truth=subject,
                metrics=point_metrics,
                token_usage=token_usage,
                population_ground_truth=pop_gt,
                distribution_metrics=dist_metrics,
            )
            results.append(result)
            self._save_intermediate(result)

        return results

    async def _predict_with_rate_limit_retry(
        self,
        predictor: "ClaudePredictor",
        text_description: str,
        approach: str,
    ):
        """Call predictor.predict() and retry with backoff on rate-limit errors.

        predict() never raises — it returns an error PredictionResult instead.
        We detect rate-limit failures via the error string and wait before
        retrying so the caller doesn't need to worry about rate limits.
        """
        for delay_idx, delay in enumerate(self._RATE_LIMIT_DELAYS + [None]):
            prediction, token_usage = await predictor.predict(text_description)

            # Success or non-rate-limit error — return immediately
            if not prediction.error or "rate_limit" not in prediction.error.lower():
                return prediction, token_usage

            # Rate limited
            if delay is None:
                # Exhausted all retries
                print(
                    f"  [{approach}] Rate limit retries exhausted. "
                    "Returning error result."
                )
                return prediction, token_usage

            print(
                f"  [{approach}] Rate limited "
                f"(attempt {delay_idx + 1}/{len(self._RATE_LIMIT_DELAYS)}). "
                f"Waiting {delay}s before retry..."
            )
            await asyncio.sleep(delay)

        # Should not be reached
        return prediction, token_usage

    def _save_intermediate(self, result: ExperimentResult):
        """Save intermediate results to avoid data loss."""
        intermediate_dir = os.path.join(self.results_dir, "intermediate")
        os.makedirs(intermediate_dir, exist_ok=True)
        filename = os.path.join(
            intermediate_dir, f"{result.approach}_{result.subject_id}.json"
        )
        with open(filename, "w") as f:
            json.dump(result.model_dump(), f, indent=2)

    async def run_all_experiments(
        self,
        subjects: List[GroundTruth],
        approaches: List[str] | None = None,
    ) -> List[AggregatedMetrics]:
        """Run the specified approaches (default: all three) and aggregate results."""
        if approaches is None:
            approaches = ["baseline", "web_search", "probabilistic"]
        all_aggregated = []
        all_dist_aggregated = []

        for approach in approaches:
            print(f"\n{'=' * 60}")
            print(f"Running experiment: {approach}")
            print(f"{'=' * 60}\n")

            results = await self.run_single_experiment(approach, subjects)
            self.results.extend(results)

            # Point-based aggregation (existing)
            aggregated = aggregate_results(results)
            all_aggregated.append(aggregated)

            # Distribution-based aggregation (new)
            dist_agg = aggregate_distribution_results(results)
            all_dist_aggregated.append((approach, dist_agg))

            # Print summary
            print(f"\n{approach} completed:")
            print(f"  Valid: {aggregated.n_valid}/{aggregated.n_total}")
            if aggregated.n_valid > 0:
                print(f"  Mean NLL (height): {aggregated.mean_nll_height:.2f}")
                print(f"  Mean NLL (weight): {aggregated.mean_nll_weight:.2f}")

            if dist_agg.get("n_with_dist_metrics", 0) > 0:
                print(f"  Distribution metrics ({dist_agg['n_with_dist_metrics']} subjects):")
                print(f"    KL Div (H): {dist_agg['mean_kl_div_height']:.3f}")
                print(f"    KL Div (W): {dist_agg['mean_kl_div_weight']:.3f}")
                print(f"    Wasserstein (H): {dist_agg['mean_wasserstein_height']:.1f}")
                print(f"    Wasserstein (W): {dist_agg['mean_wasserstein_weight']:.1f}")
                print(f"    σ Ratio (H): {dist_agg['mean_sigma_ratio_height']:.2f}")
                print(f"    σ Ratio (W): {dist_agg['mean_sigma_ratio_weight']:.2f}")
                print(f"    Overlap (H): {dist_agg['mean_overlap_height']:.3f}")
                print(f"    Overlap (W): {dist_agg['mean_overlap_weight']:.3f}")

            if aggregated.mean_total_tokens is not None:
                print(f"  Mean total tokens: {aggregated.mean_total_tokens:.0f}")

        self._dist_aggregated = all_dist_aggregated
        return all_aggregated

    def save_results(self, aggregated_metrics: List[AggregatedMetrics]):
        """Save final results including distribution metrics."""
        import pandas as pd

        os.makedirs(self.results_dir, exist_ok=True)

        # Save markdown table with both metric types
        point_table = format_results_table(aggregated_metrics)
        dist_table = format_distribution_results_table(self._dist_aggregated)

        with open(os.path.join(self.results_dir, "experiment_results.md"), "w") as f:
            f.write("# Experiment Results\n\n")
            f.write("## Point-Based Metrics (Predicted Distribution vs Individual Measurement)\n\n")
            f.write(point_table)
            f.write("\n\n## Distribution-Based Metrics (Predicted Distribution vs Population Distribution)\n\n")
            f.write(dist_table)
            f.write("\n\n## Metric Descriptions\n\n")
            f.write("### Point-Based Metrics\n")
            f.write("- **NLL**: Negative log-likelihood of true value under predicted distribution (lower is better)\n")
            f.write("- **Abs Error**: |predicted_mean - true_value| in cm/kg\n")
            f.write("- **Mean |z|**: Mean absolute z-score (~0.8 for well-calibrated)\n")
            f.write("- **95% CI Coverage**: % of true values within 95% credible interval (~95% ideal)\n")
            f.write("\n### Distribution-Based Metrics\n")
            f.write("- **KL Div**: KL divergence from population to predicted distribution (lower is better, 0 = identical)\n")
            f.write("- **W₂²**: Squared Wasserstein-2 distance (lower is better, 0 = identical)\n")
            f.write("- **Shift**: |predicted_mean - population_mean| in cm/kg\n")
            f.write("- **σ Ratio**: predicted_σ / population_σ (ideal = 1.0; >1 underconfident, <1 overconfident)\n")
            f.write("- **Overlap**: Overlap coefficient of the two distributions (higher is better, 0-1)\n")

        # Save CSV
        df = pd.DataFrame([m.model_dump() for m in aggregated_metrics])
        df.to_csv(os.path.join(self.results_dir, "experiment_results.csv"), index=False)

        # Save detailed results JSON
        with open(os.path.join(self.results_dir, "all_results.json"), "w") as f:
            json.dump([r.model_dump() for r in self.results], f, indent=2)

        # Save distribution aggregated metrics separately
        with open(os.path.join(self.results_dir, "distribution_metrics.json"), "w") as f:
            json.dump(
                {approach: metrics for approach, metrics in self._dist_aggregated},
                f,
                indent=2,
            )

        print(f"\n✓ Results saved to {self.results_dir}/")


def load_test_data() -> List[GroundTruth]:
    """Load test subjects with demographics for distribution matching."""
    project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    data_path = os.path.join(project_root, "data", "subjects.json")

    with open(data_path, "r") as f:
        data = json.load(f)

    subjects = []
    for item in data:
        demographics = None
        if "demographics" in item and item["demographics"]:
            demographics = SubjectDemographics(**item["demographics"])

        subjects.append(
            GroundTruth(
                subject_id=item["subject_id"],
                height_cm=item["height_cm"],
                weight_kg=item["weight_kg"],
                text_description=item["text_description"],
                demographics=demographics,
            )
        )

    return subjects


def parse_args():
    parser = argparse.ArgumentParser(
        description="Run height/weight prediction evaluation with distribution metrics."
    )
    parser.add_argument(
        "--approach",
        nargs="+",
        choices=["baseline", "web_search", "probabilistic"],
        default=None,
        metavar="APPROACH",
        help=(
            "Which approach(es) to run. Defaults to all three. "
            "Example: --approach baseline web_search"
        ),
    )
    return parser.parse_args()


async def main():
    """Main entry point."""
    args = parse_args()
    approaches = args.approach  # None means "all three"

    print("Height/Weight Prediction Evaluation (with Distribution Metrics)")
    print("=" * 60)
    if approaches:
        print(f"Approaches: {', '.join(approaches)}")
    else:
        print("Approaches: baseline, web_search, probabilistic")
    print("Resume: subjects with cached intermediate results are skipped")
    print("=" * 60)

    # Load test data
    print("\nLoading test subjects...")
    subjects = load_test_data()
    print(f"Loaded {len(subjects)} subjects")

    n_with_demo = sum(1 for s in subjects if s.demographics is not None)
    print(f"Subjects with demographics: {n_with_demo}/{len(subjects)}")

    # Run experiments
    runner = DistributionExperimentRunner()
    aggregated_metrics = await runner.run_all_experiments(subjects, approaches=approaches)

    # Save results
    runner.save_results(aggregated_metrics)

    # Print summary tables
    print("\n" + "=" * 60)
    print("FINAL RESULTS — Point Metrics")
    print("=" * 60 + "\n")
    print(format_results_table(aggregated_metrics))

    print("\n" + "=" * 60)
    print("FINAL RESULTS — Distribution Metrics")
    print("=" * 60 + "\n")
    print(format_distribution_results_table(runner._dist_aggregated))


if __name__ == "__main__":
    asyncio.run(main())
