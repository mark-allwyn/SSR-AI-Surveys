"""Analysis functions for SSR results."""

from typing import Dict, List, Optional, Tuple
import numpy as np
import pandas as pd
from dataclasses import dataclass

from .ssr_model import RatingDistribution


@dataclass
class QuestionAnalysis:
    """Analysis results for a single question."""
    question_id: str
    n_responses: int
    mean_expected_value: float
    std_expected_value: float
    mean_entropy: float
    mode_distribution: Dict[int, int]  # Count of each mode
    aggregated_distribution: np.ndarray
    scale_labels: Dict[int, str]


@dataclass
class SurveyAnalysis:
    """Analysis results for entire survey."""
    question_analyses: Dict[str, QuestionAnalysis]
    n_total_responses: int
    overall_uncertainty: float  # Mean entropy across all responses


def aggregate_distributions(
    distributions: List[RatingDistribution],
    method: str = "mean"
) -> np.ndarray:
    """
    Aggregate multiple probability distributions.

    Args:
        distributions: List of RatingDistribution objects
        method: Aggregation method ('mean', 'median', 'mode')

    Returns:
        Aggregated probability distribution
    """
    dist_arrays = np.array([d.distribution for d in distributions])

    if method == "mean":
        return np.mean(dist_arrays, axis=0)
    elif method == "median":
        return np.median(dist_arrays, axis=0)
    elif method == "mode":
        # Mode: most common distribution (simplified - take mean for now)
        return np.mean(dist_arrays, axis=0)
    else:
        raise ValueError(f"Unknown aggregation method: {method}")


def analyze_question(
    distributions: List[RatingDistribution],
    question_id: str
) -> QuestionAnalysis:
    """
    Analyze distributions for a single question.

    Args:
        distributions: List of RatingDistribution objects for this question
        question_id: Question identifier

    Returns:
        QuestionAnalysis object
    """
    if not distributions:
        raise ValueError(f"No distributions provided for question {question_id}")

    # Extract metrics
    expected_values = [d.expected_value for d in distributions]
    entropies = [d.entropy for d in distributions]
    modes = [d.mode for d in distributions]

    # Mode distribution (count of each mode)
    unique_modes, mode_counts = np.unique(modes, return_counts=True)
    mode_distribution = dict(zip(unique_modes.tolist(), mode_counts.tolist()))

    # Aggregate distributions
    aggregated = aggregate_distributions(distributions, method="mean")

    return QuestionAnalysis(
        question_id=question_id,
        n_responses=len(distributions),
        mean_expected_value=np.mean(expected_values),
        std_expected_value=np.std(expected_values),
        mean_entropy=np.mean(entropies),
        mode_distribution=mode_distribution,
        aggregated_distribution=aggregated,
        scale_labels=distributions[0].scale_labels
    )


def analyze_survey(
    distributions: List[RatingDistribution]
) -> SurveyAnalysis:
    """
    Analyze all distributions from a survey.

    Args:
        distributions: List of all RatingDistribution objects

    Returns:
        SurveyAnalysis object
    """
    # Group by question
    by_question: Dict[str, List[RatingDistribution]] = {}
    for dist in distributions:
        if dist.question_id not in by_question:
            by_question[dist.question_id] = []
        by_question[dist.question_id].append(dist)

    # Analyze each question
    question_analyses = {}
    for question_id, q_distributions in by_question.items():
        question_analyses[question_id] = analyze_question(q_distributions, question_id)

    # Overall metrics
    all_entropies = [d.entropy for d in distributions]
    overall_uncertainty = np.mean(all_entropies)

    return SurveyAnalysis(
        question_analyses=question_analyses,
        n_total_responses=len(distributions),
        overall_uncertainty=overall_uncertainty
    )


def create_results_dataframe(
    distributions: List[RatingDistribution]
) -> pd.DataFrame:
    """
    Create a pandas DataFrame from rating distributions.

    Args:
        distributions: List of RatingDistribution objects

    Returns:
        DataFrame with columns: respondent_id, question_id, expected_value,
                                entropy, mode, text_response
    """
    records = []
    for dist in distributions:
        record = {
            'respondent_id': dist.respondent_id,
            'question_id': dist.question_id,
            'expected_value': dist.expected_value,
            'entropy': dist.entropy,
            'mode': dist.mode,
            'text_response': dist.text_response
        }

        # Add individual probabilities
        for scale_point, label in dist.scale_labels.items():
            record[f'prob_{scale_point}'] = dist.distribution[scale_point - min(dist.scale_labels.keys())]

        records.append(record)

    return pd.DataFrame(records)


def compare_to_ground_truth(
    distributions: List[RatingDistribution],
    ground_truth: Dict[Tuple[str, str], int]  # (respondent_id, question_id) -> true_rating
) -> Dict[str, float]:
    """
    Compare predicted distributions to ground truth ratings.

    Args:
        distributions: List of RatingDistribution objects
        ground_truth: Dictionary mapping (respondent_id, question_id) to true rating

    Returns:
        Dictionary with evaluation metrics
    """
    predicted_modes = []
    predicted_expected = []
    true_ratings = []

    for dist in distributions:
        key = (dist.respondent_id, dist.question_id)
        if key in ground_truth:
            predicted_modes.append(dist.mode)
            predicted_expected.append(dist.expected_value)
            true_ratings.append(ground_truth[key])

    if not true_ratings:
        return {}

    predicted_modes = np.array(predicted_modes)
    predicted_expected = np.array(predicted_expected)
    true_ratings = np.array(true_ratings)

    # Calculate metrics
    mode_accuracy = np.mean(predicted_modes == true_ratings)
    mae_mode = np.mean(np.abs(predicted_modes - true_ratings))
    mae_expected = np.mean(np.abs(predicted_expected - true_ratings))
    rmse_expected = np.sqrt(np.mean((predicted_expected - true_ratings) ** 2))

    return {
        'mode_accuracy': mode_accuracy,
        'mae_mode': mae_mode,
        'mae_expected_value': mae_expected,
        'rmse_expected_value': rmse_expected,
        'n_comparisons': len(true_ratings)
    }


def print_analysis_summary(analysis: SurveyAnalysis) -> None:
    """Print a summary of survey analysis results."""
    print("\n" + "=" * 60)
    print("SURVEY ANALYSIS SUMMARY")
    print("=" * 60)
    print(f"\nTotal Responses: {analysis.n_total_responses}")
    print(f"Overall Uncertainty (Mean Entropy): {analysis.overall_uncertainty:.3f}")
    print(f"\nNumber of Questions Analyzed: {len(analysis.question_analyses)}")

    for question_id, q_analysis in analysis.question_analyses.items():
        print(f"\n{'-' * 60}")
        print(f"Question: {question_id}")
        print(f"Responses: {q_analysis.n_responses}")
        print(f"Mean Rating (Expected Value): {q_analysis.mean_expected_value:.2f} ± {q_analysis.std_expected_value:.2f}")
        print(f"Mean Uncertainty (Entropy): {q_analysis.mean_entropy:.3f}")

        print("\nAggregated Distribution:")
        for i, (scale_point, label) in enumerate(sorted(q_analysis.scale_labels.items())):
            prob = q_analysis.aggregated_distribution[i]
            bar = "█" * int(prob * 50)
            print(f"  {scale_point}. {label:30s} [{prob:5.1%}] {bar}")

        print("\nMode Distribution (Most Likely Ratings):")
        for mode, count in sorted(q_analysis.mode_distribution.items()):
            pct = count / q_analysis.n_responses * 100
            print(f"  {mode}: {count:3d} ({pct:5.1f}%)")

    print("\n" + "=" * 60)
