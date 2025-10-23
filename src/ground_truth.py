"""Ground truth comparison and evaluation module."""

from typing import Dict, List, Tuple, Optional
import numpy as np
import pandas as pd
from dataclasses import dataclass
from scipy import stats

from .ssr_model import RatingDistribution
from .survey import Question


@dataclass
class GroundTruthComparison:
    """Comparison between predicted and ground truth responses."""
    question_id: str
    question_type: str

    # Accuracy metrics
    mode_accuracy: float  # % where predicted mode matches ground truth
    top2_accuracy: float  # % where ground truth is in top 2 predicted

    # Error metrics
    mae: float  # Mean absolute error (for ordinal scales)
    rmse: float  # Root mean squared error

    # Probabilistic metrics
    mean_probability_at_truth: float  # Average probability assigned to true answer
    log_likelihood: float  # Sum of log probabilities

    # Distribution metrics
    kl_divergence: float  # KL divergence from empirical distribution

    # Detailed breakdown
    confusion_matrix: np.ndarray
    n_samples: int


def create_ground_truth_dict(
    responses_df: pd.DataFrame,
    ground_truth_column: str = 'ground_truth'
) -> Dict[Tuple[str, str], int]:
    """
    Create ground truth dictionary from DataFrame.

    Args:
        responses_df: DataFrame with columns [respondent_id, question_id, ground_truth]
        ground_truth_column: Name of column containing ground truth ratings

    Returns:
        Dictionary mapping (respondent_id, question_id) to ground truth value
    """
    ground_truth = {}
    for _, row in responses_df.iterrows():
        key = (row['respondent_id'], row['question_id'])
        ground_truth[key] = int(row[ground_truth_column])
    return ground_truth


def evaluate_against_ground_truth(
    distributions: List[RatingDistribution],
    ground_truth: Dict[Tuple[str, str], int],
    question: Question
) -> GroundTruthComparison:
    """
    Evaluate predictions against ground truth for a single question.

    Args:
        distributions: List of RatingDistribution objects
        ground_truth: Dictionary mapping (respondent_id, question_id) to true rating
        question: Question object

    Returns:
        GroundTruthComparison object with evaluation metrics
    """
    predicted_modes = []
    predicted_probs = []
    true_values = []
    prob_at_truth = []

    # Extract data
    for dist in distributions:
        key = (dist.respondent_id, dist.question_id)
        if key in ground_truth:
            true_val = ground_truth[key]
            true_values.append(true_val)
            predicted_modes.append(dist.mode)
            predicted_probs.append(dist.distribution)

            # Get probability assigned to true answer
            ref_statements = question.get_reference_statements()
            scale_points = sorted(ref_statements.keys())
            true_idx = scale_points.index(true_val)
            prob_at_truth.append(dist.distribution[true_idx])

    if not true_values:
        raise ValueError("No matching ground truth data found")

    predicted_modes = np.array(predicted_modes)
    true_values = np.array(true_values)
    predicted_probs = np.array(predicted_probs)

    # Accuracy metrics
    mode_accuracy = np.mean(predicted_modes == true_values)

    # Top-2 accuracy
    top2_correct = 0
    for i, true_val in enumerate(true_values):
        ref_statements = question.get_reference_statements()
        scale_points = np.array(sorted(ref_statements.keys()))
        true_idx = np.where(scale_points == true_val)[0][0]

        # Get top 2 predicted indices
        top2_indices = np.argsort(predicted_probs[i])[-2:]
        top2_correct += (true_idx in top2_indices)

    top2_accuracy = top2_correct / len(true_values)

    # Error metrics (for ordinal scales)
    mae = np.mean(np.abs(predicted_modes - true_values))
    rmse = np.sqrt(np.mean((predicted_modes - true_values) ** 2))

    # Probabilistic metrics
    mean_prob_at_truth = np.mean(prob_at_truth)
    log_likelihood = np.sum(np.log(np.maximum(prob_at_truth, 1e-10)))

    # Confusion matrix
    ref_statements = question.get_reference_statements()
    scale_points = sorted(ref_statements.keys())
    n_options = len(scale_points)

    confusion = np.zeros((n_options, n_options))
    for true_val, pred_mode in zip(true_values, predicted_modes):
        true_idx = scale_points.index(true_val)
        pred_idx = scale_points.index(pred_mode)
        confusion[true_idx, pred_idx] += 1

    # KL divergence (compare predicted distribution to empirical)
    empirical_dist = np.bincount([scale_points.index(v) for v in true_values], minlength=n_options)
    empirical_dist = empirical_dist / empirical_dist.sum()

    avg_predicted_dist = predicted_probs.mean(axis=0)

    # KL divergence: sum(empirical * log(empirical / predicted))
    kl_div = np.sum(
        empirical_dist * np.log(
            np.maximum(empirical_dist, 1e-10) / np.maximum(avg_predicted_dist, 1e-10)
        )
    )

    return GroundTruthComparison(
        question_id=question.id,
        question_type=question.type,
        mode_accuracy=mode_accuracy,
        top2_accuracy=top2_accuracy,
        mae=mae,
        rmse=rmse,
        mean_probability_at_truth=mean_prob_at_truth,
        log_likelihood=log_likelihood,
        kl_divergence=kl_div,
        confusion_matrix=confusion,
        n_samples=len(true_values)
    )


def compare_human_vs_llm_ground_truth(
    human_distributions: List[RatingDistribution],
    llm_distributions: List[RatingDistribution],
    ground_truth: Dict[Tuple[str, str], int],
    question: Question
) -> Dict[str, GroundTruthComparison]:
    """
    Compare both human and LLM predictions against ground truth.

    Args:
        human_distributions: Human response distributions
        llm_distributions: LLM response distributions
        ground_truth: Ground truth ratings
        question: Question object

    Returns:
        Dictionary with 'human' and 'llm' GroundTruthComparison objects
    """
    human_eval = evaluate_against_ground_truth(human_distributions, ground_truth, question)
    llm_eval = evaluate_against_ground_truth(llm_distributions, ground_truth, question)

    return {
        'human': human_eval,
        'llm': llm_eval
    }


def print_ground_truth_comparison(comparison: GroundTruthComparison) -> None:
    """Print a summary of ground truth comparison."""
    print(f"\n{'='*70}")
    print(f"GROUND TRUTH EVALUATION: {comparison.question_id}")
    print(f"Question Type: {comparison.question_type}")
    print(f"Samples: {comparison.n_samples}")
    print(f"{'='*70}")

    print(f"\n📊 ACCURACY METRICS:")
    print(f"  Mode Accuracy:      {comparison.mode_accuracy:.1%}")
    print(f"  Top-2 Accuracy:     {comparison.top2_accuracy:.1%}")

    print(f"\n📏 ERROR METRICS:")
    print(f"  MAE:                {comparison.mae:.3f}")
    print(f"  RMSE:               {comparison.rmse:.3f}")

    print(f"\n🎲 PROBABILISTIC METRICS:")
    print(f"  Avg Prob at Truth:  {comparison.mean_probability_at_truth:.3f}")
    print(f"  Log Likelihood:     {comparison.log_likelihood:.2f}")
    print(f"  KL Divergence:      {comparison.kl_divergence:.4f}")

    print(f"\n📋 CONFUSION MATRIX:")
    print("  Rows: True, Columns: Predicted")

    # Format confusion matrix
    n = comparison.confusion_matrix.shape[0]
    header = "     " + "".join([f"{i+1:4d} " for i in range(n)])
    print(header)
    print("  " + "-" * len(header))

    for i, row in enumerate(comparison.confusion_matrix):
        row_str = f"  {i+1} |" + "".join([f"{int(val):4d} " for val in row])
        print(row_str)

    print(f"\n{'='*70}\n")


def create_comparison_dataframe(
    human_comparison: GroundTruthComparison,
    llm_comparison: GroundTruthComparison
) -> pd.DataFrame:
    """
    Create a comparison DataFrame for human vs LLM performance.

    Args:
        human_comparison: Human evaluation results
        llm_comparison: LLM evaluation results

    Returns:
        DataFrame comparing metrics
    """
    data = {
        'Metric': [
            'Mode Accuracy',
            'Top-2 Accuracy',
            'MAE',
            'RMSE',
            'Avg Prob at Truth',
            'Log Likelihood',
            'KL Divergence'
        ],
        'Human': [
            f"{human_comparison.mode_accuracy:.1%}",
            f"{human_comparison.top2_accuracy:.1%}",
            f"{human_comparison.mae:.3f}",
            f"{human_comparison.rmse:.3f}",
            f"{human_comparison.mean_probability_at_truth:.3f}",
            f"{human_comparison.log_likelihood:.2f}",
            f"{human_comparison.kl_divergence:.4f}"
        ],
        'LLM': [
            f"{llm_comparison.mode_accuracy:.1%}",
            f"{llm_comparison.top2_accuracy:.1%}",
            f"{llm_comparison.mae:.3f}",
            f"{llm_comparison.rmse:.3f}",
            f"{llm_comparison.mean_probability_at_truth:.3f}",
            f"{llm_comparison.log_likelihood:.2f}",
            f"{llm_comparison.kl_divergence:.4f}"
        ]
    }

    df = pd.DataFrame(data)

    # Add winner column
    winners = []
    metrics_higher_better = ['Mode Accuracy', 'Top-2 Accuracy', 'Avg Prob at Truth', 'Log Likelihood']

    for metric in df['Metric']:
        h_val = float(df[df['Metric']==metric]['Human'].values[0].rstrip('%'))
        l_val = float(df[df['Metric']==metric]['LLM'].values[0].rstrip('%'))

        if metric in metrics_higher_better:
            winner = 'Human' if h_val > l_val else ('LLM' if l_val > h_val else 'Tie')
        else:
            winner = 'Human' if h_val < l_val else ('LLM' if l_val < h_val else 'Tie')

        winners.append(winner)

    df['Better'] = winners

    return df
