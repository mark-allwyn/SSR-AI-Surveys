"""Advanced metrics calculator for radar charts and comparisons."""

import numpy as np
from typing import Dict, List, Optional


def calculate_radar_metrics(
    overall_llm: float,
    question_metrics: Dict,
    distributions_data: Optional[Dict] = None
) -> Dict[str, float]:
    """
    Calculate multi-dimensional metrics for radar chart visualization.

    Args:
        overall_llm: Overall LLM accuracy percentage
        question_metrics: Dict of question-level metrics
        distributions_data: Optional LLM distributions for confidence calculation

    Returns:
        Dict with 6-8 normalized metrics (0-100 scale) for radar chart
    """
    # Extract MAE values
    maes = [q['llm_mae'] for q in question_metrics.values()]
    avg_mae = np.mean(maes)

    # Extract accuracies
    accuracies = [q['llm_accuracy'] for q in question_metrics.values()]

    metrics = {}

    # 1. Accuracy (already 0-100)
    metrics['Accuracy'] = overall_llm

    # 2. Precision (inverse of MAE, normalized to 0-100)
    # MAE typically ranges 0-2 for Likert-5, so we use 100 - (MAE * 20)
    metrics['Precision'] = max(0, 100 - (avg_mae * 20))

    # 3. Consistency (inverse of standard deviation)
    std_accuracy = np.std(accuracies)
    metrics['Consistency'] = max(0, 100 - std_accuracy)

    # 4. Confidence (based on entropy if distributions available)
    if distributions_data:
        entropies = []
        for question_data in distributions_data.values():
            for resp_data in question_data.values():
                if 'entropy' in resp_data:
                    entropies.append(resp_data['entropy'])

        if entropies:
            avg_entropy = np.mean(entropies)
            # Entropy ranges roughly 0-2 for typical distributions
            # Lower entropy = higher confidence
            # Convert to 0-100 where 100 is high confidence (low entropy)
            metrics['Confidence'] = max(0, 100 - (avg_entropy * 50))
        else:
            metrics['Confidence'] = 65  # Default middle value
    else:
        metrics['Confidence'] = 65  # Default middle value

    # 5. Coverage (always 100% for completed experiments)
    metrics['Coverage'] = 100

    # 6. Calibration (simplified - based on how close predictions are to actual)
    # Use MAE as proxy - better calibration = lower MAE
    metrics['Calibration'] = max(0, 100 - (avg_mae * 25))

    return metrics


def calculate_confidence_from_distributions(distributions_data: Dict) -> float:
    """
    Calculate average confidence from probability distributions.

    Uses entropy as measure of confidence (lower entropy = higher confidence).

    Returns:
        Average confidence score (0-100, higher is more confident)
    """
    if not distributions_data:
        return 65.0

    entropies = []
    for question_data in distributions_data.values():
        for resp_data in question_data.values():
            if 'entropy' in resp_data:
                entropies.append(resp_data['entropy'])

    if not entropies:
        return 65.0

    avg_entropy = np.mean(entropies)
    # Convert entropy (typically 0-2) to confidence (0-100)
    confidence = max(0, min(100, 100 - (avg_entropy * 50)))

    return confidence


def calculate_calibration_score(distributions_data: Dict, ground_truth_data: Dict) -> float:
    """
    Calculate how well the predicted probabilities match actual outcomes.

    Perfect calibration: If model predicts 70% probability, outcome occurs 70% of the time.

    Returns:
        Calibration score (0-100, higher is better calibrated)
    """
    if not distributions_data:
        return 65.0

    # Simplified calibration: Compare predicted mode probabilities to accuracy
    mode_probs = []
    correct = []

    for question_id, question_data in distributions_data.items():
        for respondent_id, resp_data in question_data.items():
            probs = resp_data['probabilities']
            gt = resp_data['ground_truth']
            mode = resp_data['mode']

            if gt is not None:
                # Probability assigned to mode
                mode_prob = probs[mode - 1] if mode <= len(probs) else 0
                mode_probs.append(mode_prob)

                # Was prediction correct?
                correct.append(1 if mode == gt else 0)

    if not mode_probs:
        return 65.0

    # Calculate expected vs actual accuracy
    expected_accuracy = np.mean(mode_probs) * 100
    actual_accuracy = np.mean(correct) * 100

    # Calibration score: lower difference = better calibration
    diff = abs(expected_accuracy - actual_accuracy)
    calibration = max(0, 100 - diff)

    return calibration


def aggregate_distribution_stats(distributions_data: Dict, question_id: str) -> Dict:
    """
    Calculate aggregate statistics for a question's distributions.

    Args:
        distributions_data: Full distributions dict
        question_id: Specific question to analyze

    Returns:
        Dict with mean probabilities, std dev, entropy stats
    """
    if question_id not in distributions_data:
        return {}

    question_data = distributions_data[question_id]

    # Collect all probability vectors
    all_probs = []
    all_entropies = []
    all_ground_truths = []

    for resp_data in question_data.values():
        all_probs.append(resp_data['probabilities'])
        all_entropies.append(resp_data['entropy'])
        if resp_data['ground_truth'] is not None:
            all_ground_truths.append(resp_data['ground_truth'])

    # Calculate aggregates
    mean_probs = np.mean(all_probs, axis=0).tolist() if all_probs else []
    std_probs = np.std(all_probs, axis=0).tolist() if all_probs else []
    mean_entropy = np.mean(all_entropies) if all_entropies else 0

    # Ground truth distribution
    if all_ground_truths:
        gt_counts = np.bincount(all_ground_truths, minlength=len(mean_probs)+1)[1:]
        gt_distribution = (gt_counts / len(all_ground_truths)).tolist()
    else:
        gt_distribution = []

    return {
        'mean_probabilities': mean_probs,
        'std_probabilities': std_probs,
        'mean_entropy': mean_entropy,
        'ground_truth_distribution': gt_distribution,
        'n_samples': len(all_probs)
    }
