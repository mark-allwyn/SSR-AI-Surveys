"""Visualization functions for SSR results."""

from typing import Dict, List, Optional
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
from pathlib import Path

from .ssr_model import RatingDistribution
from .analysis import QuestionAnalysis, SurveyAnalysis


# Set style
sns.set_style("whitegrid")
plt.rcParams['figure.figsize'] = (12, 6)


def plot_distribution(
    distribution: RatingDistribution,
    ax: Optional[plt.Axes] = None,
    title: Optional[str] = None
) -> plt.Axes:
    """
    Plot a single rating distribution.

    Args:
        distribution: RatingDistribution object
        ax: Matplotlib axes (creates new figure if None)
        title: Plot title

    Returns:
        Matplotlib axes
    """
    if ax is None:
        fig, ax = plt.subplots(figsize=(10, 6))

    scale_points = sorted(distribution.scale_labels.keys())
    labels = [distribution.scale_labels[p] for p in scale_points]

    # Bar plot
    bars = ax.bar(scale_points, distribution.distribution, color='steelblue', alpha=0.7)

    # Add value labels on bars
    for bar, prob in zip(bars, distribution.distribution):
        height = bar.get_height()
        ax.text(bar.get_x() + bar.get_width()/2., height,
                f'{prob:.2%}',
                ha='center', va='bottom', fontsize=10)

    ax.set_xlabel('Scale Point', fontsize=12)
    ax.set_ylabel('Probability', fontsize=12)
    ax.set_xticks(scale_points)
    ax.set_xticklabels(labels, rotation=45, ha='right')
    ax.set_ylim(0, 1.0)

    if title is None:
        title = f"Distribution for {distribution.respondent_id} - {distribution.question_id}"
    ax.set_title(title, fontsize=14, fontweight='bold')

    # Add metrics as text
    metrics_text = f"Expected Value: {distribution.expected_value:.2f}\n"
    metrics_text += f"Mode: {distribution.mode}\n"
    metrics_text += f"Entropy: {distribution.entropy:.3f}"
    ax.text(0.98, 0.97, metrics_text,
            transform=ax.transAxes,
            fontsize=10,
            verticalalignment='top',
            horizontalalignment='right',
            bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5))

    plt.tight_layout()
    return ax


def plot_question_analysis(
    analysis: QuestionAnalysis,
    ax: Optional[plt.Axes] = None
) -> plt.Axes:
    """
    Plot aggregated distribution for a question.

    Args:
        analysis: QuestionAnalysis object
        ax: Matplotlib axes (creates new figure if None)

    Returns:
        Matplotlib axes
    """
    if ax is None:
        fig, ax = plt.subplots(figsize=(10, 6))

    scale_points = sorted(analysis.scale_labels.keys())
    labels = [analysis.scale_labels[p] for p in scale_points]

    bars = ax.bar(scale_points, analysis.aggregated_distribution,
                   color='darkgreen', alpha=0.7)

    for bar, prob in zip(bars, analysis.aggregated_distribution):
        height = bar.get_height()
        ax.text(bar.get_x() + bar.get_width()/2., height,
                f'{prob:.2%}',
                ha='center', va='bottom', fontsize=10)

    ax.set_xlabel('Scale Point', fontsize=12)
    ax.set_ylabel('Aggregated Probability', fontsize=12)
    ax.set_xticks(scale_points)
    ax.set_xticklabels(labels, rotation=45, ha='right')
    ax.set_ylim(0, max(analysis.aggregated_distribution) * 1.2)

    title = f"Aggregated Distribution: {analysis.question_id}"
    title += f"\n(n={analysis.n_responses}, Mean={analysis.mean_expected_value:.2f}±{analysis.std_expected_value:.2f})"
    ax.set_title(title, fontsize=14, fontweight='bold')

    plt.tight_layout()
    return ax


def plot_survey_heatmap(
    distributions: List[RatingDistribution],
    question_ids: Optional[List[str]] = None,
    figsize: tuple = (14, 10)
) -> plt.Figure:
    """
    Plot heatmap of expected values across questions and respondents.

    Args:
        distributions: List of RatingDistribution objects
        question_ids: List of question IDs to include (None = all)
        figsize: Figure size

    Returns:
        Matplotlib figure
    """
    # Create pivot table
    data = []
    for dist in distributions:
        if question_ids is None or dist.question_id in question_ids:
            data.append({
                'respondent_id': dist.respondent_id,
                'question_id': dist.question_id,
                'expected_value': dist.expected_value
            })

    df = pd.DataFrame(data)
    pivot = df.pivot(index='respondent_id', columns='question_id', values='expected_value')

    fig, ax = plt.subplots(figsize=figsize)
    sns.heatmap(pivot, cmap='RdYlGn', center=3, vmin=1, vmax=5,
                cbar_kws={'label': 'Expected Rating'},
                ax=ax, linewidths=0.5)

    ax.set_title('Survey Responses Heatmap (Expected Values)', fontsize=16, fontweight='bold')
    ax.set_xlabel('Question ID', fontsize=12)
    ax.set_ylabel('Respondent ID', fontsize=12)

    plt.tight_layout()
    return fig


def plot_entropy_distribution(
    distributions: List[RatingDistribution],
    by_question: bool = True,
    figsize: tuple = (12, 6)
) -> plt.Figure:
    """
    Plot distribution of entropy (uncertainty) values.

    Args:
        distributions: List of RatingDistribution objects
        by_question: Whether to separate by question
        figsize: Figure size

    Returns:
        Matplotlib figure
    """
    entropies = [d.entropy for d in distributions]

    if by_question:
        # Group by question
        data = []
        for dist in distributions:
            data.append({
                'question_id': dist.question_id,
                'entropy': dist.entropy
            })
        df = pd.DataFrame(data)

        fig, ax = plt.subplots(figsize=figsize)
        sns.boxplot(data=df, x='question_id', y='entropy', ax=ax)
        ax.set_title('Response Uncertainty by Question', fontsize=14, fontweight='bold')
        ax.set_xlabel('Question ID', fontsize=12)
        ax.set_ylabel('Entropy (bits)', fontsize=12)
        plt.xticks(rotation=45, ha='right')

    else:
        fig, ax = plt.subplots(figsize=figsize)
        ax.hist(entropies, bins=30, color='coral', alpha=0.7, edgecolor='black')
        ax.axvline(np.mean(entropies), color='red', linestyle='--',
                   linewidth=2, label=f'Mean: {np.mean(entropies):.3f}')
        ax.set_xlabel('Entropy (bits)', fontsize=12)
        ax.set_ylabel('Frequency', fontsize=12)
        ax.set_title('Distribution of Response Uncertainty', fontsize=14, fontweight='bold')
        ax.legend()

    plt.tight_layout()
    return fig


def plot_comparison(
    distributions: List[RatingDistribution],
    ground_truth: Dict,
    question_id: str,
    figsize: tuple = (12, 6)
) -> plt.Figure:
    """
    Plot comparison between predicted and ground truth ratings.

    Args:
        distributions: List of RatingDistribution objects
        ground_truth: Dictionary mapping (respondent_id, question_id) to true rating
        question_id: Question to plot
        figsize: Figure size

    Returns:
        Matplotlib figure
    """
    # Filter distributions for this question
    q_dists = [d for d in distributions if d.question_id == question_id]

    predicted = []
    actual = []
    respondents = []

    for dist in q_dists:
        key = (dist.respondent_id, dist.question_id)
        if key in ground_truth:
            predicted.append(dist.expected_value)
            actual.append(ground_truth[key])
            respondents.append(dist.respondent_id)

    if not predicted:
        print(f"No ground truth data for question {question_id}")
        return None

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=figsize)

    # Scatter plot
    ax1.scatter(actual, predicted, alpha=0.6, s=50)
    ax1.plot([1, 5], [1, 5], 'r--', label='Perfect prediction')
    ax1.set_xlabel('Ground Truth Rating', fontsize=12)
    ax1.set_ylabel('Predicted Expected Value', fontsize=12)
    ax1.set_title(f'Predicted vs Actual: {question_id}', fontsize=14, fontweight='bold')
    ax1.legend()
    ax1.grid(True, alpha=0.3)

    # Error distribution
    errors = np.array(predicted) - np.array(actual)
    ax2.hist(errors, bins=20, color='orange', alpha=0.7, edgecolor='black')
    ax2.axvline(0, color='red', linestyle='--', linewidth=2)
    ax2.axvline(np.mean(errors), color='blue', linestyle='--',
                linewidth=2, label=f'Mean Error: {np.mean(errors):.3f}')
    ax2.set_xlabel('Prediction Error', fontsize=12)
    ax2.set_ylabel('Frequency', fontsize=12)
    ax2.set_title('Prediction Error Distribution', fontsize=14, fontweight='bold')
    ax2.legend()

    plt.tight_layout()
    return fig


def create_report(
    analysis: SurveyAnalysis,
    distributions: List[RatingDistribution],
    output_dir: str = "results"
) -> None:
    """
    Create a comprehensive visual report.

    Args:
        analysis: SurveyAnalysis object
        distributions: List of all RatingDistribution objects
        output_dir: Directory to save figures
    """
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)

    print(f"\nGenerating visual report in {output_dir}/...")

    # 1. Plot each question
    for question_id, q_analysis in analysis.question_analyses.items():
        fig, ax = plt.subplots(figsize=(10, 6))
        plot_question_analysis(q_analysis, ax=ax)
        plt.savefig(output_path / f"{question_id}_distribution.png", dpi=300, bbox_inches='tight')
        plt.close()

    # 2. Survey heatmap
    fig = plot_survey_heatmap(distributions)
    plt.savefig(output_path / "survey_heatmap.png", dpi=300, bbox_inches='tight')
    plt.close()

    # 3. Entropy distribution
    fig = plot_entropy_distribution(distributions, by_question=True)
    plt.savefig(output_path / "entropy_by_question.png", dpi=300, bbox_inches='tight')
    plt.close()

    fig = plot_entropy_distribution(distributions, by_question=False)
    plt.savefig(output_path / "entropy_distribution.png", dpi=300, bbox_inches='tight')
    plt.close()

    # 4. Sample individual distributions (first 6)
    sample_dists = distributions[:6]
    fig, axes = plt.subplots(2, 3, figsize=(18, 10))
    axes = axes.flatten()

    for i, dist in enumerate(sample_dists):
        plot_distribution(dist, ax=axes[i], title=f"{dist.respondent_id} - {dist.question_id}")

    plt.tight_layout()
    plt.savefig(output_path / "sample_distributions.png", dpi=300, bbox_inches='tight')
    plt.close()

    print(f"Report generated successfully! Check {output_dir}/ for figures.")
