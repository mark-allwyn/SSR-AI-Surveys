"""Comparison module for analyzing human vs LLM responses."""

from typing import Dict, List, Tuple
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from scipy import stats
from dataclasses import dataclass

from .ssr_model import RatingDistribution
from .analysis import QuestionAnalysis, analyze_survey


@dataclass
class ComparisonMetrics:
    """Metrics comparing two sets of distributions."""
    question_id: str

    # Mean comparisons
    human_mean: float
    llm_mean: float
    mean_difference: float
    mean_effect_size: float  # Cohen's d

    # Spread comparisons
    human_std: float
    llm_std: float
    std_ratio: float

    # Distribution comparisons
    human_entropy: float
    llm_entropy: float
    entropy_difference: float

    # Statistical tests
    ks_statistic: float
    ks_pvalue: float
    ttest_statistic: float
    ttest_pvalue: float


def compare_distributions(
    human_dists: List[RatingDistribution],
    llm_dists: List[RatingDistribution],
    question_id: str
) -> ComparisonMetrics:
    """
    Compare human and LLM distributions for a single question.

    Args:
        human_dists: List of RatingDistribution objects from human responses
        llm_dists: List of RatingDistribution objects from LLM responses
        question_id: Question identifier

    Returns:
        ComparisonMetrics object
    """
    # Extract expected values
    human_values = np.array([d.expected_value for d in human_dists])
    llm_values = np.array([d.expected_value for d in llm_dists])

    # Extract entropies
    human_entropies = np.array([d.entropy for d in human_dists])
    llm_entropies = np.array([d.entropy for d in llm_dists])

    # Mean comparisons
    human_mean = np.mean(human_values)
    llm_mean = np.mean(llm_values)
    mean_difference = llm_mean - human_mean

    # Cohen's d effect size
    pooled_std = np.sqrt((np.var(human_values) + np.var(llm_values)) / 2)
    mean_effect_size = mean_difference / pooled_std if pooled_std > 0 else 0

    # Spread comparisons
    human_std = np.std(human_values)
    llm_std = np.std(llm_values)
    std_ratio = llm_std / human_std if human_std > 0 else 0

    # Entropy comparisons
    human_entropy = np.mean(human_entropies)
    llm_entropy = np.mean(llm_entropies)
    entropy_difference = llm_entropy - human_entropy

    # Statistical tests
    ks_stat, ks_pval = stats.ks_2samp(human_values, llm_values)
    t_stat, t_pval = stats.ttest_ind(human_values, llm_values)

    return ComparisonMetrics(
        question_id=question_id,
        human_mean=human_mean,
        llm_mean=llm_mean,
        mean_difference=mean_difference,
        mean_effect_size=mean_effect_size,
        human_std=human_std,
        llm_std=llm_std,
        std_ratio=std_ratio,
        human_entropy=human_entropy,
        llm_entropy=llm_entropy,
        entropy_difference=entropy_difference,
        ks_statistic=ks_stat,
        ks_pvalue=ks_pval,
        ttest_statistic=t_stat,
        ttest_pvalue=t_pval
    )


def compare_survey_responses(
    human_dists: List[RatingDistribution],
    llm_dists: List[RatingDistribution]
) -> Dict[str, ComparisonMetrics]:
    """
    Compare human and LLM responses across all questions.

    Args:
        human_dists: All human response distributions
        llm_dists: All LLM response distributions

    Returns:
        Dictionary mapping question_id to ComparisonMetrics
    """
    # Group by question
    human_by_q = {}
    llm_by_q = {}

    for dist in human_dists:
        if dist.question_id not in human_by_q:
            human_by_q[dist.question_id] = []
        human_by_q[dist.question_id].append(dist)

    for dist in llm_dists:
        if dist.question_id not in llm_by_q:
            llm_by_q[dist.question_id] = []
        llm_by_q[dist.question_id].append(dist)

    # Compare each question
    comparisons = {}
    for question_id in human_by_q.keys():
        if question_id in llm_by_q:
            comparisons[question_id] = compare_distributions(
                human_by_q[question_id],
                llm_by_q[question_id],
                question_id
            )

    return comparisons


def plot_comparison_means(
    comparisons: Dict[str, ComparisonMetrics],
    figsize: Tuple[int, int] = (12, 6)
) -> plt.Figure:
    """
    Plot comparison of means between human and LLM responses.

    Args:
        comparisons: Dictionary of ComparisonMetrics
        figsize: Figure size

    Returns:
        Matplotlib figure
    """
    questions = list(comparisons.keys())
    human_means = [comparisons[q].human_mean for q in questions]
    llm_means = [comparisons[q].llm_mean for q in questions]
    human_stds = [comparisons[q].human_std for q in questions]
    llm_stds = [comparisons[q].llm_std for q in questions]

    fig, ax = plt.subplots(figsize=figsize)

    x = np.arange(len(questions))
    width = 0.35

    bars1 = ax.bar(x - width/2, human_means, width, label='Human',
                   yerr=human_stds, capsize=5, color='steelblue', alpha=0.8)
    bars2 = ax.bar(x + width/2, llm_means, width, label='LLM',
                   yerr=llm_stds, capsize=5, color='coral', alpha=0.8)

    ax.set_xlabel('Question', fontsize=12, fontweight='bold')
    ax.set_ylabel('Mean Rating', fontsize=12, fontweight='bold')
    ax.set_title('Human vs LLM Mean Ratings Comparison', fontsize=14, fontweight='bold')
    ax.set_xticks(x)
    ax.set_xticklabels(questions, rotation=45, ha='right')
    ax.legend()
    ax.grid(axis='y', alpha=0.3)
    ax.set_ylim(1, 5)

    # Add significance markers
    for i, q in enumerate(questions):
        if comparisons[q].ttest_pvalue < 0.001:
            marker = '***'
        elif comparisons[q].ttest_pvalue < 0.01:
            marker = '**'
        elif comparisons[q].ttest_pvalue < 0.05:
            marker = '*'
        else:
            marker = 'ns'

        y_pos = max(llm_means[i] + llm_stds[i], human_means[i] + human_stds[i]) + 0.2
        ax.text(i, y_pos, marker, ha='center', fontsize=12, fontweight='bold')

    plt.tight_layout()
    return fig


def plot_comparison_distributions(
    comparisons: Dict[str, ComparisonMetrics],
    human_dists: List[RatingDistribution],
    llm_dists: List[RatingDistribution],
    figsize: Tuple[int, int] = (14, 10)
) -> plt.Figure:
    """
    Plot distribution comparisons for all questions.

    Args:
        comparisons: Dictionary of ComparisonMetrics
        human_dists: All human distributions
        llm_dists: All LLM distributions
        figsize: Figure size

    Returns:
        Matplotlib figure
    """
    questions = list(comparisons.keys())
    n_questions = len(questions)

    fig, axes = plt.subplots(n_questions, 1, figsize=figsize)
    if n_questions == 1:
        axes = [axes]

    for idx, question_id in enumerate(questions):
        ax = axes[idx]

        # Get distributions for this question
        human_q = [d for d in human_dists if d.question_id == question_id]
        llm_q = [d for d in llm_dists if d.question_id == question_id]

        human_values = [d.expected_value for d in human_q]
        llm_values = [d.expected_value for d in llm_q]

        # Create histograms
        bins = np.linspace(1, 5, 30)
        ax.hist(human_values, bins=bins, alpha=0.6, label='Human',
                color='steelblue', density=True, edgecolor='black')
        ax.hist(llm_values, bins=bins, alpha=0.6, label='LLM',
                color='coral', density=True, edgecolor='black')

        # Add vertical lines for means
        ax.axvline(comparisons[question_id].human_mean, color='darkblue',
                   linestyle='--', linewidth=2, label=f'Human Mean: {comparisons[question_id].human_mean:.2f}')
        ax.axvline(comparisons[question_id].llm_mean, color='darkred',
                   linestyle='--', linewidth=2, label=f'LLM Mean: {comparisons[question_id].llm_mean:.2f}')

        ax.set_xlabel('Expected Value', fontsize=10)
        ax.set_ylabel('Density', fontsize=10)
        ax.set_title(f'{question_id}\n(KS p={comparisons[question_id].ks_pvalue:.4f}, Effect size={comparisons[question_id].mean_effect_size:.2f})',
                     fontsize=11, fontweight='bold')
        ax.legend(fontsize=9)
        ax.grid(alpha=0.3)

    plt.tight_layout()
    return fig


def plot_comparison_spread(
    comparisons: Dict[str, ComparisonMetrics],
    figsize: Tuple[int, int] = (12, 6)
) -> plt.Figure:
    """
    Plot comparison of spread (standard deviation) between human and LLM.

    Args:
        comparisons: Dictionary of ComparisonMetrics
        figsize: Figure size

    Returns:
        Matplotlib figure
    """
    questions = list(comparisons.keys())
    human_stds = [comparisons[q].human_std for q in questions]
    llm_stds = [comparisons[q].llm_std for q in questions]

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=figsize)

    # Subplot 1: Bar chart of standard deviations
    x = np.arange(len(questions))
    width = 0.35

    ax1.bar(x - width/2, human_stds, width, label='Human',
            color='steelblue', alpha=0.8)
    ax1.bar(x + width/2, llm_stds, width, label='LLM',
            color='coral', alpha=0.8)

    ax1.set_xlabel('Question', fontsize=12, fontweight='bold')
    ax1.set_ylabel('Standard Deviation', fontsize=12, fontweight='bold')
    ax1.set_title('Response Spread Comparison', fontsize=14, fontweight='bold')
    ax1.set_xticks(x)
    ax1.set_xticklabels(questions, rotation=45, ha='right')
    ax1.legend()
    ax1.grid(axis='y', alpha=0.3)

    # Subplot 2: Scatter plot
    ax2.scatter(human_stds, llm_stds, s=100, alpha=0.6, color='purple')

    # Add diagonal line (perfect agreement)
    max_std = max(max(human_stds), max(llm_stds))
    ax2.plot([0, max_std], [0, max_std], 'r--', label='Perfect Agreement')

    # Add labels for each point
    for i, q in enumerate(questions):
        ax2.annotate(q, (human_stds[i], llm_stds[i]),
                     xytext=(5, 5), textcoords='offset points', fontsize=8)

    ax2.set_xlabel('Human Std Dev', fontsize=12, fontweight='bold')
    ax2.set_ylabel('LLM Std Dev', fontsize=12, fontweight='bold')
    ax2.set_title('Spread Correlation', fontsize=14, fontweight='bold')
    ax2.legend()
    ax2.grid(alpha=0.3)

    plt.tight_layout()
    return fig


def plot_comparison_entropy(
    comparisons: Dict[str, ComparisonMetrics],
    figsize: Tuple[int, int] = (12, 6)
) -> plt.Figure:
    """
    Plot comparison of entropy (uncertainty) between human and LLM.

    Args:
        comparisons: Dictionary of ComparisonMetrics
        figsize: Figure size

    Returns:
        Matplotlib figure
    """
    questions = list(comparisons.keys())
    human_entropies = [comparisons[q].human_entropy for q in questions]
    llm_entropies = [comparisons[q].llm_entropy for q in questions]

    fig, ax = plt.subplots(figsize=figsize)

    x = np.arange(len(questions))
    width = 0.35

    bars1 = ax.bar(x - width/2, human_entropies, width, label='Human',
                   color='steelblue', alpha=0.8)
    bars2 = ax.bar(x + width/2, llm_entropies, width, label='LLM',
                   color='coral', alpha=0.8)

    ax.set_xlabel('Question', fontsize=12, fontweight='bold')
    ax.set_ylabel('Mean Entropy (bits)', fontsize=12, fontweight='bold')
    ax.set_title('Response Uncertainty Comparison', fontsize=14, fontweight='bold')
    ax.set_xticks(x)
    ax.set_xticklabels(questions, rotation=45, ha='right')
    ax.legend()
    ax.grid(axis='y', alpha=0.3)

    # Add value labels on bars
    for bars in [bars1, bars2]:
        for bar in bars:
            height = bar.get_height()
            ax.text(bar.get_x() + bar.get_width()/2., height,
                    f'{height:.3f}',
                    ha='center', va='bottom', fontsize=9)

    plt.tight_layout()
    return fig


def print_comparison_summary(comparisons: Dict[str, ComparisonMetrics]) -> None:
    """
    Print a detailed summary of comparisons.

    Args:
        comparisons: Dictionary of ComparisonMetrics
    """
    print("\n" + "=" * 80)
    print("HUMAN vs LLM RESPONSE COMPARISON SUMMARY")
    print("=" * 80)

    for question_id, metrics in comparisons.items():
        print(f"\n{'-' * 80}")
        print(f"Question: {question_id}")
        print(f"{'-' * 80}")

        print("\n📊 MEAN RATINGS:")
        print(f"  Human:      {metrics.human_mean:.3f}")
        print(f"  LLM:        {metrics.llm_mean:.3f}")
        print(f"  Difference: {metrics.mean_difference:+.3f}")
        print(f"  Effect Size (Cohen's d): {metrics.mean_effect_size:.3f}")

        if abs(metrics.mean_effect_size) < 0.2:
            interpretation = "negligible"
        elif abs(metrics.mean_effect_size) < 0.5:
            interpretation = "small"
        elif abs(metrics.mean_effect_size) < 0.8:
            interpretation = "medium"
        else:
            interpretation = "large"
        print(f"  Interpretation: {interpretation} effect")

        print("\n📈 SPREAD (Standard Deviation):")
        print(f"  Human:      {metrics.human_std:.3f}")
        print(f"  LLM:        {metrics.llm_std:.3f}")
        print(f"  Ratio:      {metrics.std_ratio:.3f}")

        if metrics.std_ratio > 1.2:
            spread_msg = "LLM responses are MORE variable"
        elif metrics.std_ratio < 0.8:
            spread_msg = "LLM responses are LESS variable"
        else:
            spread_msg = "Similar variability"
        print(f"  Interpretation: {spread_msg}")

        print("\n🎲 UNCERTAINTY (Entropy):")
        print(f"  Human:      {metrics.human_entropy:.3f} bits")
        print(f"  LLM:        {metrics.llm_entropy:.3f} bits")
        print(f"  Difference: {metrics.entropy_difference:+.3f} bits")

        if metrics.entropy_difference > 0.1:
            entropy_msg = "LLM responses show MORE uncertainty"
        elif metrics.entropy_difference < -0.1:
            entropy_msg = "LLM responses show LESS uncertainty"
        else:
            entropy_msg = "Similar uncertainty levels"
        print(f"  Interpretation: {entropy_msg}")

        print("\n📉 STATISTICAL TESTS:")
        print(f"  t-test:")
        print(f"    Statistic: {metrics.ttest_statistic:.3f}")
        print(f"    p-value:   {metrics.ttest_pvalue:.4f}")

        if metrics.ttest_pvalue < 0.001:
            sig = "*** (highly significant)"
        elif metrics.ttest_pvalue < 0.01:
            sig = "** (very significant)"
        elif metrics.ttest_pvalue < 0.05:
            sig = "* (significant)"
        else:
            sig = "ns (not significant)"
        print(f"    Result:    {sig}")

        print(f"\n  Kolmogorov-Smirnov test:")
        print(f"    Statistic: {metrics.ks_statistic:.3f}")
        print(f"    p-value:   {metrics.ks_pvalue:.4f}")

        if metrics.ks_pvalue < 0.05:
            ks_msg = "Distributions are DIFFERENT"
        else:
            ks_msg = "Distributions are SIMILAR"
        print(f"    Result:    {ks_msg}")

    print("\n" + "=" * 80)


def create_comparison_report(
    comparisons: Dict[str, ComparisonMetrics],
    human_dists: List[RatingDistribution],
    llm_dists: List[RatingDistribution],
    output_dir: str = "results/comparison"
) -> None:
    """
    Create a comprehensive comparison report.

    Args:
        comparisons: Dictionary of ComparisonMetrics
        human_dists: All human distributions
        llm_dists: All LLM distributions
        output_dir: Directory to save figures
    """
    from pathlib import Path

    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)

    print(f"\nGenerating comparison report in {output_dir}/...")

    # Generate all plots
    fig = plot_comparison_means(comparisons)
    plt.savefig(output_path / "comparison_means.png", dpi=300, bbox_inches='tight')
    plt.close()

    fig = plot_comparison_distributions(comparisons, human_dists, llm_dists)
    plt.savefig(output_path / "comparison_distributions.png", dpi=300, bbox_inches='tight')
    plt.close()

    fig = plot_comparison_spread(comparisons)
    plt.savefig(output_path / "comparison_spread.png", dpi=300, bbox_inches='tight')
    plt.close()

    fig = plot_comparison_entropy(comparisons)
    plt.savefig(output_path / "comparison_entropy.png", dpi=300, bbox_inches='tight')
    plt.close()

    # Save comparison metrics to CSV
    records = []
    for q_id, metrics in comparisons.items():
        records.append({
            'question_id': q_id,
            'human_mean': metrics.human_mean,
            'llm_mean': metrics.llm_mean,
            'mean_difference': metrics.mean_difference,
            'effect_size': metrics.mean_effect_size,
            'human_std': metrics.human_std,
            'llm_std': metrics.llm_std,
            'std_ratio': metrics.std_ratio,
            'human_entropy': metrics.human_entropy,
            'llm_entropy': metrics.llm_entropy,
            'entropy_difference': metrics.entropy_difference,
            'ks_statistic': metrics.ks_statistic,
            'ks_pvalue': metrics.ks_pvalue,
            'ttest_statistic': metrics.ttest_statistic,
            'ttest_pvalue': metrics.ttest_pvalue
        })

    df = pd.DataFrame(records)
    df.to_csv(output_path / "comparison_metrics.csv", index=False)

    print(f"Comparison report generated successfully!")
    print(f"  • Saved 4 figures")
    print(f"  • Saved comparison_metrics.csv")
