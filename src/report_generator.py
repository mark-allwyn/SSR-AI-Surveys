"""One-page report generator for ground truth comparison."""

from typing import Dict, List
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from matplotlib.gridspec import GridSpec
import seaborn as sns
import numpy as np
import pandas as pd
from pathlib import Path

from .ground_truth import GroundTruthComparison
from .survey import Survey


def create_one_page_report(
    human_comparisons: Dict[str, GroundTruthComparison],
    llm_comparisons: Dict[str, GroundTruthComparison],
    survey: Survey,
    output_path: str = "results/ground_truth_report.png",
    title: str = "Human vs LLM Ground Truth Comparison"
):
    """
    Create a comprehensive one-page report comparing human and LLM performance.

    Args:
        human_comparisons: Dict mapping question_id to human GroundTruthComparison
        llm_comparisons: Dict mapping question_id to LLM GroundTruthComparison
        survey: Survey object
        output_path: Path to save the report
        title: Report title
    """
    # Set up the figure with custom layout
    fig = plt.figure(figsize=(18, 12))
    fig.suptitle(title, fontsize=22, fontweight='bold', y=0.98)

    # Create grid layout with more spacing
    gs = GridSpec(4, 3, figure=fig, hspace=0.5, wspace=0.35,
                  left=0.08, right=0.95, top=0.94, bottom=0.06)

    question_ids = list(human_comparisons.keys())

    # 1. Overall Accuracy Comparison (top left - wide)
    ax1 = fig.add_subplot(gs[0, :2])
    plot_accuracy_comparison(ax1, human_comparisons, llm_comparisons, question_ids)

    # 2. Error Metrics (top right)
    ax2 = fig.add_subplot(gs[0, 2])
    plot_error_summary(ax2, human_comparisons, llm_comparisons)

    # 3. Confusion Matrices (middle rows - show first 2 questions only to fit layout)
    max_questions_to_show = min(2, len(question_ids))

    for idx, q_id in enumerate(question_ids[:max_questions_to_show]):
        if idx == 0:
            # First question - row 1, columns 0-1
            ax_h = fig.add_subplot(gs[1, 0])
            ax_l = fig.add_subplot(gs[1, 1])
        else:
            # Second question - row 2, columns 0-1
            ax_h = fig.add_subplot(gs[2, 0])
            ax_l = fig.add_subplot(gs[2, 1])

        plot_confusion_matrix(ax_h, human_comparisons[q_id], f"GT: {q_id.replace('_', ' ')[:15]}")
        plot_confusion_matrix(ax_l, llm_comparisons[q_id], f"LLM: {q_id.replace('_', ' ')[:15]}")

    # 5. Summary Table (bottom)
    ax_table = fig.add_subplot(gs[3, :])
    plot_summary_table(ax_table, human_comparisons, llm_comparisons, question_ids)

    # Save figure
    Path(output_path).parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    plt.close()

    print(f"\n✓ One-page report saved to: {output_path}")


def plot_accuracy_comparison(ax, human_comps, llm_comps, question_ids):
    """Plot grouped bar chart comparing accuracies."""
    x = np.arange(len(question_ids))
    width = 0.35

    human_mode_acc = [human_comps[q].mode_accuracy * 100 for q in question_ids]
    llm_mode_acc = [llm_comps[q].mode_accuracy * 100 for q in question_ids]

    bars1 = ax.bar(x - width/2, human_mode_acc, width, label='Ground Truth',
                   color='green', alpha=0.7)
    bars2 = ax.bar(x + width/2, llm_mode_acc, width, label='LLM+SSR',
                   color='coral', alpha=0.8)

    ax.set_ylabel('Mode Accuracy (%)', fontsize=13, fontweight='bold')
    ax.set_title('Accuracy by Question', fontsize=15, fontweight='bold', pad=15)
    ax.set_xticks(x)
    ax.set_xticklabels([q.replace('_', '\n') for q in question_ids], fontsize=10)
    ax.legend(fontsize=12, loc='upper right')
    ax.set_ylim(0, 110)  # Extra space for labels
    ax.grid(axis='y', alpha=0.3)

    # Add value labels on bars
    for bars in [bars1, bars2]:
        for bar in bars:
            height = bar.get_height()
            ax.text(bar.get_x() + bar.get_width()/2., height + 1,
                    f'{height:.1f}%',
                    ha='center', va='bottom', fontsize=9, fontweight='bold')


def plot_error_summary(ax, human_comps, llm_comps):
    """Plot overall error statistics."""
    # Calculate averages
    human_mae_avg = np.mean([c.mae for c in human_comps.values()])
    llm_mae_avg = np.mean([c.mae for c in llm_comps.values()])

    human_rmse_avg = np.mean([c.rmse for c in human_comps.values()])
    llm_rmse_avg = np.mean([c.rmse for c in llm_comps.values()])

    metrics = ['MAE', 'RMSE']
    human_vals = [human_mae_avg, human_rmse_avg]
    llm_vals = [llm_mae_avg, llm_rmse_avg]

    x = np.arange(len(metrics))
    width = 0.35

    ax.bar(x - width/2, human_vals, width, label='Ground Truth',
           color='green', alpha=0.7)
    ax.bar(x + width/2, llm_vals, width, label='LLM+SSR',
           color='coral', alpha=0.8)

    ax.set_ylabel('Error', fontsize=12, fontweight='bold')
    ax.set_title('Average Error Metrics', fontsize=15, fontweight='bold', pad=15)
    ax.set_xticks(x)
    ax.set_xticklabels(metrics, fontsize=12)
    ax.legend(fontsize=11, loc='upper right')
    ax.grid(axis='y', alpha=0.3)

    # Set y-axis limit with padding for labels
    max_val = max(max(human_vals), max(llm_vals))
    ax.set_ylim(0, max_val * 1.2)

    # Add value labels
    for i, (h, l) in enumerate(zip(human_vals, llm_vals)):
        ax.text(i - width/2, h + max_val*0.02, f'{h:.2f}', ha='center', va='bottom', fontsize=10, fontweight='bold')
        ax.text(i + width/2, l + max_val*0.02, f'{l:.2f}', ha='center', va='bottom', fontsize=10, fontweight='bold')


def plot_confusion_matrix(ax, comparison, title):
    """Plot a single confusion matrix."""
    cm = comparison.confusion_matrix
    n = cm.shape[0]

    # Normalize by row
    cm_norm = cm / cm.sum(axis=1, keepdims=True)
    cm_norm = np.nan_to_num(cm_norm)

    # Plot heatmap
    im = ax.imshow(cm_norm, cmap='Blues', aspect='auto', vmin=0, vmax=1)

    # Set ticks
    ax.set_xticks(np.arange(n))
    ax.set_yticks(np.arange(n))
    ax.set_xticklabels(np.arange(1, n+1), fontsize=10)
    ax.set_yticklabels(np.arange(1, n+1), fontsize=10)

    # Add labels with more padding
    ax.set_xlabel('Predicted', fontsize=11, fontweight='bold', labelpad=8)
    ax.set_ylabel('True', fontsize=11, fontweight='bold', labelpad=8)
    ax.set_title(title, fontsize=12, fontweight='bold', pad=12)

    # Add text annotations
    for i in range(n):
        for j in range(n):
            count = int(cm[i, j])
            if count > 0:
                text = ax.text(j, i, f'{count}',
                              ha="center", va="center", color="black" if cm_norm[i, j] < 0.5 else "white",
                              fontsize=10, fontweight='bold')

    # Add colorbar
    cbar = plt.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
    cbar.ax.tick_params(labelsize=9)


def plot_summary_table(ax, human_comps, llm_comps, question_ids):
    """Plot summary table with key metrics."""
    ax.axis('off')

    # Prepare data
    rows = []
    for q_id in question_ids:
        h = human_comps[q_id]
        l = llm_comps[q_id]

        row = [
            q_id.replace('_', ' '),
            f"{h.mode_accuracy:.1%} / {l.mode_accuracy:.1%}",
            f"{h.mae:.2f} / {l.mae:.2f}",
            f"{h.mean_probability_at_truth:.2f} / {l.mean_probability_at_truth:.2f}",
            "✓" if h.mode_accuracy >= l.mode_accuracy else "✗",
            f"{h.n_samples}"
        ]
        rows.append(row)

    # Add summary row
    h_avg_acc = np.mean([h.mode_accuracy for h in human_comps.values()])
    l_avg_acc = np.mean([l.mode_accuracy for l in llm_comps.values()])
    h_avg_mae = np.mean([h.mae for h in human_comps.values()])
    l_avg_mae = np.mean([l.mae for l in llm_comps.values()])
    h_avg_prob = np.mean([h.mean_probability_at_truth for h in human_comps.values()])
    l_avg_prob = np.mean([l.mean_probability_at_truth for l in llm_comps.values()])
    total_samples = sum([h.n_samples for h in human_comps.values()])

    summary_row = [
        "AVERAGE",
        f"{h_avg_acc:.1%} / {l_avg_acc:.1%}",
        f"{h_avg_mae:.2f} / {l_avg_mae:.2f}",
        f"{h_avg_prob:.2f} / {l_avg_prob:.2f}",
        "✓" if h_avg_acc >= l_avg_acc else "✗",
        f"{total_samples}"
    ]
    rows.append(summary_row)

    # Create table with better column headers
    columns = ['Question', 'Accuracy\n(GT/LLM)', 'MAE\n(GT/LLM)', 'Prob@Truth\n(GT/LLM)', 'GT Wins?', 'N']

    table = ax.table(cellText=rows, colLabels=columns,
                     cellLoc='center', loc='center',
                     bbox=[0, 0, 1, 1])

    table.auto_set_font_size(False)
    table.set_fontsize(10)

    # Adjust column widths for better fit
    table.auto_set_column_width(col=list(range(len(columns))))

    # Style table
    for i in range(len(columns)):
        table[(0, i)].set_facecolor('#4CAF50')
        table[(0, i)].set_text_props(weight='bold', color='white')

    # Style summary row
    for i in range(len(columns)):
        table[(len(rows), i)].set_facecolor('#FFE082')
        table[(len(rows), i)].set_text_props(weight='bold')

    # Alternate row colors
    for i in range(1, len(rows)):
        for j in range(len(columns)):
            if i % 2 == 0:
                table[(i, j)].set_facecolor('#F5F5F5')

    ax.set_title('Summary: Ground Truth (GT) vs LLM+SSR Performance', fontsize=15,
                 fontweight='bold', pad=15)


def generate_text_report(
    human_comparisons: Dict[str, GroundTruthComparison],
    llm_comparisons: Dict[str, GroundTruthComparison],
    survey: Survey,
    output_path: str = "results/ground_truth_report.txt"
):
    """Generate a text-based report."""
    Path(output_path).parent.mkdir(parents=True, exist_ok=True)

    with open(output_path, 'w') as f:
        f.write("="*80 + "\n")
        f.write("GROUND TRUTH COMPARISON REPORT: HUMAN GROUND TRUTH VS LLM+SSR\n")
        f.write("="*80 + "\n\n")

        f.write(f"Survey: {survey.name}\n")
        f.write(f"Questions Analyzed: {len(human_comparisons)}\n\n")

        # Overall summary
        f.write("-"*80 + "\n")
        f.write("OVERALL SUMMARY\n")
        f.write("-"*80 + "\n\n")

        h_avg_acc = np.mean([h.mode_accuracy for h in human_comparisons.values()])
        l_avg_acc = np.mean([l.mode_accuracy for l in llm_comparisons.values()])
        h_avg_mae = np.mean([h.mae for h in human_comparisons.values()])
        l_avg_mae = np.mean([l.mae for l in llm_comparisons.values()])

        f.write(f"Average Mode Accuracy:\n")
        f.write(f"  Ground Truth (Human): {h_avg_acc:.1%}\n")
        f.write(f"  LLM+SSR:              {l_avg_acc:.1%}\n")
        f.write(f"  Gap:                  {(h_avg_acc - l_avg_acc):.1%}\n\n")

        f.write(f"Average MAE:\n")
        f.write(f"  Ground Truth (Human): {h_avg_mae:.3f}\n")
        f.write(f"  LLM+SSR:              {l_avg_mae:.3f}\n\n")

        # Per-question details
        for q_id in human_comparisons.keys():
            h = human_comparisons[q_id]
            l = llm_comparisons[q_id]

            f.write("-"*80 + "\n")
            f.write(f"QUESTION: {q_id}\n")
            f.write("-"*80 + "\n\n")

            f.write(f"Question Type: {h.question_type}\n")
            f.write(f"Sample Size: {h.n_samples}\n\n")

            f.write(f"Mode Accuracy:      Ground Truth: {h.mode_accuracy:.1%}  |  LLM+SSR: {l.mode_accuracy:.1%}\n")
            f.write(f"Top-2 Accuracy:     Ground Truth: {h.top2_accuracy:.1%}  |  LLM+SSR: {l.top2_accuracy:.1%}\n")
            f.write(f"MAE:                Ground Truth: {h.mae:.3f}  |  LLM+SSR: {l.mae:.3f}\n")
            f.write(f"RMSE:               Ground Truth: {h.rmse:.3f}  |  LLM+SSR: {l.rmse:.3f}\n")
            f.write(f"Prob at Truth:      Ground Truth: {h.mean_probability_at_truth:.3f}  |  LLM+SSR: {l.mean_probability_at_truth:.3f}\n")
            f.write(f"KL Divergence:      Ground Truth: {h.kl_divergence:.4f}  |  LLM+SSR: {l.kl_divergence:.4f}\n\n")

        f.write("="*80 + "\n")
        f.write("END OF REPORT\n")
        f.write("="*80 + "\n")

    print(f"✓ Text report saved to: {output_path}")
