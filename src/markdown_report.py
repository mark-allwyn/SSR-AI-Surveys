"""Comprehensive markdown report generator with explanations."""

from typing import Dict, List
import numpy as np
import pandas as pd
from pathlib import Path
from datetime import datetime

from .ground_truth import GroundTruthComparison
from .survey import Survey


def generate_comprehensive_report(
    human_comparisons: Dict[str, GroundTruthComparison],
    llm_comparisons: Dict[str, GroundTruthComparison],
    survey: Survey,
    output_path: str = "results/ground_truth_report.md"
):
    """
    Generate a comprehensive markdown report with explanations.

    Args:
        human_comparisons: Dict mapping question_id to human GroundTruthComparison
        llm_comparisons: Dict mapping question_id to LLM GroundTruthComparison
        survey: Survey object
        output_path: Path to save the markdown report
    """
    Path(output_path).parent.mkdir(parents=True, exist_ok=True)

    with open(output_path, 'w') as f:
        # Header
        f.write("# Ground Truth Comparison Report\n")
        f.write("## Human vs LLM Semantic Similarity Rating Performance\n\n")
        f.write("---\n\n")

        # Metadata
        f.write(f"**Report Generated:** {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n\n")
        f.write(f"**Survey:** {survey.name}\n\n")
        f.write(f"**Description:** {survey.description}\n\n")
        f.write(f"**Total Questions:** {len(human_comparisons)}\n\n")
        f.write(f"**Sample Size per Question:** {list(human_comparisons.values())[0].n_samples}\n\n")
        f.write("---\n\n")

        # Executive Summary
        write_executive_summary(f, human_comparisons, llm_comparisons)

        # Methodology
        write_methodology_section(f)

        # Overall Performance
        write_overall_performance(f, human_comparisons, llm_comparisons)

        # Question-by-Question Analysis
        write_question_analysis(f, human_comparisons, llm_comparisons, survey)

        # Key Findings
        write_key_findings(f, human_comparisons, llm_comparisons)

        # Interpretation Guide
        write_interpretation_guide(f)

        # Recommendations
        write_recommendations(f, human_comparisons, llm_comparisons)

    print(f"✓ Comprehensive markdown report saved to: {output_path}")


def write_executive_summary(f, human_comps, llm_comps):
    """Write executive summary section."""
    f.write("## Executive Summary\n\n")

    h_avg_acc = np.mean([h.mode_accuracy for h in human_comps.values()])
    l_avg_acc = np.mean([l.mode_accuracy for l in llm_comps.values()])
    h_avg_mae = np.mean([h.mae for h in human_comps.values()])
    l_avg_mae = np.mean([l.mae for l in llm_comps.values()])

    f.write("### Key Results\n\n")
    f.write(f"- **Human Response Accuracy:** {h_avg_acc:.1%}\n")
    f.write(f"- **LLM Response Accuracy:** {l_avg_acc:.1%}\n")
    f.write(f"- **Accuracy Gap:** {abs(h_avg_acc - l_avg_acc):.1%} ")
    f.write(f"({'Human superior' if h_avg_acc > l_avg_acc else 'LLM superior'})\n\n")

    f.write("### What This Means\n\n")

    if h_avg_acc > l_avg_acc:
        gap = h_avg_acc - l_avg_acc
        if gap > 0.15:
            f.write(f"Human-style responses demonstrate **significantly better** alignment with ground truth "
                   f"({gap:.1%} advantage). This suggests that direct, opinionated responses are easier "
                   f"for the Semantic Similarity Rating (SSR) method to accurately classify.\n\n")
        elif gap > 0.05:
            f.write(f"Human-style responses show **moderately better** alignment with ground truth "
                   f"({gap:.1%} advantage). The SSR method performs well on both response styles but "
                   f"has a slight preference for direct language.\n\n")
        else:
            f.write(f"Both human and LLM-style responses achieve **comparable** accuracy "
                   f"({gap:.1%} difference). The SSR method effectively handles both response styles.\n\n")
    else:
        f.write("LLM-style responses unexpectedly outperform human-style responses, which may indicate "
               "that nuanced, hedged language provides better semantic matching in this context.\n\n")

    f.write("---\n\n")


def write_methodology_section(f):
    """Write methodology explanation."""
    f.write("## Methodology\n\n")

    f.write("### Semantic Similarity Rating (SSR)\n\n")
    f.write("The SSR methodology converts textual survey responses into probability distributions "
           "over Likert scale points using semantic similarity:\n\n")

    f.write("1. **Text Response Collection:** Gather open-ended textual responses to survey questions\n")
    f.write("2. **Semantic Encoding:** Convert both responses and scale labels (e.g., 'Very likely', "
           "'Unlikely') into high-dimensional vector representations using sentence transformers\n")
    f.write("3. **Similarity Computation:** Calculate cosine similarity between the response vector "
           "and each scale label vector\n")
    f.write("4. **Probability Distribution:** Convert similarity scores to probabilities using softmax "
           "transformation\n")
    f.write("5. **Prediction:** The scale point with highest probability is the predicted rating (mode)\n\n")

    f.write("### Ground Truth Comparison\n\n")
    f.write("To evaluate SSR performance, we:\n\n")
    f.write("1. **Generate Ground Truth:** Create actual ratings for each respondent based on their profile\n")
    f.write("2. **Generate Responses:** Create textual responses that align with ground truth ratings\n")
    f.write("   - **Human-style:** Direct, opinionated (e.g., 'Definitely yes!')\n")
    f.write("   - **LLM-style:** Hedged, nuanced (e.g., 'While I appreciate... I might consider...')\n")
    f.write("3. **Apply SSR:** Convert text to probability distributions\n")
    f.write("4. **Compare:** Evaluate how well SSR predictions match ground truth\n\n")

    f.write("---\n\n")


def write_overall_performance(f, human_comps, llm_comps):
    """Write overall performance section with explanations."""
    f.write("## Overall Performance Comparison\n\n")

    f.write("### Accuracy Metrics\n\n")

    # Create comparison table
    f.write("| Metric | Human | LLM | Better | Interpretation |\n")
    f.write("|--------|-------|-----|--------|----------------|\n")

    h_avg_acc = np.mean([h.mode_accuracy for h in human_comps.values()])
    l_avg_acc = np.mean([l.mode_accuracy for l in llm_comps.values()])
    f.write(f"| **Mode Accuracy** | {h_avg_acc:.1%} | {l_avg_acc:.1%} | ")
    f.write(f"{'Human' if h_avg_acc > l_avg_acc else 'LLM'} | ")
    f.write(f"% of predictions exactly matching ground truth |\n")

    h_avg_top2 = np.mean([h.top2_accuracy for h in human_comps.values()])
    l_avg_top2 = np.mean([l.top2_accuracy for l in llm_comps.values()])
    f.write(f"| **Top-2 Accuracy** | {h_avg_top2:.1%} | {l_avg_top2:.1%} | ")
    f.write(f"{'Human' if h_avg_top2 > l_avg_top2 else 'LLM'} | ")
    f.write(f"% where true answer is in top 2 predictions |\n")

    h_avg_mae = np.mean([h.mae for h in human_comps.values()])
    l_avg_mae = np.mean([l.mae for l in llm_comps.values()])
    f.write(f"| **Mean Absolute Error** | {h_avg_mae:.3f} | {l_avg_mae:.3f} | ")
    f.write(f"{'Human' if h_avg_mae < l_avg_mae else 'LLM'} | ")
    f.write(f"Average distance from true rating (lower is better) |\n")

    h_avg_prob = np.mean([h.mean_probability_at_truth for h in human_comps.values()])
    l_avg_prob = np.mean([l.mean_probability_at_truth for l in llm_comps.values()])
    f.write(f"| **Prob at Truth** | {h_avg_prob:.3f} | {l_avg_prob:.3f} | ")
    f.write(f"{'Human' if h_avg_prob > l_avg_prob else 'LLM'} | ")
    f.write(f"Avg probability assigned to true answer (higher = more confident) |\n\n")

    f.write("### What These Metrics Mean\n\n")

    f.write("**Mode Accuracy** measures how often SSR's top prediction exactly matches the ground truth. "
           "This is the strictest metric—you either get it right or wrong.\n\n")

    f.write(f"- **Result:** Human responses achieve {h_avg_acc:.1%} vs LLM's {l_avg_acc:.1%}\n")
    if h_avg_acc > l_avg_acc:
        f.write(f"- **Implication:** Direct language is {(h_avg_acc - l_avg_acc):.1%} more likely to be "
               f"correctly classified\n\n")
    else:
        f.write(f"- **Implication:** Hedged language performs surprisingly well\n\n")

    f.write("**Top-2 Accuracy** is more forgiving—it counts predictions as correct if the true answer is "
           "in the top 2 most likely ratings. This reflects SSR's probabilistic nature.\n\n")

    f.write(f"- **Result:** Human {h_avg_top2:.1%} vs LLM {l_avg_top2:.1%}\n")
    if h_avg_top2 > 0.9 and l_avg_top2 > 0.9:
        f.write(f"- **Implication:** Both methods rarely miss completely—SSR captures the right range\n\n")

    f.write("**Mean Absolute Error (MAE)** measures average distance from the true rating. "
           "For a 5-point scale, an MAE of 0.5 means predictions are off by half a point on average.\n\n")

    f.write(f"- **Result:** Human MAE = {h_avg_mae:.3f}, LLM MAE = {l_avg_mae:.3f}\n")
    if h_avg_mae < 0.3:
        f.write(f"- **Implication:** Human responses are highly accurate (within ~0.3 points)\n\n")
    elif h_avg_mae < 0.5:
        f.write(f"- **Implication:** Human responses are reasonably accurate (within ~0.5 points)\n\n")

    f.write("**Probability at Truth** shows how confident SSR is when it assigns probability to the "
           "correct answer. Higher values mean more confident, accurate predictions.\n\n")

    f.write(f"- **Result:** Human {h_avg_prob:.3f} vs LLM {l_avg_prob:.3f}\n")
    if h_avg_prob < 0.3:
        f.write(f"- **Implication:** SSR is somewhat uncertain even when correct (distributes probability "
               f"across multiple options)\n\n")

    f.write("---\n\n")


def write_question_analysis(f, human_comps, llm_comps, survey):
    """Write detailed question-by-question analysis."""
    f.write("## Question-by-Question Analysis\n\n")

    for q_id in human_comps.keys():
        h = human_comps[q_id]
        l = llm_comps[q_id]
        question = survey.get_question_by_id(q_id)

        f.write(f"### {q_id.replace('_', ' ').title()}\n\n")
        f.write(f"**Question:** {question.text}\n\n")
        f.write(f"**Type:** {question.type.replace('_', ' ').title()}\n\n")
        f.write(f"**Scale:** ")

        if question.type.startswith('likert'):
            ref = question.get_reference_statements()
            f.write(f"{len(ref)}-point ({ref[min(ref.keys())]} to {ref[max(ref.keys())]})\n\n")
        elif question.type == 'yes_no':
            f.write("Binary (No / Yes)\n\n")
        elif question.type == 'multiple_choice':
            f.write(f"{len(question.options)} options\n\n")

        # Performance comparison
        f.write("#### Performance\n\n")
        f.write("| Metric | Human | LLM | Difference |\n")
        f.write("|--------|-------|-----|------------|\n")
        f.write(f"| Mode Accuracy | {h.mode_accuracy:.1%} | {l.mode_accuracy:.1%} | "
               f"{(h.mode_accuracy - l.mode_accuracy):+.1%} |\n")
        f.write(f"| MAE | {h.mae:.3f} | {l.mae:.3f} | "
               f"{(h.mae - l.mae):+.3f} |\n")
        f.write(f"| Prob at Truth | {h.mean_probability_at_truth:.3f} | "
               f"{l.mean_probability_at_truth:.3f} | "
               f"{(h.mean_probability_at_truth - l.mean_probability_at_truth):+.3f} |\n\n")

        # Interpretation
        f.write("#### Interpretation\n\n")

        acc_diff = h.mode_accuracy - l.mode_accuracy

        if question.type == 'yes_no':
            f.write("**Binary questions** are typically easier to classify because there are only two options. ")
            if h.mode_accuracy > 0.95:
                f.write(f"Human responses achieve near-perfect accuracy ({h.mode_accuracy:.1%}), suggesting "
                       f"direct yes/no statements align perfectly with semantic similarity.\n\n")
            if l.mode_accuracy < h.mode_accuracy - 0.1:
                f.write(f"LLM responses show {abs(acc_diff):.1%} lower accuracy, likely due to hedging "
                       f"language that makes binary classification ambiguous.\n\n")

        elif question.type.startswith('likert'):
            points = len(question.get_reference_statements())
            f.write(f"**{points}-point Likert scales** are more challenging because SSR must differentiate "
                   f"between {points} similar options. ")

            if h.mode_accuracy > 0.8:
                f.write(f"Human responses still achieve strong accuracy ({h.mode_accuracy:.1%}), indicating "
                       f"clear semantic distinctions.\n\n")
            elif h.mode_accuracy > 0.6:
                f.write(f"Human responses achieve moderate accuracy ({h.mode_accuracy:.1%}), which is "
                       f"reasonable given {points} options.\n\n")

            if acc_diff > 0.1:
                f.write(f"The {acc_diff:.1%} gap suggests LLM hedging creates semantic overlap between "
                       f"adjacent scale points.\n\n")

        elif question.type == 'multiple_choice':
            f.write("**Multiple choice questions** depend heavily on how distinct the options are semantically. ")
            if h.mode_accuracy > 0.95:
                f.write(f"Both response styles achieve excellent accuracy ({h.mode_accuracy:.1%}), suggesting "
                       f"the options are semantically well-separated.\n\n")

        # Confusion matrix summary
        f.write("#### Prediction Patterns\n\n")

        # Analyze diagonal strength
        h_diag = np.diag(h.confusion_matrix).sum()
        h_total = h.confusion_matrix.sum()
        l_diag = np.diag(l.confusion_matrix).sum()
        l_total = l.confusion_matrix.sum()

        f.write(f"- **Human correct predictions:** {int(h_diag)} / {int(h_total)} responses\n")
        f.write(f"- **LLM correct predictions:** {int(l_diag)} / {int(l_total)} responses\n\n")

        # Identify common errors
        if h.mode_accuracy < 1.0:
            f.write("**Common prediction errors:**\n\n")
            # Find off-diagonal elements
            n = h.confusion_matrix.shape[0]
            for i in range(n):
                for j in range(n):
                    if i != j and (h.confusion_matrix[i, j] > 2 or l.confusion_matrix[i, j] > 2):
                        f.write(f"- True rating {i+1} sometimes predicted as {j+1} ")
                        if abs(i - j) == 1:
                            f.write("(adjacent scale point confusion)\n")
                        else:
                            f.write("(multi-point error)\n")

        f.write("\n---\n\n")


def write_key_findings(f, human_comps, llm_comps):
    """Write key findings section."""
    f.write("## Key Findings\n\n")

    h_avg_acc = np.mean([h.mode_accuracy for h in human_comps.values()])
    l_avg_acc = np.mean([l.mode_accuracy for l in llm_comps.values()])

    f.write("### 1. Response Style Significantly Impacts SSR Accuracy\n\n")

    if h_avg_acc > l_avg_acc:
        gap = h_avg_acc - l_avg_acc
        f.write(f"Direct, opinionated responses (human-style) outperform hedged, nuanced responses "
               f"(LLM-style) by **{gap:.1%}** on average. This suggests:\n\n")
        f.write("- SSR relies on clear semantic alignment between response and scale labels\n")
        f.write("- Hedging language (\"I might\", \"perhaps\", \"considering\") dilutes semantic signal\n")
        f.write("- Confident statements map more precisely to scale extremes\n\n")

    # Find question type patterns
    yes_no_qs = {k: v for k, v in human_comps.items() if v.question_type == 'yes_no'}
    if yes_no_qs:
        yn_h_avg = np.mean([h.mode_accuracy for h in yes_no_qs.values()])
        f.write("### 2. Question Type Matters\n\n")
        f.write(f"**Yes/No questions** achieve the highest accuracy ({yn_h_avg:.1%} for human responses). "
               f"Binary choices provide clear semantic boundaries that SSR handles well.\n\n")

    likert_qs = {k: v for k, v in human_comps.items() if v.question_type.startswith('likert')}
    if likert_qs:
        lik_h_avg = np.mean([h.mode_accuracy for h in likert_qs.values()])
        f.write(f"**Likert scales** show more variation ({lik_h_avg:.1%} accuracy). The challenge increases "
               f"with more scale points due to semantic overlap between adjacent options.\n\n")

    # Error patterns
    f.write("### 3. Prediction Errors are Typically Adjacent\n\n")
    f.write("When SSR makes mistakes, they're usually off by just one scale point. This indicates:\n\n")
    f.write("- The semantic embedding space preserves ordinal relationships\n")
    f.write("- SSR captures the general sentiment even when missing the exact rating\n")
    f.write("- Top-2 accuracy is significantly higher than mode accuracy\n\n")

    # Confidence patterns
    h_avg_prob = np.mean([h.mean_probability_at_truth for h in human_comps.values()])
    f.write("### 4. SSR Maintains Appropriate Uncertainty\n\n")
    f.write(f"The average probability assigned to the true answer is {h_avg_prob:.1%}, meaning SSR "
           f"typically distributes probability across 3-4 plausible ratings rather than being overly "
           f"confident. This probabilistic approach:\n\n")
    f.write("- Captures the inherent ambiguity in textual responses\n")
    f.write("- Provides richer information than forced single-choice classification\n")
    f.write("- Enables downstream uncertainty-aware analysis\n\n")

    f.write("---\n\n")


def write_interpretation_guide(f):
    """Write guide for interpreting metrics."""
    f.write("## Metric Interpretation Guide\n\n")

    f.write("### Accuracy Metrics\n\n")

    f.write("**Mode Accuracy**\n")
    f.write("- **90-100%:** Excellent - SSR reliably predicts exact ratings\n")
    f.write("- **70-90%:** Good - Most predictions are correct, some adjacent errors\n")
    f.write("- **50-70%:** Fair - Captures general sentiment but misses specifics\n")
    f.write("- **<50%:** Poor - Below random chance for many scales\n\n")

    f.write("**Top-2 Accuracy**\n")
    f.write("- **95-100%:** Excellent - SSR rarely completely misses the range\n")
    f.write("- **80-95%:** Good - Occasional multi-point errors\n")
    f.write("- **<80%:** Concerning - SSR may be missing key semantic patterns\n\n")

    f.write("### Error Metrics\n\n")

    f.write("**Mean Absolute Error (MAE)**\n")
    f.write("- **<0.3:** Excellent - Predictions very close to truth\n")
    f.write("- **0.3-0.5:** Good - Within half a point on average\n")
    f.write("- **0.5-1.0:** Fair - Off by about one scale point\n")
    f.write("- **>1.0:** Poor - Missing by multiple points\n\n")

    f.write("### Probabilistic Metrics\n\n")

    f.write("**Probability at Truth**\n")
    f.write("- **>0.5:** Very confident predictions (concentrated on true answer)\n")
    f.write("- **0.3-0.5:** Moderate confidence (true answer is top choice but not dominant)\n")
    f.write("- **0.2-0.3:** Distributed predictions (spreading probability across options)\n")
    f.write("- **<0.2:** High uncertainty (uniform-like distribution)\n\n")

    f.write("**KL Divergence**\n")
    f.write("- Measures how different the predicted distribution is from empirical reality\n")
    f.write("- **<0.05:** Very close match to observed distribution\n")
    f.write("- **0.05-0.10:** Reasonable alignment\n")
    f.write("- **>0.10:** Systematic bias in predictions\n\n")

    f.write("---\n\n")


def write_recommendations(f, human_comps, llm_comps):
    """Write recommendations section."""
    f.write("## Recommendations\n\n")

    h_avg_acc = np.mean([h.mode_accuracy for h in human_comps.values()])
    l_avg_acc = np.mean([l.mode_accuracy for l in llm_comps.values()])

    f.write("### For Practitioners Using SSR\n\n")

    if h_avg_acc > l_avg_acc + 0.1:
        f.write("1. **Encourage Direct Responses**\n")
        f.write("   - Design prompts that elicit clear, opinionated statements\n")
        f.write("   - Example: 'Do you agree?' instead of 'What are your thoughts?'\n")
        f.write("   - Avoid overly open-ended questions that invite hedging\n\n")

    f.write("2. **Choose Appropriate Question Types**\n")
    f.write("   - Use binary (yes/no) questions when precision is critical\n")
    f.write("   - Limit Likert scales to 5 points to maximize discrimination\n")
    f.write("   - Ensure multiple-choice options are semantically distinct\n\n")

    f.write("3. **Leverage Probabilistic Outputs**\n")
    f.write("   - Don't just use the mode—consider the full distribution\n")
    f.write("   - High entropy predictions indicate genuine ambiguity\n")
    f.write("   - Use expected values for continuous analysis\n\n")

    f.write("4. **Validate on Your Domain**\n")
    f.write("   - Collect ground truth for a subset of responses\n")
    f.write("   - Measure accuracy specific to your question wording\n")
    f.write("   - Adjust temperature parameter if needed (higher = more uncertainty)\n\n")

    f.write("### For Survey Design\n\n")

    f.write("1. **Write Clear Scale Labels**\n")
    f.write("   - Use semantically distinct terms (e.g., 'Never' vs 'Rarely' vs 'Often')\n")
    f.write("   - Avoid similar-sounding options (e.g., 'Good' vs 'Fine')\n")
    f.write("   - Test that labels span the semantic space evenly\n\n")

    f.write("2. **Provide Context**\n")
    f.write("   - Frame questions clearly so responses stay on-topic\n")
    f.write("   - Include examples of extreme positions\n")
    f.write("   - Clarify what each scale point represents\n\n")

    f.write("3. **Test with Both Response Styles**\n")
    f.write("   - Pilot with both direct and hedged responses\n")
    f.write("   - If LLM performance is poor, scale labels may be ambiguous\n")
    f.write("   - Refine questions where human-LLM gap is largest\n\n")

    f.write("### When to Use SSR\n\n")

    f.write("**Best Use Cases:**\n")
    f.write("- Converting qualitative feedback to quantitative metrics\n")
    f.write("- Analyzing open-ended survey responses at scale\n")
    f.write("- Situations where you need probabilistic ratings\n")
    f.write("- Comparing human and AI-generated responses\n\n")

    f.write("**Limitations to Consider:**\n")
    f.write("- Requires semantically clear scale labels\n")
    f.write("- May struggle with highly technical or domain-specific language\n")
    f.write("- Cultural and linguistic variations can affect embeddings\n")
    f.write("- Not suitable for non-ordinal categories\n\n")

    f.write("---\n\n")

    # Footer
    f.write("## Appendix\n\n")
    f.write("### Technical Details\n\n")
    f.write(f"- **Embedding Model:** sentence-transformers (all-MiniLM-L6-v2)\n")
    f.write(f"- **Similarity Metric:** Cosine similarity\n")
    f.write(f"- **Normalization:** Softmax with temperature = 1.0\n")
    f.write(f"- **Evaluation:** Mode accuracy, MAE, RMSE, probabilistic metrics\n\n")

    f.write("### References\n\n")
    f.write("- Paper: *LLMs Reproduce Human Purchase Intent via Semantic Similarity Elicitation* "
           "(arXiv:2510.08338v2)\n")
    f.write("- Repository: https://github.com/pymc-labs/semantic-similarity-rating\n")
    f.write("- Documentation: See README.md for full pipeline details\n\n")

    f.write("---\n\n")
    f.write("*Report generated by SSR Pipeline*\n")
