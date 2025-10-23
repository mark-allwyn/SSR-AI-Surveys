#!/usr/bin/env python
"""
Ground Truth Comparison Pipeline.

This script:
1. Generates ground truth ratings for survey respondents
2. Generates textual responses from both "humans" and "LLMs"
3. Applies SSR to convert text to probability distributions
4. Compares SSR predictions against ground truth
5. Generates a one-page comparison report
"""

import sys
from pathlib import Path
import numpy as np
import pandas as pd
from datetime import datetime
import json

sys.path.insert(0, str(Path(__file__).parent / "src"))

from src.survey import Survey
from src.llm_client import Response, generate_diverse_profiles
from src.ssr_model import SemanticSimilarityRater
from src.ground_truth import (
    create_ground_truth_dict,
    evaluate_against_ground_truth,
    print_ground_truth_comparison
)
from src.report_generator import create_one_page_report, generate_text_report
from src.markdown_report import generate_comprehensive_report


def generate_ground_truth_ratings(survey: Survey, profiles: list, seed: int = 100) -> pd.DataFrame:
    """
    Generate ground truth ratings based on respondent profiles.

    For description-based personas, generates realistic varied distributions.

    Returns DataFrame with columns: respondent_id, question_id, ground_truth
    """
    np.random.seed(seed)

    records = []

    for i, profile in enumerate(profiles):
        respondent_id = f"R{i+1:03d}"

        # For description-based personas, use hash of description to determine tendency
        # This ensures same persona gets consistent tendency across runs
        description = profile.description if hasattr(profile, 'description') else str(profile)
        description_hash = hash(description) % 3

        if description_hash == 0:
            tendency = "positive"
        elif description_hash == 1:
            tendency = "negative"
        else:
            tendency = "neutral"

        for question in survey.questions:
            ref_statements = question.get_reference_statements()
            scale_points = sorted(ref_statements.keys())
            n_options = len(scale_points)

            # Generate ground truth based on tendency and question type
            if tendency == "positive":
                # Skew towards higher ratings
                if n_options == 2:
                    probs = np.array([0.2, 0.8])
                elif n_options == 5:
                    probs = np.array([0.05, 0.10, 0.20, 0.30, 0.35])
                elif n_options == 7:
                    probs = np.array([0.02, 0.05, 0.08, 0.15, 0.20, 0.25, 0.25])
                else:
                    # General case: exponential decay from high to low
                    probs = np.exp(np.linspace(0, 2, n_options))
                    probs = probs / probs.sum()
            elif tendency == "negative":
                # Skew towards lower ratings
                if n_options == 2:
                    probs = np.array([0.8, 0.2])
                elif n_options == 5:
                    probs = np.array([0.35, 0.30, 0.20, 0.10, 0.05])
                elif n_options == 7:
                    probs = np.array([0.25, 0.25, 0.20, 0.15, 0.08, 0.05, 0.02])
                else:
                    # General case: exponential decay from low to high
                    probs = np.exp(np.linspace(2, 0, n_options))
                    probs = probs / probs.sum()
            else:
                # More uniform
                probs = np.ones(n_options) / n_options

            # Ensure probs sum to 1 (handle any rounding errors)
            probs = probs / probs.sum()

            # Sample ground truth
            ground_truth = np.random.choice(scale_points, p=probs)

            records.append({
                'respondent_id': respondent_id,
                'question_id': question.id,
                'ground_truth': ground_truth
            })

    return pd.DataFrame(records)


def generate_responses_from_ground_truth(
    survey: Survey,
    profiles: list,
    ground_truth_df: pd.DataFrame,
    response_style: str = "human",  # "human" or "llm"
    seed: int = 101
) -> list:
    """
    Generate textual responses that align with ground truth ratings.

    Args:
        survey: Survey object
        profiles: List of respondent profiles
        ground_truth_df: DataFrame with ground truth ratings
        response_style: "human" (direct) or "llm" (hedged)
        seed: Random seed

    Returns:
        List of Response objects
    """
    np.random.seed(seed)

    responses = []

    for i, profile in enumerate(profiles):
        respondent_id = f"R{i+1:03d}"

        for question in survey.questions:
            # Get ground truth rating
            gt_row = ground_truth_df[
                (ground_truth_df['respondent_id'] == respondent_id) &
                (ground_truth_df['question_id'] == question.id)
            ]
            ground_truth = int(gt_row['ground_truth'].values[0])

            # Generate response text based on ground truth
            ref_statements = question.get_reference_statements()
            target_statement = ref_statements[ground_truth]

            if response_style == "human":
                # Direct, matches ground truth closely
                text_response = generate_human_style_response(target_statement, ground_truth, question.num_options)
            else:
                # Hedged LLM style
                text_response = generate_llm_style_response(target_statement, ground_truth, question.num_options)

            response = Response(
                respondent_id=respondent_id,
                question_id=question.id,
                text_response=text_response,
                respondent_profile=profile.to_dict()
            )
            responses.append(response)

    return responses


def generate_human_style_response(target_statement: str, rating: int, max_rating: int) -> str:
    """Generate direct human-style response."""
    # Add some variation but stay close to target
    variations = [
        f"{target_statement}",
        f"I'd say {target_statement.lower()}",
        f"Definitely {target_statement.lower()}",
        f"My answer is {target_statement.lower()}",
    ]
    return np.random.choice(variations)


def generate_llm_style_response(target_statement: str, rating: int, max_rating: int) -> str:
    """Generate hedged LLM-style response."""
    hedges = [
        "I would say that",
        "It seems to me that",
        "I think",
        "In my view",
        "I'd probably describe it as"
    ]

    qualifiers = [
        ", though it's somewhat difficult to say definitively",
        ", although there are nuances to consider",
        ", taking various factors into account",
        ", considering the overall context"
    ]

    hedge = np.random.choice(hedges)
    qualifier = np.random.choice(qualifiers) if np.random.random() > 0.5 else ""

    return f"{hedge} {target_statement.lower()}{qualifier}."


def main(persona_config=None):
    """Run the ground truth comparison pipeline.

    Args:
        persona_config: Optional dict with persona configuration
    """
    # Generate timestamp for this run
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")

    # Create experiment folder with timestamp
    experiment_dir = Path(f"experiments/run_{timestamp}")
    experiment_dir.mkdir(parents=True, exist_ok=True)

    print("\n" + "=" * 80)
    print("GROUND TRUTH COMPARISON PIPELINE")
    print("=" * 80)
    print(f"\nExperiment Folder: {experiment_dir}")

    # Save persona configuration if provided
    if persona_config:
        persona_config_path = experiment_dir / 'persona_config.json'
        with open(persona_config_path, 'w') as f:
            json.dump(persona_config, f, indent=2)
        print(f"    ✓ Saved persona configuration to {persona_config_path}")

    # 1. Load survey
    print("\n[1/8] Loading survey configuration...")
    survey = Survey.from_config('config/mixed_survey_config.yaml')
    print(f"    ✓ Loaded: '{survey.name}'")
    print(f"    ✓ Questions: {len(survey.questions)}")
    for q in survey.questions:
        print(f"      - {q.id} ({q.type})")

    # 2. Generate respondent profiles
    print("\n[2/8] Generating respondent profiles...")
    n_respondents = survey.sample_size
    profiles = generate_diverse_profiles(n_respondents, persona_config=persona_config)
    print(f"    ✓ Generated {len(profiles)} profiles")
    if persona_config and persona_config.get('custom_fields'):
        custom_fields = persona_config['custom_fields']
        print(f"    ✓ Using {len(custom_fields)} custom persona fields: {', '.join(custom_fields.keys())}")

    # 3. Generate ground truth ratings
    print("\n[3/8] Generating ground truth ratings...")
    ground_truth_df = generate_ground_truth_ratings(survey, profiles)

    # Save to experiment folder
    gt_path = experiment_dir / 'ground_truth.csv'
    ground_truth_df.to_csv(gt_path, index=False)
    print(f"    ✓ Generated {len(ground_truth_df)} ground truth ratings")
    print(f"    ✓ Saved to {gt_path}")

    # 4. Generate "human" responses aligned with ground truth
    print("\n[4/8] Generating human-style responses...")
    human_responses = generate_responses_from_ground_truth(
        survey, profiles, ground_truth_df, response_style="human", seed=101
    )
    print(f"    ✓ Generated {len(human_responses)} human responses")

    # 5. Generate "LLM" responses aligned with ground truth
    print("\n[5/8] Generating LLM-style responses...")
    llm_responses = generate_responses_from_ground_truth(
        survey, profiles, ground_truth_df, response_style="llm", seed=102
    )
    print(f"    ✓ Generated {len(llm_responses)} LLM responses")

    # 6. Apply SSR to convert text responses to distributions
    print("\n[6/8] Applying Semantic Similarity Rating...")
    print("    • Using paper's methodology (arXiv:2510.08338v2)")
    print("    • Model: OpenAI text-embedding-3-small")
    print("    • Normalization: Paper's method (subtract min + proportional)")

    rater = SemanticSimilarityRater(
        model_name="text-embedding-3-small",
        temperature=1.0,
        normalize_method="paper",
        use_openai=True
    )

    print("    • Rating human responses...")
    human_distributions = rater.rate_responses(human_responses, survey, show_progress=True)

    print("    • Rating LLM responses...")
    llm_distributions = rater.rate_responses(llm_responses, survey, show_progress=True)

    print(f"    ✓ Created {len(human_distributions) + len(llm_distributions)} distributions")

    # 7. Evaluate against ground truth
    print("\n[7/8] Evaluating against ground truth...")
    ground_truth_dict = create_ground_truth_dict(ground_truth_df)

    human_comparisons = {}
    llm_comparisons = {}

    for question in survey.questions:
        # Filter distributions for this question
        h_dists = [d for d in human_distributions if d.question_id == question.id]
        l_dists = [d for d in llm_distributions if d.question_id == question.id]

        # Evaluate
        human_comp = evaluate_against_ground_truth(h_dists, ground_truth_dict, question)
        llm_comp = evaluate_against_ground_truth(l_dists, ground_truth_dict, question)

        human_comparisons[question.id] = human_comp
        llm_comparisons[question.id] = llm_comp

        print(f"\n    {question.id}:")
        print(f"      Human Accuracy: {human_comp.mode_accuracy:.1%}")
        print(f"      LLM Accuracy:   {llm_comp.mode_accuracy:.1%}")

    # 8. Generate reports in experiment folder
    print("\n[8/8] Generating reports...")

    # Create file paths in experiment folder
    png_path = experiment_dir / "report.png"
    txt_path = experiment_dir / "report.txt"
    md_path = experiment_dir / "report.md"

    # One-page visual report
    create_one_page_report(
        human_comparisons,
        llm_comparisons,
        survey,
        output_path=str(png_path),
        title="Ground Truth Comparison: Human vs LLM SSR Predictions"
    )

    # Text report
    generate_text_report(
        human_comparisons,
        llm_comparisons,
        survey,
        output_path=str(txt_path)
    )

    # Comprehensive markdown report
    generate_comprehensive_report(
        human_comparisons,
        llm_comparisons,
        survey,
        output_path=str(md_path)
    )

    # Summary
    print("\n" + "=" * 80)
    print("PIPELINE COMPLETED!")
    print("=" * 80)
    print(f"\nAll files saved to: {experiment_dir}")
    print("\nGenerated Files:")
    print(f"  • ground_truth.csv - Ground truth ratings")
    print(f"  • report.png - One-page visual report")
    print(f"  • report.txt - Detailed text report")
    print(f"  • report.md - Comprehensive markdown report with explanations")

    print("\nOverall Results:")
    h_avg = np.mean([c.mode_accuracy for c in human_comparisons.values()])
    l_avg = np.mean([c.mode_accuracy for c in llm_comparisons.values()])
    print(f"  Average Human Accuracy: {h_avg:.1%}")
    print(f"  Average LLM Accuracy:   {l_avg:.1%}")
    print(f"  Winner: {'Human' if h_avg > l_avg else ('LLM' if l_avg > h_avg else 'Tie')}")
    print("\n")


if __name__ == "__main__":
    # Ensure experiments directory exists
    Path("experiments").mkdir(parents=True, exist_ok=True)

    main()
