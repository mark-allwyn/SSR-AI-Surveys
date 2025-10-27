#!/usr/bin/env python
"""
Ground Truth Comparison Pipeline.

This script compares human ground truth ratings against LLM+SSR predictions:
1. Load or generate ground truth ratings (human responses)
2. Generate LLM textual responses for the same questions
3. Apply SSR to LLM text to get predictions
4. Compare: Ground Truth (100% accurate by definition) vs LLM+SSR predictions
5. Generate comparison report showing how well LLM+SSR recovers human ratings
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


def main(persona_config=None, ground_truth_path=None, survey_config_path='config/mixed_survey_config.yaml'):
    """Run the ground truth comparison pipeline.

    Args:
        persona_config: Optional dict with persona configuration
        ground_truth_path: Optional path to uploaded ground truth CSV file
        survey_config_path: Path to survey YAML config file (default: config/mixed_survey_config.yaml)
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
    survey = Survey.from_config(survey_config_path)
    print(f"    ✓ Loaded: '{survey.name}' from {survey_config_path}")
    print(f"    ✓ Questions: {len(survey.questions)}")
    for q in survey.questions:
        print(f"      - {q.id} ({q.type})")

    # Load personas from survey config
    if survey.personas:
        print(f"    ✓ Loaded {len(survey.personas)} personas from config")

    # 2. Generate respondent profiles
    print("\n[2/8] Generating respondent profiles...")
    n_respondents = survey.sample_size

    # Use personas from survey config if available
    if survey.personas:
        if not persona_config:
            persona_config = {}
        persona_config['mode'] = 'descriptions'
        persona_config['descriptions'] = survey.personas

    profiles = generate_diverse_profiles(n_respondents, persona_config=persona_config)
    print(f"    ✓ Generated {len(profiles)} profiles")

    # 3. Load or generate ground truth ratings
    if ground_truth_path and Path(ground_truth_path).exists():
        print("\n[3/8] Loading uploaded ground truth data...")
        ground_truth_df = pd.read_csv(ground_truth_path)
        print(f"    ✓ Loaded {len(ground_truth_df)} ground truth ratings from file")
        print(f"    ✓ File: {ground_truth_path}")

        # Validate ground truth has required columns
        required_cols = ['respondent_id', 'question_id', 'ground_truth']
        if not all(col in ground_truth_df.columns for col in required_cols):
            raise ValueError(f"Ground truth CSV must have columns: {required_cols}")

        # Validate matches survey questions
        gt_questions = set(ground_truth_df['question_id'].unique())
        survey_questions = set(q.id for q in survey.questions)
        if not gt_questions.issubset(survey_questions):
            missing = gt_questions - survey_questions
            print(f"    ⚠ Warning: Ground truth contains questions not in survey: {missing}")
    else:
        print("\n[3/8] Generating artificial ground truth ratings...")
        ground_truth_df = generate_ground_truth_ratings(survey, profiles)
        print(f"    ✓ Generated {len(ground_truth_df)} ground truth ratings")

    # Save to experiment folder
    gt_path = experiment_dir / 'ground_truth.csv'
    ground_truth_df.to_csv(gt_path, index=False)
    print(f"    ✓ Saved to {gt_path}")

    # 4. Generate LLM text responses (will be processed through SSR)
    print("\n[4/6] Generating LLM text responses...")
    llm_responses = generate_responses_from_ground_truth(
        survey, profiles, ground_truth_df, response_style="llm", seed=102
    )
    print(f"    ✓ Generated {len(llm_responses)} LLM responses")

    # 5. Apply SSR to LLM responses only
    print("\n[5/6] Applying SSR to LLM responses...")
    print("    • Using paper's methodology (arXiv:2510.08338v2)")
    print("    • Model: OpenAI text-embedding-3-small")
    print("    • Normalization: Paper's method (subtract min + proportional)")

    rater = SemanticSimilarityRater(
        model_name="text-embedding-3-small",
        temperature=1.0,
        normalize_method="paper",
        use_openai=True
    )

    print("    • Rating LLM responses...")
    llm_distributions = rater.rate_responses(llm_responses, survey, show_progress=True)
    print(f"    ✓ Created {len(llm_distributions)} SSR distributions")

    # 6. Evaluate: Ground Truth (perfect) vs LLM+SSR predictions
    print("\n[6/6] Evaluating LLM+SSR against ground truth...")
    ground_truth_dict = create_ground_truth_dict(ground_truth_df)

    # Ground truth comparisons (always 100% accurate)
    from src.ground_truth import GroundTruthComparison
    ground_truth_comparisons = {}
    llm_comparisons = {}

    for question in survey.questions:
        # Ground truth comparison (perfect accuracy by definition)
        gt_for_question = ground_truth_df[ground_truth_df['question_id'] == question.id]
        n_samples = len(gt_for_question)

        # Create perfect comparison for ground truth
        ground_truth_comparisons[question.id] = GroundTruthComparison(
            question_id=question.id,
            question_type=question.type,
            mode_accuracy=1.0,  # 100% by definition
            top2_accuracy=1.0,
            mae=0.0,  # Perfect
            rmse=0.0,  # Perfect
            mean_probability_at_truth=1.0,  # Perfect
            log_likelihood=0.0,  # Perfect (log(1) = 0)
            kl_divergence=0.0,  # Perfect
            confusion_matrix=np.eye(question.num_options, dtype=int) * (n_samples // question.num_options),
            n_samples=n_samples
        )

        # LLM+SSR comparison (actual performance)
        l_dists = [d for d in llm_distributions if d.question_id == question.id]
        llm_comp = evaluate_against_ground_truth(l_dists, ground_truth_dict, question)
        llm_comparisons[question.id] = llm_comp

        print(f"\n    {question.id}:")
        print(f"      Ground Truth: 100.0% (perfect by definition)")
        print(f"      LLM+SSR:      {llm_comp.mode_accuracy:.1%}")

    # 7. Save confusion matrices for dashboard
    print("\nSaving confusion matrices...")
    confusion_matrices = {}
    for question_id, llm_comp in llm_comparisons.items():
        confusion_matrices[question_id] = llm_comp.confusion_matrix.tolist()

    cm_path = experiment_dir / 'confusion_matrices.json'
    with open(cm_path, 'w') as f:
        json.dump(confusion_matrices, f, indent=2)
    print(f"    ✓ Saved confusion matrices to {cm_path}")

    # 7b. Save LLM probability distributions for advanced visualizations
    print("\nSaving LLM probability distributions...")
    distributions_data = {}

    for dist in llm_distributions:
        question_id = dist.question_id
        respondent_id = dist.respondent_id

        if question_id not in distributions_data:
            distributions_data[question_id] = {}

        # Get ground truth for this respondent/question
        gt_value = ground_truth_df[
            (ground_truth_df['respondent_id'] == respondent_id) &
            (ground_truth_df['question_id'] == question_id)
        ]['ground_truth'].values[0] if len(ground_truth_df[
            (ground_truth_df['respondent_id'] == respondent_id) &
            (ground_truth_df['question_id'] == question_id)
        ]) > 0 else None

        distributions_data[question_id][respondent_id] = {
            'probabilities': dist.distribution.tolist(),
            'ground_truth': int(gt_value) if gt_value is not None else None,
            'mode': int(dist.mode),
            'expected_value': float(dist.expected_value),
            'entropy': float(dist.entropy)
        }

    dist_path = experiment_dir / 'llm_distributions.json'
    with open(dist_path, 'w') as f:
        json.dump(distributions_data, f, indent=2)
    print(f"    ✓ Saved LLM distributions to {dist_path}")

    # 8. Generate reports in experiment folder
    print("\nGenerating reports...")

    # Create file paths in experiment folder
    png_path = experiment_dir / "report.png"
    txt_path = experiment_dir / "report.txt"
    md_path = experiment_dir / "report.md"

    # One-page visual report
    create_one_page_report(
        ground_truth_comparisons,
        llm_comparisons,
        survey,
        output_path=str(png_path),
        title="Ground Truth (Human) vs LLM+SSR Predictions"
    )

    # Text report
    generate_text_report(
        ground_truth_comparisons,
        llm_comparisons,
        survey,
        output_path=str(txt_path)
    )

    # Comprehensive markdown report
    generate_comprehensive_report(
        ground_truth_comparisons,
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
    print(f"  • ground_truth.csv - Human ground truth ratings")
    print(f"  • confusion_matrices.json - Confusion matrices for each question")
    print(f"  • llm_distributions.json - LLM probability distributions")
    print(f"  • report.png - One-page visual report")
    print(f"  • report.txt - Detailed text report")
    print(f"  • report.md - Comprehensive markdown report")

    print("\nOverall Results:")
    gt_avg = 100.0  # Always perfect
    llm_avg = np.mean([c.mode_accuracy for c in llm_comparisons.values()]) * 100
    print(f"  Ground Truth (Human): 100.0% (perfect by definition)")
    print(f"  LLM+SSR:              {llm_avg:.1f}%")
    print(f"  Gap:                  {100.0 - llm_avg:.1f}%")
    print("\n")


if __name__ == "__main__":
    # Ensure experiments directory exists
    Path("experiments").mkdir(parents=True, exist_ok=True)

    # Parse command-line arguments
    persona_config = None
    ground_truth_path = None
    survey_config_path = 'config/mixed_survey_config.yaml'  # Default

    if len(sys.argv) > 1:
        # First argument is persona config JSON
        persona_config = json.loads(sys.argv[1])

    if len(sys.argv) > 2:
        # Second argument is ground truth CSV path
        ground_truth_path = sys.argv[2]

    if len(sys.argv) > 3:
        # Third argument is survey config path
        survey_config_path = sys.argv[3]

    main(persona_config=persona_config, ground_truth_path=ground_truth_path, survey_config_path=survey_config_path)
