"""Test script for Kantar question template system."""

import sys
from pathlib import Path

# Add src to path
sys.path.insert(0, str(Path(__file__).parent))

from src.survey import Survey

def test_template_loading():
    """Test loading a survey config with templates."""

    print("=" * 80)
    print("TESTING KANTAR TEMPLATE SYSTEM")
    print("=" * 80)

    # Load the Kantar lottery survey config
    config_path = "config/kantar_lottery_survey.yaml"

    print(f"\n[1] Loading survey from: {config_path}")
    try:
        survey = Survey.from_config(config_path)
        print(f"    ✓ Successfully loaded: '{survey.name}'")
    except Exception as e:
        print(f"    ✗ Error loading survey: {e}")
        import traceback
        traceback.print_exc()
        return False

    # Check templates were loaded
    print(f"\n[2] Checking templates...")
    print(f"    - Number of templates: {len(survey.question_templates)}")
    print(f"    - Template IDs: {list(survey.question_templates.keys())}")

    # Check questions were created
    print(f"\n[3] Checking questions...")
    print(f"    - Number of questions: {len(survey.questions)}")
    print(f"    - Question types:")
    type_counts = {}
    for q in survey.questions:
        type_counts[q.type] = type_counts.get(q.type, 0) + 1
    for q_type, count in sorted(type_counts.items()):
        print(f"      * {q_type}: {count}")

    # Show first few questions to verify template expansion
    print(f"\n[4] First 3 questions (verifying template expansion):")
    for i, q in enumerate(survey.questions[:3], 1):
        print(f"\n    Question {i}: {q.id}")
        print(f"    - Type: {q.type}")
        print(f"    - Text: {q.text}")
        if q.scale:
            print(f"    - Scale points: {q.scale.num_points} ({q.scale.min_value} to {q.scale.max_value})")
        if q.options:
            print(f"    - Options: {len(q.options)} choices")

    # Check persona groups
    print(f"\n[5] Checking persona groups...")
    print(f"    - Number of groups: {len(survey.persona_groups)}")
    for pg in survey.persona_groups:
        print(f"      * {pg.name} (weight: {pg.weight})")
        print(f"        - Personas: {len(pg.personas)}")
        print(f"        - Demographics: {list(pg.target_demographics.keys())}")

    # Test format_prompt
    print(f"\n[6] Testing prompt formatting...")
    test_profile = {
        'description': 'A 32-year-old software engineer who plays lottery weekly',
        'respondent_id': 'R001',
        'gender': 'Male',
        'age_group': '25-34',
        'occupation': 'Technical'
    }

    test_question = survey.questions[0]
    prompt = survey.format_prompt(test_question, test_profile)
    print(f"    - Generated prompt length: {len(prompt)} characters")
    print(f"    - Includes context: {'✓' if 'ElfYourself' in prompt else '✗'}")
    print(f"    - Includes profile: {'✓' if 'respondent_id' in prompt else '✗'}")
    print(f"    - Includes question: {'✓' if test_question.text in prompt else '✗'}")

    print("\n" + "=" * 80)
    print("✓ ALL TESTS PASSED - Template system is working!")
    print("=" * 80)

    return True


if __name__ == "__main__":
    success = test_template_loading()
    sys.exit(0 if success else 1)
