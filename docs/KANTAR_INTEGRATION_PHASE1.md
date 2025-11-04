# Kantar Survey Integration - Phase 1: Template System

## Overview

Phase 1 implements a **question template system** that allows you to define reusable Kantar-style survey questions. This enables:

1. **Standardization**: Define common Kantar questions once, reuse across surveys
2. **Consistency**: Ensure all surveys use the same scales and wording
3. **Efficiency**: Reduce repetition in survey config files
4. **Maintainability**: Update a template in one place, affects all uses

## What Was Implemented

### 1. New Data Structure: `QuestionTemplate`

Located in `src/survey.py:131-139`

```python
@dataclass
class QuestionTemplate:
    """Represents a reusable question template for Kantar-style surveys."""
    id: str
    text: str
    type: str
    scale: Optional[Dict[int, str]] = None
    options: Optional[List[str]] = None
    description: Optional[str] = None
```

### 2. Updated `Survey` Class

- Added `question_templates` field to store templates (line 153)
- Enhanced `from_config()` method to parse and expand templates (lines 168-266)

### 3. Template Parsing Logic

The system now:
1. **Parses templates first** from the `question_templates` section
2. **Expands template references** when questions use `template: <template_id>`
3. **Allows overrides** - questions can override template values if needed
4. **Maintains backward compatibility** - surveys without templates still work

### 4. Example Survey Configuration

Created `config/kantar_lottery_survey.yaml` with:
- **10 standard Kantar templates** (purchase intent, uniqueness, value for money, etc.)
- **14 complete questions** (10 using templates, 4 custom)
- **4 persona groups** matching lottery player segments
- **Full concept description** following Kantar format

## How to Use Templates

### Basic Template Definition

In your survey YAML, define templates in the `question_templates` section:

```yaml
survey:
  name: "My Survey"
  # ... other fields ...

  question_templates:
    purchase_intent:
      text: "How likely would you be to subscribe to this product?"
      type: "likert_5"
      description: "Kantar standard purchase intent"
      scale:
        1: "Definitely would not subscribe"
        2: "Probably would not subscribe"
        3: "Might or might not subscribe"
        4: "Probably would subscribe"
        5: "Definitely would subscribe"

    uniqueness:
      text: "This product is unique and different"
      type: "likert_5"
      scale:
        1: "Strongly disagree"
        2: "Disagree"
        3: "Neither agree nor disagree"
        4: "Agree"
        5: "Strongly agree"
```

### Using Templates in Questions

Reference a template using the `template` field:

```yaml
  questions:
    # Using a template - only need to provide ID
    - template: "purchase_intent"
      id: "q1_purchase_intent"

    # Using a template with text override
    - template: "uniqueness"
      id: "q2_uniqueness"
      text: "This lottery platform is unique and different from competitors"

    # Regular question (no template)
    - id: "q3_custom"
      text: "What is your favorite feature?"
      type: "multiple_choice"
      options:
        - "Mobile app"
        - "Automated purchasing"
        - "Multiple games"
```

### Template Override Rules

When a question references a template:

1. **ID is required** - must be specified in the question
2. **Type is inherited** - from template unless overridden
3. **Text is inherited** - from template unless overridden
4. **Scale/Options inherited** - from template unless overridden

Example with overrides:

```yaml
  questions:
    - template: "purchase_intent"
      id: "q1_lottery_intent"
      text: "How likely would you be to buy lottery tickets online?"
      # Inherits type: "likert_5" and scale from template
```

## Standard Kantar Templates Included

The `kantar_lottery_survey.yaml` includes these 10 standard templates:

| Template ID | Type | Description |
|------------|------|-------------|
| `purchase_intent` | likert_5 | Standard Kantar purchase likelihood |
| `uniqueness` | likert_5 | Differentiation/uniqueness measure |
| `value_for_money` | likert_5 | Value perception |
| `likeability` | likert_7 | Overall liking (7-point) |
| `relevance` | likert_5 | Personal relevance |
| `excitement` | likert_5 | Excitement/interest |
| `believability` | likert_5 | Credibility of claims |
| `understanding` | likert_7 | Comprehension (7-point) |
| `trust` | likert_5 | Trust and security |
| `recommendation` | likert_5 | NPS-style recommendation |

These match the question types found in the Kantar questionnaires you provided.

## Testing

Run the test script to verify template functionality:

```bash
python3 test_kantar_templates.py
```

Expected output:
```
✓ Successfully loaded: 'Online Lottery Gaming Platform Evaluation'
- Number of templates: 10
- Number of questions: 14
- Question types: likert_5 (8), likert_7 (2), multiple_choice (3), yes_no (1)
```

## Benefits of This Approach

### 1. Standardization
All surveys using the same template will have identical scales and wording, ensuring comparability across studies.

### 2. DRY (Don't Repeat Yourself)
Define common questions once, reuse everywhere:

```yaml
# Without templates (repetitive):
questions:
  - id: "q1_intent_productA"
    text: "How likely would you be to subscribe?"
    type: "likert_5"
    scale:
      1: "Definitely would not subscribe"
      # ... all 5 labels

  - id: "q1_intent_productB"
    text: "How likely would you be to subscribe?"
    type: "likert_5"
    scale:
      1: "Definitely would not subscribe"
      # ... same 5 labels again

# With templates (clean):
questions:
  - template: "purchase_intent"
    id: "q1_intent_productA"

  - template: "purchase_intent"
    id: "q1_intent_productB"
```

### 3. Easy Updates
Need to change the purchase intent scale? Update the template once, all questions automatically updated.

### 4. Self-Documenting
Templates include descriptions explaining their purpose:

```yaml
purchase_intent:
  description: "Kantar standard purchase intent - 5-point scale"
  # ...
```

### 5. Backward Compatible
Existing surveys without templates continue to work exactly as before.

## Using with the SSR Pipeline

The template system integrates seamlessly with the existing pipeline:

```bash
# Generate ground truth with Kantar-style survey
python3 ground_truth_pipeline.py \
  '{}' \
  '' \
  config/kantar_lottery_survey.yaml
```

The pipeline will:
1. Load the survey config
2. Expand all template references
3. Generate respondent profiles from persona groups
4. Create LLM responses
5. Process with SSR
6. Output results with demographics

## Next Steps (Future Phases)

Phase 1 provides the foundation. Future phases will add:

### Phase 2: Screener Support
- Add `screener` section to YAML
- Implement skip logic (terminate respondents based on answers)
- Include screener responses in output

### Phase 3: Kantar Export Format
- Add `kantar_export_format` parameter to pipeline
- Output CSV matching Kantar data structure
- Question code mapping (SSR IDs → Kantar codes)

### Phase 4: Comparison Module
- New `src/kantar_comparison.py` module
- Load actual Kantar ground truth data
- Compare SSR outputs against Kantar results
- Demographic weighting for comparison

## Example: Creating Your Own Kantar Survey

1. **Copy the template file**:
```bash
cp config/kantar_lottery_survey.yaml config/my_kantar_survey.yaml
```

2. **Update the concept**:
```yaml
survey:
  name: "Your Product Name"
  context: |
    Description of your concept...
    Features, pricing, etc.
```

3. **Keep or modify templates**:
The 10 standard templates work for most concept tests. Add custom templates if needed.

4. **Define your questions**:
```yaml
  questions:
    - template: "purchase_intent"
      id: "q1_purchase_intent"

    - template: "uniqueness"
      id: "q2_uniqueness"

    # Add custom questions as needed
```

5. **Update persona groups**:
Define target audiences matching your product category.

6. **Run the pipeline**:
```bash
python3 ground_truth_pipeline.py '{}' '' config/my_kantar_survey.yaml
```

## Files Modified

- `src/survey.py` - Added QuestionTemplate class and template parsing logic
- `config/kantar_lottery_survey.yaml` - New comprehensive example survey
- `test_kantar_templates.py` - Test script for template functionality

## Conclusion

Phase 1 successfully implements a reusable question template system that:
- Matches Kantar's standard question formats
- Reduces config file repetition
- Maintains consistency across surveys
- Integrates seamlessly with existing SSR Pipeline

This provides the foundation for building Kantar-compatible surveys that can be compared against actual Kantar ground truth data in future phases.
