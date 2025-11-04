# Kantar Standard Question Templates - Quick Reference

## Overview

This document provides a quick reference for the 10 standard Kantar question templates included in the SSR Pipeline. These templates are based on the common question patterns found across multiple Kantar questionnaires.

## Usage

In your survey YAML file, reference templates like this:

```yaml
questions:
  - template: "purchase_intent"
    id: "q1_purchase_intent"
```

## Standard Templates

### 1. Purchase Intent (5-point)

**Template ID**: `purchase_intent`

**Question Type**: `likert_5`

**Standard Text**: "How likely would you be to subscribe to this [product/service]?"

**Scale**:
1. Definitely would not subscribe
2. Probably would not subscribe
3. Might or might not subscribe
4. Probably would subscribe
5. Definitely would subscribe

**When to Use**: Primary purchase intention measure. Always include in concept tests.

**Kantar Standard**: Top 2 box (4+5) is key metric for purchase intent.

---

### 2. Uniqueness/Differentiation

**Template ID**: `uniqueness`

**Question Type**: `likert_5`

**Standard Text**: "This [product/service] is unique and different from other [category] options"

**Scale**:
1. Strongly disagree
2. Disagree
3. Neither agree nor disagree
4. Agree
5. Strongly agree

**When to Use**: To assess how differentiated the concept is perceived to be.

**Key Metric**: Top 2 box (4+5) indicates strong differentiation.

---

### 3. Value for Money

**Template ID**: `value_for_money`

**Question Type**: `likert_5`

**Standard Text**: "The pricing for this service represents good value for money"

**Scale**:
1. Strongly disagree
2. Disagree
3. Neither agree nor disagree
4. Agree
5. Strongly agree

**When to Use**: Always include when pricing information is shown.

**Key Insight**: Low scores indicate price resistance; high scores validate pricing strategy.

---

### 4. Likeability (7-point)

**Template ID**: `likeability`

**Question Type**: `likert_7`

**Standard Text**: "Overall, how much do you like or dislike this [product/service] concept?"

**Scale**:
1. Dislike it a great deal
2. Dislike it
3. Dislike it a little
4. Neither like nor dislike it
5. Like it a little
6. Like it
7. Like it a great deal

**When to Use**: Overall emotional response to concept.

**Key Metric**: Top 3 box (5+6+7) indicates positive reception.

---

### 5. Relevance

**Template ID**: `relevance`

**Question Type**: `likert_5`

**Standard Text**: "This [product/service] is relevant to people like me"

**Scale**:
1. Strongly disagree
2. Disagree
3. Neither agree nor disagree
4. Agree
5. Strongly agree

**When to Use**: To assess personal applicability and target audience fit.

**Key Insight**: High relevance scores predict stronger purchase intent.

---

### 6. Excitement/Interest

**Template ID**: `excitement`

**Question Type**: `likert_5`

**Standard Text**: "This [product/service] concept is exciting and interesting"

**Scale**:
1. Strongly disagree
2. Disagree
3. Neither agree nor disagree
4. Agree
5. Strongly agree

**When to Use**: To measure emotional engagement with concept.

**Key Metric**: Predicts word-of-mouth and trial behavior.

---

### 7. Believability/Credibility

**Template ID**: `believability`

**Question Type**: `likert_5`

**Standard Text**: "The claims made about this [product/service] are believable"

**Scale**:
1. Strongly disagree
2. Disagree
3. Neither agree nor disagree
4. Agree
5. Strongly agree

**When to Use**: When concept includes specific claims or benefits.

**Key Insight**: Low believability undermines purchase intent regardless of appeal.

---

### 8. Understanding/Comprehension (7-point)

**Template ID**: `understanding`

**Question Type**: `likert_7`

**Standard Text**: "How well do you feel you understand what this [product/service] is offering?"

**Scale**:
1. Do not understand at all
2. Understand very little
3. Understand a little
4. Understand somewhat
5. Understand fairly well
6. Understand very well
7. Understand completely

**When to Use**: To validate concept clarity, especially for complex or innovative products.

**Key Metric**: Top 3 box (5+6+7) indicates clear communication.

---

### 9. Trust

**Template ID**: `trust`

**Question Type**: `likert_5`

**Standard Text**: "I would trust this platform with my payment information and [sensitive data]"

**Scale**:
1. Strongly disagree
2. Disagree
3. Neither agree nor disagree
4. Agree
5. Strongly agree

**When to Use**: For services requiring personal information, payments, or sensitive data.

**Key Insight**: Critical for fintech, health, or security-related products.

---

### 10. Recommendation (NPS-style)

**Template ID**: `recommendation`

**Question Type**: `likert_5`

**Standard Text**: "How likely would you be to recommend this [product/service] to a friend or family member?"

**Scale**:
1. Definitely would not recommend
2. Probably would not recommend
3. Might or might not recommend
4. Probably would recommend
5. Definitely would recommend

**When to Use**: Word-of-mouth potential, adapted from Net Promoter Score.

**Key Metric**: Top 2 box (4+5) indicates strong advocacy potential.

---

## Standard Kantar Evaluation Battery

For a comprehensive concept test, use questions in this order:

1. **Purchase Intent** - Primary outcome measure
2. **Uniqueness** - Differentiation
3. **Value for Money** - Price perception (if pricing shown)
4. **Likeability** - Overall appeal
5. **Relevance** - Target fit
6. **Excitement** - Emotional engagement
7. **Believability** - Credibility
8. **Understanding** - Concept clarity
9. **Trust** - Security/confidence (if applicable)
10. **Recommendation** - Advocacy potential

## Customizing Templates

You can override any template field:

```yaml
questions:
  # Use template but change text
  - template: "uniqueness"
    id: "q2_uniqueness"
    text: "This lottery app is completely different from traditional lottery purchasing"

  # Use template but change type (not recommended - breaks standardization)
  - template: "purchase_intent"
    id: "q1_intent"
    type: "likert_7"  # Override to 7-point scale
```

**Best Practice**: Only override `text` to adapt wording to your specific concept. Avoid changing `type` or `scale` as this breaks comparability.

## Adding Your Own Templates

To create custom templates for your organization:

```yaml
survey:
  question_templates:
    # Your custom template
    brand_fit:
      text: "This product fits well with the [BRAND] brand"
      type: "likert_5"
      description: "Custom brand fit measure"
      scale:
        1: "Strongly disagree"
        2: "Disagree"
        3: "Neither agree nor disagree"
        4: "Agree"
        5: "Strongly agree"

  questions:
    - template: "brand_fit"
      id: "q11_brand_fit"
      text: "This lottery platform fits well with the Powerball brand"
```

## Template Naming Conventions

Follow these conventions when creating new templates:

- **Snake case**: `purchase_intent`, not `PurchaseIntent`
- **Descriptive**: Name should indicate what it measures
- **No version numbers**: Use `purchase_intent` not `purchase_intent_v2`
- **Generic**: Avoid concept-specific names like `lottery_intent`

## References

These templates are based on analysis of:
- UK non players Wave 2 Kantar questionnaire
- Board games concept test questionnaires
- Standard Kantar evaluation methodologies

## See Also

- `KANTAR_INTEGRATION_PHASE1.md` - Full implementation guide
- `config/kantar_lottery_survey.yaml` - Complete example survey
- `test_kantar_templates.py` - Test suite for templates
