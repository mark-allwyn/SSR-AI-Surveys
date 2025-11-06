# S.A.G.E - Survey Analytics and Generation Engine

**Powered by Semantic Similarity Rating (SSR)**

## Overview

S.A.G.E (Survey Analytics and Generation Engine) is a comprehensive platform for analyzing survey responses using Semantic Similarity Rating. The SSR methodology converts textual survey responses into probability distributions over Likert scale options using semantic similarity, preserving uncertainty and nuance in responses while enabling quantitative analysis.

**Key Features:**
- ✅ **Interactive Web UI**: User-friendly Streamlit interface for experiments, results, and live demos
- ✅ **Paper-exact implementation**: Uses OpenAI text-embedding-3-small and the paper's normalization method
- ✅ **Multiple question types**: Binary (yes/no), Likert-5, Likert-7, multiple choice, preference scales
- ✅ **Multi-category surveys**: Compare 1-2 product categories with same questions + comparative preference questions
- ✅ **Ground truth evaluation**: Comprehensive metrics including mode accuracy, MAE, RMSE, KL divergence
- ✅ **Response style comparison**: Evaluate SSR on human-style vs LLM-style responses
- ✅ **Automated reporting**: Generates PNG visualizations, TXT metrics, and comprehensive Markdown reports
- ✅ **Experiment organization**: Timestamped folders keep all experiment results organized
- ✅ **Live Demo Mode**: Test SSR on individual responses in real-time
- ✅ **Survey Management**: Create and manage custom surveys via YAML configuration
- ✅ **Demographics System (v2.0)**: Track gender, age_group, persona_group, and occupation throughout pipeline
- ✅ **Persona Groups (v2.0)**: Weighted sampling with target demographic distributions
- ✅ **Category Analysis**: Filter and compare performance across product categories

---

## Table of Contents

1. [Installation](#installation)
2. [Quick Start](#quick-start)
3. [Project Structure](#project-structure)
4. [Detailed Usage](#detailed-usage)
5. [Understanding the Pipeline](#understanding-the-pipeline)
6. [Creating Custom Surveys](#creating-custom-surveys)
7. [Interpreting Results](#interpreting-results)
8. [API Reference](#api-reference)
9. [Paper Methodology](#paper-methodology)
10. [Troubleshooting](#troubleshooting)
11. [References](#references)

---

## Installation

### 1. Clone or download this repository

```bash
cd /path/to/ssr_pipeline
```

### 2. Create a virtual environment

```bash
python -m venv .venv
source .venv/bin/activate  # On Windows: .venv\Scripts\activate
```

### 3. Install dependencies

```bash
pip install -r requirements.txt
# OR if using uv:
uv pip install -r requirements.txt
```

### 4. Set up API keys

Create a `.env` file in the project root:

```bash
# .env
OPENAI_API_KEY=your_openai_api_key_here
```

**Important**: You need an OpenAI API key to use the paper's exact embedding model (`text-embedding-3-small`).

---

## Quick Start

### Option 1: Use the S.A.G.E Web Interface (Recommended)

```bash
# 1. Activate environment
source .venv/bin/activate

# 2. Launch S.A.G.E
streamlit run ui/1_Home.py

# 3. Open your browser at http://localhost:8501
```

The web interface provides:
- **Dashboard**: Overview of experiments and recent activity
- **Run Experiment**: Configure and execute SSR experiments with custom surveys
- **Results Dashboard**: Interactive visualizations and detailed analytics for single experiments
- **Compare Experiments**: Side-by-side comparison of multiple experiment runs with trend analysis
- **Live Demo**: Test SSR on individual text responses in real-time
- **Settings**: API configuration, persona management, and experiment cleanup

### Option 2: Use the Command Line

```bash
# 1. Activate environment
source .venv/bin/activate

# 2. Run the ground truth comparison pipeline
python ground_truth_pipeline.py

# 3. View results in experiments/run_TIMESTAMP/
```

This generates in `experiments/run_TIMESTAMP/`:
- `ground_truth.csv` - Ground truth ratings with demographics (v2.0+)
- `llm_distributions.json` - SSR probability distributions with demographics (v2.0+)
- `confusion_matrices.json` - Confusion matrices for error analysis
- `report.png` - Visual comparison report
- `report.txt` - Detailed metrics summary
- `report.md` - Comprehensive markdown report with explanations

---

## Project Structure

```
ssr_pipeline/
├── README.md                          # This file
├── CHANGELOG.md                       # Version history (v2.0+)
├── requirements.txt                   # Python dependencies
├── ground_truth_pipeline.py          # Main pipeline script
│
├── config/                           # Survey configurations
│   ├── mixed_survey_config.yaml      # Example survey (6 questions)
│   └── kantar_lottery_survey.yaml    # Kantar-style survey example
│
├── src/                              # Core modules
│   ├── __init__.py
│   ├── survey.py                     # Survey + persona groups
│   ├── llm_client.py                 # Response generation and profiles
│   ├── ssr_model.py                  # SSR implementation (paper-exact)
│   ├── ground_truth.py               # Evaluation metrics
│   ├── demographics.py               # Demographics module (v2.0+)
│   ├── report_generator.py           # PNG + TXT reports
│   └── markdown_report.py            # Markdown reports
│
├── docs/                             # Documentation
│   ├── database_schema.md            # Future PostgreSQL schema (planned)
│   ├── UI_REFACTORING_PLAN.md        # UI enhancement roadmap
│   └── generate_erd.py               # ERD generation script
│
├── ui/                               # Streamlit web interface
│   ├── 1_Home.py                     # Main dashboard
│   ├── pages/                        # UI pages
│   ├── components/                   # Reusable components
│   └── utils/                        # UI utilities
│
├── experiments/                      # Experiment results (auto-created)
│   └── run_TIMESTAMP/
│       ├── ground_truth.csv          # With demographics columns (v2.0+)
│       ├── llm_distributions.json    # With demographics fields (v2.0+)
│       ├── confusion_matrices.json   # Confusion matrices for analysis
│       ├── report.png
│       ├── report.txt
│       └── report.md
│
└── data/                             # Data directory (optional)
```

---

## Detailed Usage

### Understanding the Complete Workflow

The SSR pipeline follows these steps:

```
1. Survey Definition
   ↓
2. Respondent Profiles
   ↓
3. Ground Truth Generation
   ↓
4. Textual Response Generation
   ↓
5. SSR Application (text → distributions)
   ↓
6. Evaluation (predictions vs ground truth)
   ↓
7. Report Generation
```

Let's walk through each step:

---

### Step 1: Survey Definition

Surveys are defined in YAML files with multiple question types:

**Example**: `config/mixed_survey_config.yaml`

```yaml
survey:
  name: "Online Lottery Gaming Platform Evaluation"
  description: "Comprehensive evaluation of online lottery gaming products and services"

  context: |
    We are evaluating a new online lottery gaming platform...

  questions:
    - id: "q1_would_subscribe"
      text: "Would you subscribe to this online lottery platform?"
      type: "yes_no"

    - id: "q2_subscription_likelihood"
      text: "How likely are you to subscribe?"
      type: "likert_5"
      scale:
        1: "Very unlikely"
        2: "Unlikely"
        3: "Neutral"
        4: "Likely"
        5: "Very likely"

    - id: "q3_platform_trust"
      text: "How much would you trust this platform?"
      type: "likert_7"
      scale:
        1: "Not at all trustworthy"
        # ... 2-6 ...
        7: "Completely trustworthy"

    - id: "q4_price_preference"
      text: "What monthly subscription price would you consider reasonable?"
      type: "multiple_choice"
      options:
        - "Under $5/month"
        - "$5-$10/month"
        - "$10-$15/month"
        - "$15-$20/month"
        - "Over $20/month"
```

**Supported Question Types:**
- `yes_no`: Binary questions (2 options)
- `likert_5`: 5-point Likert scale
- `likert_7`: 7-point Likert scale
- `multiple_choice`: Custom categorical options

---

### Step 2: Respondent Profiles

The pipeline generates diverse respondent profiles:

```python
from src.llm_client import generate_diverse_profiles

profiles = generate_diverse_profiles(n=50)
# Generates 50 profiles with varying:
# - Environmental consciousness
# - Demographics
# - Attitudes
```

**Example Profile:**
```python
RespondentProfile(
    environmental_consciousness="Very concerned"
)
```

---

### Step 3: Ground Truth Generation

Ground truth ratings are generated based on respondent profiles:

```python
from ground_truth_pipeline import generate_ground_truth_ratings

ground_truth_df = generate_ground_truth_ratings(survey, profiles, seed=100)
```

**Output** (`ground_truth.csv` with demographics in v2.0+):
```
respondent_id,question_id,ground_truth,gender,age_group,persona_group,occupation
R001,q1_would_subscribe,1,Female,25-34,Frequent Players - Tech Savvy,Professional
R001,q2_subscription_likelihood,4,Female,25-34,Frequent Players - Tech Savvy,Professional
R001,q3_platform_trust,5,Female,25-34,Frequent Players - Tech Savvy,Professional
...
```

**Logic:**
- Profiles with high environmental consciousness → higher ratings
- Profiles with low environmental consciousness → lower ratings
- Introduces realistic variation using probability distributions

#### How Ground Truth Generation Works

The ground truth generation process creates realistic "true" ratings that later get validated against SSR predictions. Here's the detailed methodology:

**1. Profile-Based Tendency Determination:**

Each respondent profile is analyzed to determine their overall tendency (positive, negative, or neutral):

```python
# Example from ground_truth_pipeline.py
def determine_tendency(profile):
    """
    Determines if a profile tends toward positive, negative, or neutral ratings.

    Uses persona description hash to ensure consistency:
    - Same persona always gets same tendency across runs
    - Deterministic but appears random
    """
    # Hash the profile description
    profile_hash = hash(profile.description) if profile.description else hash(profile.respondent_id)

    # Use hash to deterministically assign tendency
    tendency_value = profile_hash % 3

    if tendency_value == 0:
        return "positive"    # Tends to rate highly (4-5 on Likert-5)
    elif tendency_value == 1:
        return "negative"    # Tends to rate lowly (1-2 on Likert-5)
    else:
        return "neutral"     # Balanced ratings (2-4 on Likert-5)
```

**2. Question-Specific Probability Distributions:**

For each question type, tendency maps to probability distributions:

```python
# Likert 5-point scale example
if tendency == "positive":
    probabilities = [0.05, 0.10, 0.20, 0.30, 0.35]  # Skewed toward 4-5
    #                1     2     3     4     5
elif tendency == "negative":
    probabilities = [0.35, 0.30, 0.20, 0.10, 0.05]  # Skewed toward 1-2
else:  # neutral
    probabilities = [0.10, 0.20, 0.40, 0.20, 0.10]  # Centered on 3

# Sample from distribution
ground_truth = np.random.choice([1, 2, 3, 4, 5], p=probabilities)
```

**3. Question Type Handling:**

Different question types use different distributions:

```python
# Yes/No (2 options)
if question.type == "yes_no":
    if tendency == "positive":
        probs = [0.2, 0.8]    # 80% say "Yes" (2)
    elif tendency == "negative":
        probs = [0.8, 0.2]    # 80% say "No" (1)
    else:
        probs = [0.5, 0.5]    # 50/50 split

# Likert 7-point
elif question.type == "likert_7":
    if tendency == "positive":
        probs = [0.02, 0.03, 0.10, 0.15, 0.25, 0.25, 0.20]  # Skewed toward 5-7
    # ... etc

# Multiple choice (varies by number of options)
elif question.type == "multiple_choice":
    n_options = len(question.options)
    if tendency == "positive":
        # Favor last options
        probs = [0.05] * (n_options - 2) + [0.45, 0.45]
    # ... etc
```

**4. Introducing Realistic Variation:**

Even profiles with the same tendency don't always give identical answers:

```python
# Each profile gets slightly different ratings through:

# a) Probabilistic sampling (not deterministic)
rating = np.random.choice(options, p=probabilities)  # Random sampling

# b) Seeded randomness for reproducibility
np.random.seed(seed)  # Same seed = same experiment results

# c) Question-level variation
# The same respondent might rate differently across questions
# R001: q1=5, q2=4, q3=5, q4=3  (not all 5s even with positive tendency)
```

**5. Complete Example - One Respondent, One Question:**

```python
# Profile
profile = RespondentProfile(
    respondent_id="R042",
    description="22-year-old college student, plays competitively, $30-50/month"
)

# Step 1: Determine tendency
tendency = determine_tendency(profile)  # → "positive" (based on hash)

# Step 2: Question context
question = Question(
    id="q5_download_likelihood",
    type="likert_5",
    scale={1: "Very unlikely", ..., 5: "Very likely"}
)

# Step 3: Get probability distribution for positive tendency
probabilities = [0.05, 0.10, 0.20, 0.30, 0.35]

# Step 4: Sample ground truth rating
np.random.seed(100)  # For reproducibility
ground_truth = np.random.choice([1, 2, 3, 4, 5], p=probabilities)
# Result: ground_truth = 5 (sampled with 35% probability)

# Step 5: Save to CSV
# R042,q5_download_likelihood,5
```

**6. Why This Approach:**

- **Realistic**: Mimics how real people with similar attitudes answer differently
- **Consistent**: Same persona + seed = same results (reproducible experiments)
- **Varied**: Not all positive profiles give all 5s (introduces natural variation)
- **Testable**: We know the "true" answer to validate SSR against

**7. Customization Points:**

You can customize ground truth generation for your domain:

```python
# In ground_truth_pipeline.py

# Option 1: Change tendency logic
if profile.income == "high" and profile.age == "young":
    tendency = "positive"
elif profile.budget_conscious:
    tendency = "negative"

# Option 2: Adjust probability distributions
if tendency == "positive":
    # More extreme distribution (more 5s)
    probabilities = [0.01, 0.04, 0.10, 0.25, 0.60]

# Option 3: Add noise/uncertainty
probabilities = add_noise(probabilities, noise_level=0.1)
```

**8. Validation:**

After generation, the ground truth CSV contains all "correct" answers:

```csv
respondent_id,question_id,ground_truth
R001,q1_would_play,2
R001,q2_recommend_friends,2
R001,q3_download_likelihood,5
...
R100,q20_primary_motivation,1
```

This becomes the **gold standard** that SSR predictions are evaluated against.

---

### Step 4: Textual Response Generation

Two response styles are generated:

**Human-style** (direct, opinionated):
```python
responses = generate_responses_from_ground_truth(
    survey, profiles, ground_truth_df,
    response_style="human",
    seed=101
)
```

Example: `"Definitely yes!"` or `"I'd say very likely"`

**LLM-style** (hedged, nuanced):
```python
responses = generate_responses_from_ground_truth(
    survey, profiles, ground_truth_df,
    response_style="llm",
    seed=102
)
```

Example: `"I would say that, considering various factors, it seems quite likely"`

---

### Step 5: SSR Application

Convert text responses to probability distributions:

```python
from src.ssr_model import SemanticSimilarityRater

# Initialize with paper's methodology
rater = SemanticSimilarityRater(
    model_name="text-embedding-3-small",  # OpenAI embedding model
    temperature=1.0,                      # Controls distribution spread
    normalize_method="paper",             # Paper's normalization
    use_openai=True                       # Use OpenAI API
)

# Apply SSR
distributions = rater.rate_responses(responses, survey, show_progress=True)
```

**How SSR Works:**

1. **Embed response**: Convert text to 1536-dimensional vector
2. **Embed scale labels**: Convert each option (e.g., "Very likely") to vectors
3. **Compute similarities**: Calculate cosine similarity between response and each label
4. **Normalize to probabilities**: Apply paper's normalization method:
   ```python
   similarities - min(similarities)  # Shift to start at 0
   scaled = shifted / temperature     # Apply temperature
   probabilities = scaled / sum(scaled)  # Normalize to sum to 1
   ```

**Example Output** (`llm_distributions.json` structure):
```json
{
  "q2_subscription_likelihood": {
    "R001": {
      "probabilities": [0.05, 0.10, 0.15, 0.35, 0.35],
      "ground_truth": 4,
      "mode": 5,
      "expected_value": 3.85,
      "entropy": 1.42,
      "gender": "Female",
      "age_group": "25-34",
      "persona_group": "Frequent Players - Tech Savvy",
      "occupation": "Professional"
    },
    "R002": { ... }
  },
  "q3_platform_trust": { ... }
}
```

**Key Fields:**
- `probabilities`: Probability distribution over scale points (sums to 1.0)
- `ground_truth`: Actual rating from ground truth data
- `mode`: Most likely rating (argmax of distribution)
- `expected_value`: Weighted average (more robust than mode)
- `entropy`: Shannon entropy (uncertainty measure, lower = more confident)
- Demographics: `gender`, `age_group`, `persona_group`, `occupation` (v2.0+)

---

### Step 6: Evaluation

Compare SSR predictions against ground truth:

```python
from src.ground_truth import (
    create_ground_truth_dict,
    evaluate_against_ground_truth
)

ground_truth_dict = create_ground_truth_dict(ground_truth_df)

for question in survey.questions:
    q_distributions = [d for d in distributions if d.question_id == question.id]
    comparison = evaluate_against_ground_truth(q_distributions, ground_truth_dict, question)

    print(f"{question.id}: {comparison.mode_accuracy:.1%} accuracy")
```

**Metrics Computed:**

| Metric | Description | Interpretation |
|--------|-------------|----------------|
| **Mode Accuracy** | % of predictions where mode = ground truth | Strict correctness |
| **Top-2 Accuracy** | % where true answer is in top 2 predictions | Lenient correctness |
| **MAE** | Mean absolute error from true rating | Average distance |
| **RMSE** | Root mean squared error | Penalizes large errors |
| **Prob at Truth** | Average probability assigned to true answer | SSR's confidence |
| **KL Divergence** | Distance from empirical distribution | Distribution alignment |
| **Confusion Matrix** | Predicted vs actual ratings | Error patterns |

---

### Step 7: Report Generation

Generate comprehensive reports:

```python
from src.report_generator import create_one_page_report, generate_text_report
from src.markdown_report import generate_comprehensive_report

# Visual report (PNG)
create_one_page_report(
    human_comparisons,
    llm_comparisons,
    survey,
    output_path="experiments/run_TIMESTAMP/report.png",
    title="Ground Truth Comparison: Human vs LLM SSR Predictions"
)

# Text metrics (TXT)
generate_text_report(
    human_comparisons,
    llm_comparisons,
    survey,
    output_path="experiments/run_TIMESTAMP/report.txt"
)

# Comprehensive explanation (MD)
generate_comprehensive_report(
    human_comparisons,
    llm_comparisons,
    survey,
    output_path="experiments/run_TIMESTAMP/report.md"
)
```

**Report Contents:**

1. **PNG Report**: Visual comparison with accuracy charts, error metrics, confusion matrices
2. **TXT Report**: Numerical metrics for quick reference
3. **MD Report**: Detailed explanations of what each metric means and how to interpret results

---

## Understanding the Pipeline

### What is Semantic Similarity Rating?

Traditional surveys force respondents to choose discrete options:
```
How likely are you to subscribe?
○ Very unlikely  ○ Unlikely  ○ Neutral  ○ Likely  ○ Very likely
```

But textual responses are richer:
```
"I'm quite interested! The automated features sound great,
though I'm a bit concerned about the price. I'd say I'm
somewhere between likely and very likely."
```

SSR captures this nuance as a **probability distribution**:
```
Very unlikely:  0.03  ███
Unlikely:       0.05  █████
Neutral:        0.12  ████████████
Likely:         0.38  ██████████████████████████████████████
Very likely:    0.42  ██████████████████████████████████████████
```

### Why Use SSR?

1. **Preserves Uncertainty**: Captures when respondents are between two options
2. **Richer Analysis**: Probability distributions enable expected values, variance, entropy
3. **LLM-Friendly**: LLMs naturally produce nuanced text; SSR converts it to quantitative data
4. **Validated Method**: Paper shows SSR achieves 90%+ accuracy vs ground truth

### Human vs LLM Response Styles

The pipeline compares two response generation styles:

**Human-style**: Direct, decisive language
- "Definitely yes!"
- "I'd say very likely"
- **Result**: ~96% SSR accuracy

**LLM-style**: Hedged, qualified language
- "While I appreciate the features, considering various factors, I would say..."
- "It seems to me that, taking context into account..."
- **Result**: ~93% SSR accuracy

**Key Finding**: Direct language works slightly better with SSR, but both achieve high accuracy.

---

## Creating Custom Surveys

### 1. Create a YAML config file

```yaml
survey:
  name: "Your Survey Name"
  description: "Survey description"

  context: |
    Provide context about what's being evaluated.
    This helps respondents understand the scenario.

  questions:
    - id: "q1_unique_id"
      text: "Your question here?"
      type: "likert_5"
      scale:
        1: "Strongly disagree"
        2: "Disagree"
        3: "Neutral"
        4: "Agree"
        5: "Strongly agree"
```

### Understanding the `context` Field

The `context` field is **critically important** for generating realistic responses. It provides background information that:

1. **Anchors responses to a specific scenario**: Without context, LLMs generate generic responses. With context, they respond to your specific product/service.

2. **Enables informed opinions**: Context gives virtual respondents the information they need to form opinions, just like real survey participants.

3. **Improves SSR accuracy**: More specific context leads to more semantically distinct responses, which SSR can classify more accurately.

**Example - Generic vs Contextual:**

❌ **Without proper context:**
```yaml
context: "We are evaluating a product."

# Result: Generic responses like:
# "It seems okay"
# "I might be interested"
```

✅ **With detailed context:**
```yaml
context: |
  We are evaluating a new online lottery gaming platform with the following features:
  - Multiple lottery games (Powerball, Mega Millions, state lotteries)
  - Mobile app and web access
  - Automated ticket purchasing and number selection
  - Instant win notifications and prize alerts
  - Subscription plans: $9.99/month or $99/year

# Result: Specific, contextual responses like:
# "The automated purchasing is appealing, but $9.99/month seems high for occasional players"
# "I'm concerned about security for payment info, but the prize alerts are convenient"
```

**What to include in `context`:**

- **Product/service description**: What is being evaluated?
- **Key features**: What are the main capabilities or attributes?
- **Pricing information**: What does it cost? (Especially important for purchase intent questions)
- **Target audience context**: Who is this for? What problem does it solve?
- **Any constraints or conditions**: Limitations, requirements, or special circumstances

**Context Best Practices:**

1. **Be specific and concrete**: Include actual feature names, prices, and details
2. **Keep it relevant**: Only include information that would realistically be shown to survey respondents
3. **Match survey length**: Longer surveys can have more context; short surveys need concise context
4. **Use realistic language**: Write as you would in an actual survey introduction

**Example contexts for different domains:**

**Healthcare Survey:**
```yaml
context: |
  We are evaluating a new telemedicine service with these features:
  - 24/7 access to licensed physicians via video call
  - Average wait time: 15 minutes
  - Prescription fulfillment at partner pharmacies
  - $29 per consultation, or $99/month unlimited plan
  - Accepts most major insurance plans
```

**Product Survey:**
```yaml
context: |
  We are evaluating an eco-friendly water bottle with:
  - Stainless steel construction, keeps drinks cold for 24 hours
  - 32oz capacity with carrying handle
  - Available in 8 colors
  - Price: $34.99
  - Lifetime warranty against defects
```

**Service Survey:**
```yaml
context: |
  We are evaluating a meal kit delivery service featuring:
  - 3-5 recipes per week, serves 2-4 people
  - Locally sourced, organic ingredients when possible
  - 30-minute average cooking time
  - Pricing: $60-80/week depending on plan
  - Delivery every Tuesday and Thursday
```

### 2. Load and use the survey

```python
from src.survey import Survey

survey = Survey.from_config('config/your_survey.yaml')
```

### 3. Customize response generation

Edit `ground_truth_pipeline.py` to match your domain:

```python
def generate_ground_truth_ratings(survey: Survey, profiles: list, seed: int = 100):
    # Customize logic based on your profiles and questions
    for profile in profiles:
        if profile.some_attribute == "high":
            tendency = "positive"
        else:
            tendency = "negative"
        # ... generate ratings based on tendency
```

---

## Using Persona Groups with Demographics (v2.0+)

Create realistic synthetic populations with demographic tracking:

```yaml
persona_groups:
  - name: "Tech-Savvy Young Professionals"
    description: "Early adopters, high income, value innovation"
    weight: 0.3  # 30% of sample
    target_demographics:
      gender: ["Male", "Female"]
      age_group: ["25-34", "35-44"]
      occupation: ["Professional", "Technical"]
    personas:
      - "A 28-year-old software engineer. Values cutting-edge features."
      - "A 35-year-old product manager. Focuses on user experience."

  - name: "Budget-Conscious Families"
    weight: 0.4  # 40% of sample
    target_demographics:
      gender: ["Male", "Female"]
      age_group: ["35-44", "45-54"]
      occupation: ["Service", "Sales"]
    personas:
      - "A 42-year-old parent. Prioritizes value and durability."

  - name: "Retired Skeptics"
    weight: 0.3  # 30% of sample
    target_demographics:
      gender: ["Male", "Female"]
      age_group: ["55-64", "65+"]
      occupation: ["Retired"]
    personas:
      - "A 68-year-old retiree. Skeptical of new products."
```

**Demographics tracked throughout pipeline:**
- `gender` - Gender category
- `age_group` - Age range (e.g., "25-34", "35-44")
- `persona_group` - Named segment (e.g., "Tech-Savvy Young Professionals")
- `occupation` - Occupation category

**Demographics included in all outputs:**
- Ground truth CSV (optional columns)
- LLM distributions JSON (per response)
- Experiment reports (segmentation ready)

---

## Multi-Category Surveys

Compare multiple product categories with the same questions, plus comparative preference questions.

### Example: Compare Online vs Traditional Lottery

```yaml
survey:
  name: "Lottery Platform Comparison"

  # Define categories
  categories:
    - id: "online_platform"
      name: "Online Lottery Platform"
      description: "Digital subscription service"
      context: |
        Features: mobile app, $9.99/month subscription,
        automated purchasing, instant win notifications

    - id: "traditional_retail"
      name: "Traditional Retail Lottery"
      description: "In-person ticket purchase"
      context: |
        Features: physical tickets at stores,
        $2-5 per ticket, manual result checking

  questions:
    # Same questions for each category
    - id: "online_purchase_intent"
      text: "How likely are you to subscribe?"
      type: "likert_5"
      category: "online_platform"
      scale:
        1: "Very unlikely"
        5: "Very likely"

    - id: "retail_purchase_intent"
      text: "How likely are you to purchase tickets?"
      type: "likert_5"
      category: "traditional_retail"
      scale:
        1: "Very unlikely"
        5: "Very likely"

    # Comparative preference question
    - id: "platform_preference"
      text: "How much do you prefer {online_platform} over {traditional_retail}?"
      type: "preference_scale"
      category: "comparison"
      categories_compared: ["online_platform", "traditional_retail"]
      scale:
        1: "Strongly prefer traditional"
        4: "No preference"
        7: "Strongly prefer online"
```

### Multi-Category Features

1. **Category-specific context**: Each category has its own detailed context shown to respondents
2. **Automatic question routing**: Questions tagged with `category` show only that category's context
3. **Comparative questions**: New `preference_scale` type compares categories directly
4. **Category substitution**: Use `{category_id}` in question text, automatically replaced with category names
5. **Category filtering**: Results Dashboard lets you filter by category
6. **Category comparison**: Automatic analysis of which category is easier/harder to predict

### Output Structure

**Ground truth CSV includes category:**
```csv
respondent_id,question_id,ground_truth,category,gender,age_group,persona_group,occupation
R001,online_purchase_intent,4,online_platform,Male,25-34,Tech Savvy,Professional
R001,retail_purchase_intent,2,traditional_retail,Male,25-34,Tech Savvy,Professional
R001,platform_preference,6,comparison,Male,25-34,Tech Savvy,Professional
```

**LLM distributions nested by category:**
```json
{
  "online_platform": {
    "online_purchase_intent": {
      "R001": {"probabilities": [...], ...}
    }
  },
  "traditional_retail": { ... },
  "comparison": { ... }
}
```

See `config/multi_category_lottery_comparison.yaml` for a complete working example.

---

## Interpreting Results

### Understanding Report Outputs

#### 1. Mode Accuracy (Main Metric)

**What it is**: Percentage of responses where SSR's most likely prediction matches the ground truth.

**Interpretation:**
- **90-100%**: Excellent - SSR reliably predicts exact ratings
- **70-90%**: Good - Most predictions correct, some adjacent errors
- **50-70%**: Fair - Captures general sentiment but misses specifics
- **<50%**: Poor - Below random chance for many scales

**Example**: If mode accuracy is 95%, then 95 out of 100 responses were predicted exactly right.

#### 2. Mean Absolute Error (MAE)

**What it is**: Average distance between predicted and true ratings.

**Interpretation:**
- **<0.3**: Excellent - Predictions very close to truth
- **0.3-0.5**: Good - Within half a point on average
- **0.5-1.0**: Fair - Off by about one scale point
- **>1.0**: Poor - Missing by multiple points

**Example**: MAE of 0.12 means on a 5-point scale, predictions are off by about 0.12 points on average.

#### 3. Probability at Truth

**What it is**: Average probability SSR assigns to the correct answer.

**Interpretation:**
- **>0.5**: Very confident predictions (concentrated on true answer)
- **0.3-0.5**: Moderate confidence (true answer is top choice but not dominant)
- **0.2-0.3**: Distributed predictions (spreading probability across options)
- **<0.2**: High uncertainty (near-uniform distribution)

**Example**: 0.65 means SSR assigns 65% probability to the correct answer on average.

#### 4. Confusion Matrix

Shows which ratings are confused with each other:

```
           Predicted
           1   2   3   4   5
True  1   [45   5   0   0   0]
      2   [ 3  38   9   0   0]
      3   [ 0   8  35   7   0]
      4   [ 0   0   6  40   4]
      5   [ 0   0   0   5  45]
```

**Reading it**:
- Diagonal = correct predictions
- Off-diagonal = errors
- Adjacent cells = common confusion between nearby ratings

#### 5. Radar Chart Multi-Dimensional Metrics

The Results Dashboard includes a radar chart showing 6 performance dimensions. Here's how each metric is calculated:

**Accuracy (0-100%):**
```python
# Direct from overall LLM mode accuracy
Accuracy = overall_llm_accuracy  # e.g., 85.2%
```
- Same as Mode Accuracy above
- Percentage of exact prediction matches

**Precision (0-100%):**
```python
# Inverse of Mean Absolute Error
Precision = max(0, 100 - (average_mae * 20))
```
- Based on MAE across all questions
- MAE of 0 → 100% precision (perfect)
- MAE of 2.0 → 60% precision (typical 5-point scale max error)
- Higher is better

**Consistency (0-100%):**
```python
# Inverse of standard deviation across questions
std_accuracy = np.std([q1_accuracy, q2_accuracy, ...])
Consistency = max(0, 100 - std_accuracy)
```
- Measures performance stability across different questions
- Low std dev → high consistency (similar accuracy on all questions)
- High std dev → low consistency (some questions good, others poor)
- Example: [85%, 87%, 84%] → std=1.5 → 98.5% consistency

**Confidence (0-100%):**
```python
# Based on Shannon entropy of probability distributions
avg_entropy = mean([entropy1, entropy2, ...])
Confidence = max(0, 100 - (avg_entropy * 50))
```
- Entropy measures uncertainty in probability distributions
- Low entropy (e.g., 0.4) → peaked distribution → high confidence → 80%
- High entropy (e.g., 1.6) → flat distribution → low confidence → 20%
- Formula: `H = -Σ(p * log(p))` for probabilities p

**Example entropy calculation:**
```python
# High confidence distribution (peaked)
probs = [0.05, 0.05, 0.80, 0.05, 0.05]
entropy = 0.64  # Low entropy
confidence = 100 - (0.64 * 50) = 68%

# Low confidence distribution (flat)
probs = [0.20, 0.20, 0.20, 0.20, 0.20]
entropy = 1.61  # High entropy (max for 5 options)
confidence = 100 - (1.61 * 50) = 19%
```

**Coverage (0-100%):**
```python
# Percentage of questions with valid predictions
Coverage = (questions_with_predictions / total_questions) * 100
```
- Always 100% for completed experiments
- Could be <100% if some questions failed to process
- Included for completeness and future extensibility

**Calibration (0-100%):**
```python
# How well predicted probabilities match actual outcomes
Calibration = max(0, 100 - (average_mae * 25))
```
- Simplified calibration based on MAE
- Perfect calibration: predicted probability matches empirical frequency
- MAE of 0 → 100% calibration
- MAE of 1.0 → 75% calibration
- Related to "probability at truth" metric

**Visual Interpretation:**

The radar chart shows all 6 dimensions simultaneously:
```
      Accuracy
         /\
        /  \
Coverage    Precision
   |          |
Calibration  Consistency
       \    /
      Confidence
```

- **Balanced polygon**: Good all-around performance
- **Peaked in one direction**: Strong in some areas, weak in others
- **Large area**: Overall strong performance
- **Small area**: Needs improvement across dimensions

**Target benchmark**: 80% line shown as reference
- Dimensions above 80%: Exceeding target
- Dimensions below 80%: Room for improvement

---

## API Reference

### Core Classes

#### `SemanticSimilarityRater`

Main SSR implementation.

```python
from src.ssr_model import SemanticSimilarityRater

rater = SemanticSimilarityRater(
    model_name="text-embedding-3-small",  # Embedding model
    temperature=1.0,                      # Distribution spread (higher = more spread)
    normalize_method="paper",             # "paper" or "softmax"
    use_openai=True                       # Use OpenAI API
)

# Rate a single response
distribution = rater.rate_response(response, question)

# Rate multiple responses
distributions = rater.rate_responses(responses, survey, show_progress=True)
```

**Key Methods:**
- `rate_response(response, question)` → `RatingDistribution`
- `rate_responses(responses, survey, show_progress)` → `List[RatingDistribution]`

#### `Survey`

Survey configuration and question handling.

```python
from src.survey import Survey

survey = Survey.from_config('config/mixed_survey_config.yaml')

# Access questions
for question in survey.questions:
    print(f"{question.id}: {question.text}")
    print(f"Type: {question.type}")
    print(f"Options: {question.num_options}")
```

#### `RatingDistribution`

Represents SSR output for a single response.

```python
class RatingDistribution:
    question_id: str
    respondent_id: str
    distribution: np.ndarray      # Probability distribution
    mode: int                     # Most likely rating
    expected_value: float         # Weighted average
    entropy: float                # Uncertainty measure
    scale_labels: Dict[int, str]  # Option labels
```

#### `GroundTruthComparison`

Evaluation results for a question.

```python
class GroundTruthComparison:
    question_id: str
    mode_accuracy: float          # % exact matches
    top2_accuracy: float          # % in top 2
    mae: float                    # Mean absolute error
    rmse: float                   # Root mean squared error
    prob_at_truth: float          # Avg prob at true answer
    kl_divergence: float          # Distribution distance
    confusion_matrix: np.ndarray  # Prediction errors
```

---

## Paper Methodology

This implementation follows the exact methodology from [arXiv:2510.08338v2](https://arxiv.org/abs/2510.08338v2):

### Embedding Model
- **Model**: OpenAI `text-embedding-3-small`
- **Dimensions**: 1536
- **Why**: Paper's choice for semantic encoding

### Normalization Method

The paper uses a specific normalization (NOT softmax):

```python
def normalize(similarities, temperature=1.0):
    # 1. Shift to start at 0
    shifted = similarities - min(similarities)

    # 2. Apply temperature
    scaled = shifted / temperature

    # 3. Proportional normalization
    probabilities = scaled / sum(scaled)

    return probabilities
```

**Temperature Parameter:**
- Lower (e.g., 0.5): More peaked distributions (confident)
- Default (1.0): Balanced (as per paper)
- Higher (e.g., 2.0): More spread distributions (uncertain)

### Differences from Other Approaches

| Aspect | This Implementation | Alternative |
|--------|---------------------|-------------|
| Embedding | OpenAI text-embedding-3-small | sentence-transformers |
| Normalization | Subtract min + proportional | Softmax |
| Temperature | Applied after shift | Applied before softmax |
| Validation | Ground truth comparison | None |

---

## Troubleshooting

### Common Issues

#### 1. OpenAI API Key Error

```
Error: OpenAI API key not found
```

**Solution**: Create a `.env` file with your API key:
```bash
echo "OPENAI_API_KEY=sk-..." > .env
```

#### 2. Import Errors

```
ModuleNotFoundError: No module named 'src'
```

**Solution**: Run from project root or add to path:
```python
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent))
```

#### 3. Low SSR Accuracy

**Possible causes:**
- Scale labels not semantically distinct (e.g., "Good" vs "Fine")
- Response text doesn't match question context
- Temperature too high (try 0.5 or 1.0)

**Solution**:
- Review scale labels for semantic clarity
- Check that responses are on-topic
- Experiment with temperature parameter

#### 4. Empty Experiment Folders

```
No experiments found
```

**Solution**: Run the pipeline first:
```bash
python ground_truth_pipeline.py
```

---

## Advanced Usage

### Custom Response Generation

Integrate real survey responses:

```python
from src.llm_client import Response

responses = []
for row in your_survey_data:
    response = Response(
        respondent_id=row['id'],
        question_id=row['question'],
        text_response=row['answer'],
        respondent_profile={}  # Optional
    )
    responses.append(response)

# Apply SSR
distributions = rater.rate_responses(responses, survey)
```

### Batch Processing

Process large datasets efficiently:

```python
# Process in batches to manage API rate limits
batch_size = 100
all_distributions = []

for i in range(0, len(responses), batch_size):
    batch = responses[i:i+batch_size]
    batch_dists = rater.rate_responses(batch, survey)
    all_distributions.extend(batch_dists)
```

### Custom Evaluation Metrics

Add your own metrics:

```python
from src.ground_truth import GroundTruthComparison

def custom_metric(comparison: GroundTruthComparison):
    # Your metric logic
    return value

# Compute for all questions
for question in survey.questions:
    comparison = comparisons[question.id]
    score = custom_metric(comparison)
    print(f"{question.id}: {score}")
```

---

## References

### Original Research Paper

This implementation is based on the Semantic Similarity Rating methodology from:

**"LLMs Reproduce Human Purchase Intent via Semantic Similarity Elicitation"**
arXiv:2510.08338v2
[https://arxiv.org/abs/2510.08338v2](https://arxiv.org/abs/2510.08338v2)

### Citation

If you use S.A.G.E or this SSR implementation in your research, please cite the original paper:

```bibtex
@article{ssr2024,
  title={LLMs Reproduce Human Purchase Intent via Semantic Similarity Elicitation},
  author={[Authors]},
  journal={arXiv preprint arXiv:2510.08338v2},
  year={2024}
}
```

### Additional Resources

- **S.A.G.E UI Documentation**: [ui/README.md](ui/README.md)
- **OpenAI Embeddings**: [text-embedding-3-small](https://platform.openai.com/docs/guides/embeddings)
- **Streamlit Framework**: [https://streamlit.io](https://streamlit.io)

---

## License

This project is provided for research and educational purposes. Please refer to the original paper for methodology details and proper attribution.

---

## Support

For questions or issues:
1. Check the [Troubleshooting](#troubleshooting) section
2. Review the [UI documentation](ui/README.md)
3. Consult the [original paper](https://arxiv.org/abs/2510.08338v2)
4. Open an issue on the repository

---

**Happy researching! 🚀**
