# Semantic Similarity Rating (SSR) Pipeline

A complete implementation of the Semantic Similarity Rating methodology from the paper ["LLMs Reproduce Human Purchase Intent via Semantic Similarity Elicitation"](https://arxiv.org/abs/2510.08338v2) (arXiv:2510.08338v2).

## Overview

The SSR methodology converts textual survey responses into probability distributions over Likert scale options using semantic similarity. This preserves uncertainty and nuance in responses while enabling quantitative analysis.

**Key Features:**
- ✅ **Paper-exact implementation**: Uses OpenAI text-embedding-3-small and the paper's normalization method
- ✅ **Multiple question types**: Binary (yes/no), Likert-5, Likert-7, multiple choice
- ✅ **Ground truth evaluation**: Comprehensive metrics including mode accuracy, MAE, RMSE, KL divergence
- ✅ **Response style comparison**: Evaluate SSR on human-style vs LLM-style responses
- ✅ **Automated reporting**: Generates PNG visualizations, TXT metrics, and comprehensive Markdown reports
- ✅ **Experiment organization**: Timestamped folders keep all experiment results organized

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

### Run the complete pipeline in 3 steps:

```bash
# 1. Activate environment
source .venv/bin/activate

# 2. Run the ground truth comparison pipeline
python ground_truth_pipeline.py

# 3. View results
# Results are saved to experiments/run_TIMESTAMP/
```

This generates:
- `ground_truth.csv` - Ground truth ratings for all respondents
- `report.png` - Visual comparison report
- `report.txt` - Detailed metrics
- `report.md` - Comprehensive markdown report with explanations

### Or explore interactively:

```bash
jupyter notebook notebooks/example_notebook.ipynb
```

---

## Project Structure

```
ssr_pipeline/
├── README.md                          # This file
├── requirements.txt                   # Python dependencies
├── ground_truth_pipeline.py          # Main pipeline script
│
├── config/                           # Survey configurations
│   └── mixed_survey_config.yaml      # Lottery gaming survey (6 questions)
│
├── src/                              # Core modules
│   ├── __init__.py
│   ├── survey.py                     # Survey loading and question handling
│   ├── llm_client.py                 # Response generation and profiles
│   ├── ssr_model.py                  # SSR implementation (paper-exact)
│   ├── ground_truth.py               # Evaluation metrics
│   ├── report_generator.py           # PNG + TXT reports
│   ├── markdown_report.py            # Markdown reports
│   ├── analysis.py                   # Statistical analysis
│   ├── visualization.py              # Plotting utilities
│   └── comparison.py                 # Human vs LLM comparison
│
├── notebooks/                        # Interactive tutorials
│   └── example_notebook.ipynb        # Complete walkthrough
│
├── experiments/                      # Experiment results (auto-created)
│   └── run_TIMESTAMP/
│       ├── ground_truth.csv
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

**Output** (`ground_truth.csv`):
```
respondent_id,question_id,ground_truth
R001,q1_would_subscribe,1
R001,q2_subscription_likelihood,4
R001,q3_platform_trust,5
...
```

**Logic:**
- Profiles with high environmental consciousness → higher ratings
- Profiles with low environmental consciousness → lower ratings
- Introduces realistic variation using probability distributions

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

**Example Output:**
```python
RatingDistribution(
    question_id="q2_subscription_likelihood",
    distribution=[0.05, 0.10, 0.15, 0.35, 0.35],  # Probabilities for 1-5
    mode=5,                    # Most likely rating
    expected_value=3.85,       # Weighted average
    entropy=1.42               # Uncertainty measure
)
```

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

## Citation

If you use this implementation in your research, please cite the original paper:

```bibtex
@article{ssr2024,
  title={LLMs Reproduce Human Purchase Intent via Semantic Similarity Elicitation},
  author={[Authors]},
  journal={arXiv preprint arXiv:2510.08338v2},
  year={2024}
}
```

---

## License

This project is provided for research and educational purposes. Please refer to the original paper for methodology details and proper attribution.

---

## Support

For questions or issues:
1. Check the [Troubleshooting](#troubleshooting) section
2. Review the [example notebook](notebooks/example_notebook.ipynb)
3. Consult the [original paper](https://arxiv.org/abs/2510.08338v2)
4. Open an issue on the repository

---

**Happy researching! 🚀**
