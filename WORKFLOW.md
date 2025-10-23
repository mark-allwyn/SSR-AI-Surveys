# SSR Pipeline Code & File Workflow Explanation

This document explains exactly how the code works, what each file does, and how data flows through the system.

---

## Table of Contents

1. [High-Level Overview](#high-level-overview)
2. [File-by-File Explanation](#file-by-file-explanation)
3. [Complete Data Flow](#complete-data-flow)
4. [Step-by-Step Code Execution](#step-by-step-code-execution)
5. [How Files Work Together](#how-files-work-together)
6. [Understanding the Data Structures](#understanding-the-data-structures)
7. [Real Example Walkthrough](#real-example-walkthrough)

---

## High-Level Overview

### The Big Picture

```
┌─────────────────────────────────────────────────────────────────┐
│                    SEMANTIC SIMILARITY RATING                    │
│                                                                   │
│  Input: "I'm very interested in this platform!"                  │
│                           ↓                                       │
│  SSR Magic: Convert text → probability distribution              │
│                           ↓                                       │
│  Output: [0.05, 0.10, 0.15, 0.35, 0.35]                         │
│          (probabilities for ratings 1-5)                         │
└─────────────────────────────────────────────────────────────────┘
```

### What This Pipeline Does

1. **Loads a survey** with multiple questions (yes/no, Likert, multiple choice)
2. **Generates synthetic respondents** with different attitudes
3. **Creates ground truth ratings** based on respondent profiles
4. **Generates text responses** that match the ground truth (both human-style and LLM-style)
5. **Applies SSR** to convert text back to probability distributions
6. **Evaluates accuracy** by comparing SSR predictions to ground truth
7. **Generates reports** showing how well SSR performed

### Why This Matters

This validates that SSR actually works! If we generate a response that should be "Very likely", does SSR correctly predict that? The pipeline measures this.

---

## File-by-File Explanation

### Main Files

#### 1. `ground_truth_pipeline.py` - The Orchestrator

**What it does**: Runs the complete experiment from start to finish

**Key functions**:

```python
def generate_ground_truth_ratings(survey, profiles, seed):
    """
    Creates a CSV with the 'true' answers for each respondent/question.

    Input:
      - survey: Survey object with 6 questions
      - profiles: List of 50 RespondentProfile objects
      - seed: Random seed for reproducibility

    Output:
      - DataFrame with 300 rows (50 respondents × 6 questions)
      - Columns: respondent_id, question_id, ground_truth

    Logic:
      - If profile is "Very concerned" → likely to give high ratings (4-5)
      - If profile is "Not concerned" → likely to give low ratings (1-2)
      - Uses probability distributions to add realistic variation
    """

def generate_responses_from_ground_truth(survey, profiles, ground_truth_df, response_style):
    """
    Generates textual responses that align with ground truth.

    Input:
      - ground_truth_df: The 'answers' we want to generate text for
      - response_style: "human" or "llm"

    Output:
      - List of 300 Response objects with text_response field

    Example:
      Ground truth = 5 (Very likely)
      Human style: "Definitely yes!"
      LLM style: "I would say that, considering the features, it seems very likely"
    """

def main():
    """
    The main execution flow - runs everything in order.

    Creates folder: experiments/run_TIMESTAMP/
    Saves 4 files:
      1. ground_truth.csv
      2. report.png
      3. report.txt
      4. report.md
    """
```

**Execution flow**:
```python
# Step 1: Setup
timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
experiment_dir = Path(f"experiments/run_{timestamp}")

# Step 2: Load survey
survey = Survey.from_config('config/mixed_survey_config.yaml')

# Step 3: Generate profiles
profiles = generate_diverse_profiles(50)

# Step 4: Generate ground truth
ground_truth_df = generate_ground_truth_ratings(survey, profiles)
# → Saves to: experiments/run_TIMESTAMP/ground_truth.csv

# Step 5: Generate human responses
human_responses = generate_responses_from_ground_truth(
    survey, profiles, ground_truth_df, response_style="human"
)

# Step 6: Generate LLM responses
llm_responses = generate_responses_from_ground_truth(
    survey, profiles, ground_truth_df, response_style="llm"
)

# Step 7: Apply SSR to both
rater = SemanticSimilarityRater(...)
human_distributions = rater.rate_responses(human_responses, survey)
llm_distributions = rater.rate_responses(llm_responses, survey)

# Step 8: Evaluate
ground_truth_dict = create_ground_truth_dict(ground_truth_df)
human_comparisons = {}
llm_comparisons = {}

for question in survey.questions:
    human_comp = evaluate_against_ground_truth(h_dists, ground_truth_dict, question)
    llm_comp = evaluate_against_ground_truth(l_dists, ground_truth_dict, question)

# Step 9: Generate reports
create_one_page_report(...)  # → report.png
generate_text_report(...)     # → report.txt
generate_comprehensive_report(...)  # → report.md
```

---

### Core Modules (`src/` folder)

#### 2. `src/survey.py` - Survey Configuration

**What it does**: Loads and manages survey questions

**Key classes**:

```python
class Question:
    """
    Represents a single survey question.

    Attributes:
      - id: Unique identifier (e.g., "q1_would_subscribe")
      - text: The question text
      - type: "yes_no", "likert_5", "likert_7", "multiple_choice"
      - num_options: Number of possible answers (2, 5, 7, etc.)
      - scale: Dictionary mapping numbers to labels

    Example:
      Question(
          id="q2_subscription_likelihood",
          text="How likely are you to subscribe?",
          type="likert_5",
          scale={1: "Very unlikely", 2: "Unlikely", ..., 5: "Very likely"}
      )
    """

    def get_reference_statements(self) -> Dict[int, str]:
        """
        Returns the scale labels that SSR will compare against.

        For yes_no: {1: "No", 2: "Yes"}
        For likert_5: {1: "Very unlikely", ..., 5: "Very likely"}

        This is what SSR uses to convert text to probabilities!
        """

class Survey:
    """
    Container for all questions and survey metadata.

    Attributes:
      - name: Survey name
      - description: What the survey is about
      - context: Background info shown to respondents
      - questions: List of Question objects
      - sample_size: Number of respondents (from config)
    """

    @classmethod
    def from_config(cls, config_path: str):
        """
        Loads a survey from YAML file.

        Reads: config/mixed_survey_config.yaml
        Returns: Survey object with 6 questions
        """
```

**Data flow**:
```
config/mixed_survey_config.yaml
         ↓
Survey.from_config()
         ↓
Survey object with 6 Question objects
         ↓
Used by: ground_truth_pipeline.py, ssr_model.py, notebooks
```

---

#### 3. `src/llm_client.py` - Response Generation

**What it does**: Generates respondent profiles and Response objects

**Key classes**:

```python
class RespondentProfile:
    """
    Represents a person taking the survey.

    Attributes:
      - environmental_consciousness: "Very concerned", "Not concerned", etc.

    This determines their 'ground truth' ratings.
    Very concerned → high ratings
    Not concerned → low ratings
    """

class Response:
    """
    A textual response to a question.

    Attributes:
      - respondent_id: "R001"
      - question_id: "q2_subscription_likelihood"
      - text_response: "I'm very interested in this platform!"
      - respondent_profile: Dict with profile info

    This is what gets fed into SSR!
    """

def generate_diverse_profiles(n: int) -> List[RespondentProfile]:
    """
    Creates n profiles with varying environmental consciousness.

    Input: n=50
    Output: 50 RespondentProfile objects

    Distribution:
      - 20% "Extremely concerned"
      - 20% "Very concerned"
      - 20% "Moderately concerned"
      - 20% "Slightly concerned"
      - 20% "Not concerned"
    """
```

**Data flow**:
```
generate_diverse_profiles(50)
         ↓
50 RespondentProfile objects
         ↓
generate_ground_truth_ratings() → determines ratings
         ↓
generate_responses_from_ground_truth() → creates text
         ↓
300 Response objects (50 people × 6 questions)
         ↓
Fed into SSR
```

---

#### 4. `src/ssr_model.py` - THE CORE SSR IMPLEMENTATION

**What it does**: Converts text responses to probability distributions

**This is the heart of the entire system!**

```python
class SemanticSimilarityRater:
    """
    Implements the paper's SSR methodology.

    Configuration:
      - model_name: "text-embedding-3-small" (OpenAI)
      - temperature: 1.0 (controls spread)
      - normalize_method: "paper" (subtract min + proportional)
      - use_openai: True (use OpenAI API)
    """

    def __init__(self, model_name, temperature, normalize_method, use_openai):
        """
        Sets up the embedding model.

        If use_openai=True:
          - Creates OpenAI client
          - Will use text-embedding-3-small API

        If use_openai=False:
          - Loads sentence-transformers model
          - Local computation (slower but no API cost)
        """

    def get_embeddings(self, texts: List[str]) -> np.ndarray:
        """
        Converts text to vectors.

        Input: ["I'm very interested", "Very likely", "Unlikely", ...]

        Process:
          1. Send to OpenAI API
          2. Get back 1536-dimensional vectors for each text

        Output: numpy array of shape (n_texts, 1536)

        Example:
          texts = ["Very likely", "Unlikely", "Neutral"]
          embeddings = get_embeddings(texts)
          # embeddings.shape = (3, 1536)
          # embeddings[0] = [0.023, -0.145, 0.389, ..., 0.021]  (1536 numbers)
        """

    def compute_similarities(self, response_embedding, label_embeddings):
        """
        Computes cosine similarity between response and each label.

        Input:
          - response_embedding: 1536-dimensional vector for "I'm very interested"
          - label_embeddings: 5 vectors for ["Very unlikely", ..., "Very likely"]

        Process:
          cosine_similarity = dot_product / (norm1 * norm2)

          Measures: How similar is response to each label?
          Range: -1 (opposite) to +1 (identical)

        Output: [0.2, 0.3, 0.5, 0.85, 0.92]
          → Response is most similar to "Very likely" (0.92)
        """

    def similarities_to_probabilities(self, similarities):
        """
        THE KEY ALGORITHM - Paper's normalization method

        Input: [0.2, 0.3, 0.5, 0.85, 0.92]

        Step 1: Shift to start at 0
          shifted = similarities - min(similarities)
          shifted = [0.2, 0.3, 0.5, 0.85, 0.92] - 0.2
          shifted = [0.0, 0.1, 0.3, 0.65, 0.72]

        Step 2: Apply temperature (t=1.0)
          scaled = shifted / temperature
          scaled = [0.0, 0.1, 0.3, 0.65, 0.72] / 1.0
          scaled = [0.0, 0.1, 0.3, 0.65, 0.72]

        Step 3: Normalize to probabilities
          probabilities = scaled / sum(scaled)
          probabilities = [0.0, 0.1, 0.3, 0.65, 0.72] / 1.77
          probabilities = [0.00, 0.06, 0.17, 0.37, 0.41]

        Output: [0.00, 0.06, 0.17, 0.37, 0.41]
          → 41% chance of "Very likely"
          → 37% chance of "Likely"
          → Sums to 1.0 ✓
        """

    def rate_response(self, response: Response, question: Question):
        """
        THE MAIN SSR FUNCTION - converts one text response to distribution

        Input:
          response.text_response = "I'm very interested in this platform!"
          question.scale = {1: "Very unlikely", ..., 5: "Very likely"}

        Process:
          1. Get embedding for response text
             response_emb = get_embeddings(["I'm very interested..."])

          2. Get embeddings for all scale labels
             labels = ["Very unlikely", "Unlikely", "Neutral", "Likely", "Very likely"]
             label_embs = get_embeddings(labels)

          3. Compute similarities
             sims = compute_similarities(response_emb, label_embs)
             # [0.2, 0.3, 0.5, 0.85, 0.92]

          4. Convert to probabilities
             probs = similarities_to_probabilities(sims)
             # [0.00, 0.06, 0.17, 0.37, 0.41]

          5. Create RatingDistribution object

        Output:
          RatingDistribution(
              question_id="q2_subscription_likelihood",
              respondent_id="R001",
              distribution=[0.00, 0.06, 0.17, 0.37, 0.41],
              mode=5,  # Most likely rating
              expected_value=4.12,  # Weighted average
              entropy=1.23,  # Uncertainty measure
              scale_labels={1: "Very unlikely", ..., 5: "Very likely"}
          )
        """

    def rate_responses(self, responses: List[Response], survey: Survey):
        """
        Applies SSR to multiple responses (batch processing).

        Input: 300 Response objects

        Process:
          For each response:
            1. Find the corresponding question from survey
            2. Call rate_response()
            3. Add to results list

        Output: 300 RatingDistribution objects

        This is what gets evaluated against ground truth!
        """
```

**Complete SSR Example**:

```python
# Input
response_text = "I'm very interested in this platform!"
scale_labels = {
    1: "Very unlikely",
    2: "Unlikely",
    3: "Neutral",
    4: "Likely",
    5: "Very likely"
}

# Step 1: Embed response → [0.023, -0.145, ..., 0.021] (1536 dims)
# Step 2: Embed labels → 5 vectors of 1536 dims each
# Step 3: Compute cosine similarities → [0.2, 0.3, 0.5, 0.85, 0.92]
# Step 4: Normalize → [0.00, 0.06, 0.17, 0.37, 0.41]

# Output
distribution = [0.00, 0.06, 0.17, 0.37, 0.41]
mode = 5  # argmax(distribution) + 1
expected_value = 1*0.00 + 2*0.06 + 3*0.17 + 4*0.37 + 5*0.41 = 4.12
```

---

#### 5. `src/ground_truth.py` - Evaluation

**What it does**: Compares SSR predictions to ground truth

```python
def create_ground_truth_dict(ground_truth_df):
    """
    Converts ground truth CSV to a lookup dictionary.

    Input: DataFrame with columns [respondent_id, question_id, ground_truth]

    Output: Nested dict
      {
          "R001": {
              "q1_would_subscribe": 1,
              "q2_subscription_likelihood": 4,
              ...
          },
          "R002": {...},
          ...
      }

    This allows fast lookup: ground_truth_dict[respondent_id][question_id]
    """

def evaluate_against_ground_truth(distributions, ground_truth_dict, question):
    """
    THE EVALUATION ENGINE

    Input:
      - distributions: 50 RatingDistribution objects (one per respondent)
      - ground_truth_dict: The 'correct answers'
      - question: Question object

    Process:
      For each distribution:
        1. Get ground truth: gt = ground_truth_dict[respondent_id][question_id]
        2. Get SSR prediction: pred = distribution.mode
        3. Check if correct: is_correct = (pred == gt)
        4. Compute error: error = abs(pred - gt)
        5. Get probability at truth: prob = distribution.distribution[gt - 1]

    Metrics computed:
      - mode_accuracy: % of exact matches
      - top2_accuracy: % where truth is in top 2 predictions
      - mae: Mean absolute error
      - rmse: Root mean squared error
      - prob_at_truth: Avg probability assigned to correct answer
      - kl_divergence: How different is predicted distribution from empirical?
      - confusion_matrix: Which ratings are confused with each other?

    Output:
      GroundTruthComparison(
          question_id="q2_subscription_likelihood",
          mode_accuracy=0.88,  # 88% exactly correct
          top2_accuracy=1.00,  # 100% in top 2
          mae=0.12,  # Off by 0.12 points on average
          rmse=0.34,
          prob_at_truth=0.58,  # 58% probability to correct answer
          kl_divergence=0.045,
          confusion_matrix=[[...]]
      )
    """
```

**Example Evaluation**:

```python
# Ground truth for R001, q2: 5 (Very likely)
# SSR prediction:
distribution = [0.00, 0.06, 0.17, 0.37, 0.41]
mode = 5

# Evaluation:
mode_correct = (5 == 5)  # ✓ Correct!
mae = abs(5 - 5) = 0  # Perfect
prob_at_truth = distribution[5-1] = distribution[4] = 0.41  # 41% confidence
```

---

#### 6. `src/report_generator.py` - Visual Reports

**What it does**: Creates PNG and TXT reports

```python
def create_one_page_report(human_comparisons, llm_comparisons, survey, output_path):
    """
    Creates a single-page visual report (PNG).

    Input:
      - human_comparisons: Dict of GroundTruthComparison for human responses
      - llm_comparisons: Dict of GroundTruthComparison for LLM responses

    Creates:
      - Accuracy bar chart (human vs LLM for each question)
      - Error metrics chart (MAE, RMSE comparison)
      - Confusion matrices (shows which ratings are confused)
      - Summary table

    Output: Saves to experiments/run_TIMESTAMP/report.png
    """

def generate_text_report(human_comparisons, llm_comparisons, survey, output_path):
    """
    Creates a text file with all metrics.

    Format:
      ================================================================================
      GROUND TRUTH COMPARISON REPORT: HUMAN VS LLM
      ================================================================================

      Overall Summary:
        Human Accuracy: 95.7%
        LLM Accuracy: 93.3%

      --------------------------------------------------------------------------------
      QUESTION: q1_would_subscribe
      --------------------------------------------------------------------------------
      Mode Accuracy:      Human: 100.0%  |  LLM: 92.0%
      Top-2 Accuracy:     Human: 100.0%  |  LLM: 100.0%
      MAE:                Human: 0.000  |  LLM: 0.080
      ...

    Output: Saves to experiments/run_TIMESTAMP/report.txt
    """
```

---

#### 7. `src/markdown_report.py` - Comprehensive Reports

**What it does**: Creates detailed markdown reports with explanations

```python
def generate_comprehensive_report(human_comparisons, llm_comparisons, survey, output_path):
    """
    Creates a comprehensive markdown report with:
      - Executive summary
      - Methodology explanation
      - Question-by-question analysis
      - Interpretation guides
      - Key findings
      - Recommendations

    This is the most detailed output - explains what everything means!

    Output: Saves to experiments/run_TIMESTAMP/report.md
    """
```

---

#### 8. `src/visualization.py` - Plotting Utilities

**What it does**: Helper functions for creating charts

```python
def plot_distribution(distribution: RatingDistribution, ax):
    """
    Plots a bar chart of a probability distribution.

    Input: RatingDistribution with probabilities [0.00, 0.06, 0.17, 0.37, 0.41]

    Creates:
      Bar chart with:
        X-axis: Rating labels (1-5)
        Y-axis: Probability (0-1)
        Colors: Highlight the mode
    """

def plot_question_analysis(question_analysis, ax):
    """
    Creates visualization for a single question's results.
    """
```

---

#### 9. `src/analysis.py` - Statistical Analysis

**What it does**: Aggregate statistics across responses

```python
def analyze_survey(distributions: List[RatingDistribution]):
    """
    Computes overall statistics.

    Input: All distributions from a survey

    Computes:
      - Mean expected value across all responses
      - Mean entropy (uncertainty)
      - Distribution of modes
      - Per-question statistics

    Output: SurveyAnalysis object
    """

def create_results_dataframe(distributions):
    """
    Converts distributions to a pandas DataFrame.

    Useful for:
      - Exporting to CSV
      - Further analysis
      - Integration with other tools
    """
```

---

#### 10. `src/comparison.py` - Human vs LLM Comparison

**What it does**: Utilities for comparing response styles

```python
def compare_distributions(human_dists, llm_dists):
    """
    Statistical comparison between human and LLM responses.

    Computes:
      - Mean difference in expected values
      - Variance differences
      - Entropy differences
    """
```

---

### Configuration Files

#### 11. `config/mixed_survey_config.yaml` - Survey Definition

**What it contains**: Survey questions and structure

```yaml
survey:
  name: "Online Lottery Gaming Platform Evaluation"
  description: "..."
  sample_size: 50

  context: |
    Background information about the lottery platform...

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

    # ... 4 more questions ...
```

**How it's used**:
```python
survey = Survey.from_config('config/mixed_survey_config.yaml')
# Loads this file and creates Survey object with 6 questions
```

---

### Notebooks

#### 12. `notebooks/example_notebook.ipynb` - Interactive Tutorial

**What it does**: Step-by-step walkthrough with explanations

**Sections**:
1. Load survey and initialize SSR
2. Rate a single response (detailed example)
3. Load ground truth data from experiments
4. Visualize ground truth distributions
5. View experiment reports
6. Compare human vs LLM accuracy
7. Generate new responses and rate them
8. Explore temperature parameter effects
9. Run a mini-experiment (10 respondents)
10. Key takeaways and references

**How to use**:
```bash
jupyter notebook notebooks/example_notebook.ipynb
```

---

## Complete Data Flow

### Visual Flow Diagram

```
┌─────────────────────────────────────────────────────────────────────────┐
│                         GROUND TRUTH PIPELINE                            │
└─────────────────────────────────────────────────────────────────────────┘

                        ┌──────────────────┐
                        │  User runs:      │
                        │  python          │
                        │  ground_truth    │
                        │  _pipeline.py    │
                        └────────┬─────────┘
                                 │
                                 ▼
                    ┌────────────────────────┐
                    │ Load Survey Config     │
                    │ (YAML → Survey object) │
                    └───────────┬────────────┘
                                │
                                ▼
                    ┌────────────────────────────┐
                    │ Generate 50 Profiles       │
                    │ (environmental attitudes)  │
                    └───────────┬────────────────┘
                                │
                                ▼
                    ┌─────────────────────────────────┐
                    │ Generate Ground Truth Ratings   │
                    │ (300 ratings: 50 people × 6 Qs)│
                    │ Save: ground_truth.csv          │
                    └──────────┬──────────────────────┘
                               │
                ┌──────────────┴──────────────┐
                │                             │
                ▼                             ▼
    ┌───────────────────────┐    ┌───────────────────────┐
    │ Generate Human Style  │    │ Generate LLM Style    │
    │ Responses (300)       │    │ Responses (300)       │
    │ "Definitely yes!"     │    │ "I would say that..." │
    └──────────┬────────────┘    └───────────┬───────────┘
               │                             │
               ▼                             ▼
    ┌─────────────────────────────────────────────────┐
    │           Apply SSR (OpenAI Embeddings)         │
    │                                                  │
    │  For each response:                              │
    │    1. Embed text → 1536-dim vector              │
    │    2. Embed scale labels → vectors              │
    │    3. Compute similarities → [0.2, 0.3, ...]    │
    │    4. Normalize → [0.05, 0.10, 0.35, 0.41]     │
    │                                                  │
    │  Output: 300 RatingDistribution objects         │
    └──────────┬──────────────────────────────────────┘
               │
               ▼
    ┌─────────────────────────────────────────────────┐
    │         Evaluate Against Ground Truth           │
    │                                                  │
    │  For each question (6 total):                   │
    │    - Mode accuracy: % exact matches             │
    │    - MAE: Average error                         │
    │    - Prob at truth: Confidence level            │
    │    - Confusion matrix: Error patterns           │
    │                                                  │
    │  Output: 6 GroundTruthComparison objects        │
    └──────────┬──────────────────────────────────────┘
               │
               ▼
    ┌─────────────────────────────────────────────────┐
    │              Generate Reports                    │
    │                                                  │
    │  1. report.png (visual charts)                  │
    │  2. report.txt (metrics table)                  │
    │  3. report.md (comprehensive explanations)      │
    │                                                  │
    │  Save to: experiments/run_TIMESTAMP/            │
    └─────────────────────────────────────────────────┘
```

---

## Step-by-Step Code Execution

Let's trace through a real execution with actual data values:

### Step 1: Load Survey

```python
survey = Survey.from_config('config/mixed_survey_config.yaml')
```

**Result**:
```python
Survey(
    name="Online Lottery Gaming Platform Evaluation",
    questions=[
        Question(id="q1_would_subscribe", type="yes_no", num_options=2),
        Question(id="q2_subscription_likelihood", type="likert_5", num_options=5),
        Question(id="q3_platform_trust", type="likert_7", num_options=7),
        Question(id="q4_price_preference", type="multiple_choice", num_options=5),
        Question(id="q5_recommend", type="yes_no", num_options=2),
        Question(id="q6_feature_importance", type="likert_5", num_options=5)
    ]
)
```

---

### Step 2: Generate Profiles

```python
profiles = generate_diverse_profiles(50)
```

**Result**:
```python
[
    RespondentProfile(environmental_consciousness="Extremely concerned"),
    RespondentProfile(environmental_consciousness="Very concerned"),
    RespondentProfile(environmental_consciousness="Moderately concerned"),
    RespondentProfile(environmental_consciousness="Slightly concerned"),
    RespondentProfile(environmental_consciousness="Not concerned"),
    # ... 45 more ...
]
```

---

### Step 3: Generate Ground Truth

```python
ground_truth_df = generate_ground_truth_ratings(survey, profiles, seed=100)
```

**Process for one respondent**:

```python
# Respondent R001 has profile: "Extremely concerned"
# This means tendency = "positive" (likely to rate highly)

for question in survey.questions:
    if question.id == "q2_subscription_likelihood":
        # 5-point scale
        # Positive tendency → skew towards higher ratings
        probs = [0.05, 0.10, 0.20, 0.30, 0.35]
        #        1     2     3     4     5

        # Sample from this distribution
        ground_truth = np.random.choice([1, 2, 3, 4, 5], p=probs)
        # Result: ground_truth = 5 (randomly selected based on probabilities)
```

**Result** (excerpt):
```
respondent_id,question_id,ground_truth
R001,q1_would_subscribe,1
R001,q2_subscription_likelihood,5
R001,q3_platform_trust,6
R001,q4_price_preference,4
R001,q5_recommend,1
R001,q6_feature_importance,4
R002,q1_would_subscribe,2
R002,q2_subscription_likelihood,4
...
```

---

### Step 4: Generate Text Responses

```python
human_responses = generate_responses_from_ground_truth(
    survey, profiles, ground_truth_df, response_style="human", seed=101
)
```

**Process for one response**:

```python
# R001, q2_subscription_likelihood, ground_truth=5

# Get the scale label for rating 5
ref_statements = {
    1: "Very unlikely",
    2: "Unlikely",
    3: "Neutral",
    4: "Likely",
    5: "Very likely"
}
target_statement = ref_statements[5]  # "Very likely"

# Generate human-style response
variations = [
    "Very likely",
    "I'd say very likely",
    "Definitely very likely",
    "My answer is very likely"
]
text_response = random.choice(variations)  # "I'd say very likely"
```

**Result**:
```python
Response(
    respondent_id="R001",
    question_id="q2_subscription_likelihood",
    text_response="I'd say very likely",
    respondent_profile={'environmental_consciousness': 'Extremely concerned'}
)
```

---

### Step 5: Apply SSR

```python
rater = SemanticSimilarityRater(
    model_name="text-embedding-3-small",
    temperature=1.0,
    normalize_method="paper",
    use_openai=True
)

distributions = rater.rate_responses(human_responses, survey, show_progress=True)
```

**Process for one response** (the magic happens here!):

```python
# Input
response_text = "I'd say very likely"
scale_labels = ["Very unlikely", "Unlikely", "Neutral", "Likely", "Very likely"]

# Step 1: Get embeddings from OpenAI
response_embedding = openai_client.embeddings.create(
    model="text-embedding-3-small",
    input=["I'd say very likely"]
).data[0].embedding
# Result: [0.023, -0.145, 0.389, ..., 0.021] (1536 numbers)

label_embeddings = openai_client.embeddings.create(
    model="text-embedding-3-small",
    input=scale_labels
).data
# Result: 5 vectors of 1536 dimensions each

# Step 2: Compute cosine similarities
similarities = []
for label_emb in label_embeddings:
    sim = cosine_similarity(response_embedding, label_emb)
    similarities.append(sim)
# Result: [0.35, 0.42, 0.58, 0.87, 0.94]
#          1     2     3     4     5
# → Response is most similar to "Very likely" (0.94)

# Step 3: Normalize to probabilities (paper's method)
shifted = similarities - min(similarities)
# [0.35, 0.42, 0.58, 0.87, 0.94] - 0.35 = [0.00, 0.07, 0.23, 0.52, 0.59]

scaled = shifted / temperature  # temperature=1.0
# [0.00, 0.07, 0.23, 0.52, 0.59] / 1.0 = [0.00, 0.07, 0.23, 0.52, 0.59]

probabilities = scaled / sum(scaled)
# [0.00, 0.07, 0.23, 0.52, 0.59] / 1.41 = [0.00, 0.05, 0.16, 0.37, 0.42]

# Step 4: Create distribution object
mode = argmax(probabilities) + 1 = 5
expected_value = sum(i * prob[i-1] for i in 1..5) = 1*0.00 + 2*0.05 + 3*0.16 + 4*0.37 + 5*0.42 = 4.16
```

**Result**:
```python
RatingDistribution(
    question_id="q2_subscription_likelihood",
    respondent_id="R001",
    distribution=[0.00, 0.05, 0.16, 0.37, 0.42],
    mode=5,  # SSR predicts rating 5
    expected_value=4.16,
    entropy=1.15,
    scale_labels={1: "Very unlikely", ..., 5: "Very likely"}
)
```

---

### Step 6: Evaluate Against Ground Truth

```python
ground_truth_dict = create_ground_truth_dict(ground_truth_df)
# {"R001": {"q2_subscription_likelihood": 5, ...}, ...}

comparison = evaluate_against_ground_truth(distributions, ground_truth_dict, question)
```

**Process for one distribution**:

```python
# Distribution for R001, q2_subscription_likelihood
distribution = RatingDistribution(mode=5, distribution=[0.00, 0.05, 0.16, 0.37, 0.42])

# Ground truth
ground_truth = ground_truth_dict["R001"]["q2_subscription_likelihood"]  # 5

# Evaluation
mode_correct = (distribution.mode == ground_truth)  # 5 == 5 → True
error = abs(distribution.mode - ground_truth)  # abs(5 - 5) = 0
prob_at_truth = distribution.distribution[ground_truth - 1]  # distribution[4] = 0.42

# Aggregate across all 50 respondents
mode_accuracy = count(correct) / 50  # e.g., 44/50 = 88%
mae = mean(errors)  # e.g., 0.12
prob_at_truth_avg = mean(probs_at_truth)  # e.g., 0.58
```

**Result**:
```python
GroundTruthComparison(
    question_id="q2_subscription_likelihood",
    mode_accuracy=0.88,  # 88% exactly correct
    top2_accuracy=1.00,  # 100% in top 2
    mae=0.12,
    rmse=0.34,
    prob_at_truth=0.58,
    kl_divergence=0.045,
    confusion_matrix=[[...]]
)
```

---

### Step 7: Generate Reports

```python
create_one_page_report(...)  # Creates PNG
generate_text_report(...)     # Creates TXT
generate_comprehensive_report(...)  # Creates MD
```

**Files created**:
```
experiments/run_20251022_164541/
├── ground_truth.csv          (300 rows: ground truth ratings)
├── report.png                (Visual comparison)
├── report.txt                (Metrics table)
└── report.md                 (Comprehensive explanations)
```

---

## How Files Work Together

### Dependency Graph

```
ground_truth_pipeline.py (ORCHESTRATOR)
    │
    ├─→ src/survey.py (loads config/mixed_survey_config.yaml)
    │
    ├─→ src/llm_client.py (generates profiles and Response objects)
    │
    ├─→ src/ssr_model.py (THE CORE: text → distributions)
    │       │
    │       └─→ OpenAI API (text-embedding-3-small)
    │
    ├─→ src/ground_truth.py (evaluation metrics)
    │
    ├─→ src/report_generator.py (PNG + TXT reports)
    │       │
    │       └─→ src/visualization.py (plotting helpers)
    │
    └─→ src/markdown_report.py (MD report)
```

### Data Structure Flow

```
YAML Config
    ↓
Survey Object
    ↓
Question Objects (6)
    ↓
RespondentProfile Objects (50)
    ↓
Ground Truth DataFrame (300 rows)
    ↓
Response Objects (300)
    ↓
RatingDistribution Objects (300)
    ↓
GroundTruthComparison Objects (6, one per question)
    ↓
Report Files (PNG, TXT, MD)
```

---

## Understanding the Data Structures

### Key Data Objects

#### 1. Question
```python
Question(
    id="q2_subscription_likelihood",
    text="How likely are you to subscribe?",
    type="likert_5",
    scale={1: "Very unlikely", 2: "Unlikely", 3: "Neutral", 4: "Likely", 5: "Very likely"},
    num_options=5
)
```

#### 2. RespondentProfile
```python
RespondentProfile(
    environmental_consciousness="Very concerned"
)
```

#### 3. Response
```python
Response(
    respondent_id="R001",
    question_id="q2_subscription_likelihood",
    text_response="I'd say very likely",
    respondent_profile={'environmental_consciousness': 'Very concerned'}
)
```

#### 4. RatingDistribution (SSR OUTPUT)
```python
RatingDistribution(
    question_id="q2_subscription_likelihood",
    respondent_id="R001",
    distribution=np.array([0.00, 0.05, 0.16, 0.37, 0.42]),  # Probabilities for ratings 1-5
    mode=5,                    # Most likely rating
    expected_value=4.16,       # Weighted average: sum(rating * prob)
    entropy=1.15,              # Uncertainty measure (higher = more uncertain)
    scale_labels={1: "Very unlikely", ..., 5: "Very likely"}
)
```

**Understanding the distribution**:
- `distribution[0]` = probability of rating 1 = 0.00 (0%)
- `distribution[1]` = probability of rating 2 = 0.05 (5%)
- `distribution[2]` = probability of rating 3 = 0.16 (16%)
- `distribution[3]` = probability of rating 4 = 0.37 (37%)
- `distribution[4]` = probability of rating 5 = 0.42 (42%)
- Sum = 1.00 (100%) ✓

#### 5. GroundTruthComparison (EVALUATION OUTPUT)
```python
GroundTruthComparison(
    question_id="q2_subscription_likelihood",
    mode_accuracy=0.88,        # 88% of predictions exactly match ground truth
    top2_accuracy=1.00,        # 100% of true answers are in top 2 predictions
    mae=0.12,                  # Average error is 0.12 rating points
    rmse=0.34,                 # Root mean squared error
    prob_at_truth=0.58,        # SSR assigns 58% probability to correct answer on average
    kl_divergence=0.045,       # How different is predicted vs empirical distribution
    confusion_matrix=np.array([...])  # Shows which ratings are confused
)
```

---

## Real Example Walkthrough

Let's trace a complete example from start to finish:

### Scenario
- **Respondent**: R007
- **Profile**: Environmental consciousness = "Not concerned"
- **Question**: q2_subscription_likelihood (5-point scale)

### Step-by-Step Execution

#### 1. Generate Ground Truth

```python
# R007 has profile "Not concerned"
# Tendency = "negative" (likely to rate lowly)

# For 5-point scale with negative tendency:
probs = [0.35, 0.30, 0.20, 0.10, 0.05]
#        1     2     3     4     5

# Sample
ground_truth = np.random.choice([1, 2, 3, 4, 5], p=probs)
# Result: ground_truth = 2 (randomly selected, more likely to be 1 or 2)

# Saved to ground_truth.csv:
# R007,q2_subscription_likelihood,2
```

#### 2. Generate Text Response

```python
# Ground truth is 2 → "Unlikely"
target_statement = "Unlikely"

# Human-style variations:
variations = ["Unlikely", "I'd say unlikely", "Definitely unlikely", "My answer is unlikely"]
text_response = random.choice(variations)
# Result: "I'd say unlikely"

# Create Response object
response = Response(
    respondent_id="R007",
    question_id="q2_subscription_likelihood",
    text_response="I'd say unlikely",
    respondent_profile={'environmental_consciousness': 'Not concerned'}
)
```

#### 3. Apply SSR

```python
# Input text: "I'd say unlikely"
# Scale labels: ["Very unlikely", "Unlikely", "Neutral", "Likely", "Very likely"]

# Embed text and labels via OpenAI
response_emb = [0.145, -0.234, 0.567, ..., 0.012]  # 1536 dims
label_embs = [
    [0.156, -0.245, 0.578, ..., 0.023],  # "Very unlikely"
    [0.142, -0.228, 0.563, ..., 0.009],  # "Unlikely"
    [0.023, 0.145, -0.234, ..., 0.145],  # "Neutral"
    [-0.142, 0.228, -0.563, ..., -0.009],  # "Likely"
    [-0.156, 0.245, -0.578, ..., -0.023]   # "Very likely"
]

# Compute cosine similarities
similarities = [0.78, 0.92, 0.45, 0.23, 0.18]
#                1     2     3     4     5
# → Response is most similar to "Unlikely" (0.92)

# Normalize (paper's method)
shifted = [0.78, 0.92, 0.45, 0.23, 0.18] - 0.18 = [0.60, 0.74, 0.27, 0.05, 0.00]
scaled = [0.60, 0.74, 0.27, 0.05, 0.00] / 1.0 = [0.60, 0.74, 0.27, 0.05, 0.00]
probabilities = [0.60, 0.74, 0.27, 0.05, 0.00] / 1.66 = [0.36, 0.45, 0.16, 0.03, 0.00]

# Create distribution
distribution = RatingDistribution(
    question_id="q2_subscription_likelihood",
    respondent_id="R007",
    distribution=[0.36, 0.45, 0.16, 0.03, 0.00],
    mode=2,  # "Unlikely"
    expected_value=1.82,
    entropy=1.08,
    scale_labels={1: "Very unlikely", ..., 5: "Very likely"}
)
```

#### 4. Evaluate

```python
# Ground truth: 2
# SSR prediction (mode): 2

# Evaluation
mode_correct = (2 == 2)  # True ✓
error = abs(2 - 2) = 0
prob_at_truth = distribution.distribution[2-1] = distribution[1] = 0.45

# This contributes to overall metrics:
# - Mode accuracy: +1 correct prediction
# - MAE: +0 error
# - Prob at truth: +0.45 probability
```

#### 5. Final Metrics for Question

After evaluating all 50 respondents:

```python
GroundTruthComparison(
    question_id="q2_subscription_likelihood",
    mode_accuracy=0.88,  # 44 out of 50 exactly correct
    top2_accuracy=1.00,  # All 50 had true answer in top 2
    mae=0.12,            # Average error across all 50
    rmse=0.34,
    prob_at_truth=0.58,  # SSR assigned 58% probability to correct answer on average
    kl_divergence=0.045,
    confusion_matrix=[
        [18, 2, 0, 0, 0],   # True rating 1: 18 predicted as 1, 2 as 2
        [1, 19, 1, 0, 0],   # True rating 2: 1 as 1, 19 as 2, 1 as 3
        [0, 1, 8, 1, 0],    # True rating 3: ...
        [0, 0, 2, 9, 1],    # True rating 4: ...
        [0, 0, 0, 1, 6]     # True rating 5: ...
    ]
)
```

---

## Summary

### The Complete Loop

1. **Configuration** (YAML) → defines survey structure
2. **Profiles** (Python) → creates diverse respondents
3. **Ground Truth** (CSV) → establishes correct answers
4. **Responses** (Python objects) → generates text aligned with ground truth
5. **SSR** (OpenAI API + Math) → converts text to probabilities
6. **Evaluation** (Statistics) → measures SSR accuracy
7. **Reports** (PNG/TXT/MD) → presents results

### Key Insights

- **SSR is non-deterministic**: Different text generates different distributions
- **Ground truth is probabilistic**: Profiles influence but don't determine ratings
- **Evaluation is comprehensive**: 7+ metrics capture different aspects of accuracy
- **Experiment organization**: Each run is isolated in timestamped folder

### Why This Design?

- **Modular**: Each component (survey, SSR, evaluation) can be used independently
- **Reproducible**: Seeds ensure same results across runs
- **Extensible**: Easy to add new question types, metrics, or response styles
- **Validated**: Ground truth comparison proves SSR works

---

**Next Steps**: Run the pipeline yourself and watch these data transformations happen in real-time!

```bash
python ground_truth_pipeline.py
```
