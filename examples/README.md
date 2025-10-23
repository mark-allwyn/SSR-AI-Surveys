# Example Files

## Ground Truth CSV Format

The `sample_ground_truth.csv` file shows the required format for uploading human ground truth data.

### Required Columns

- `respondent_id`: Unique identifier for each respondent (e.g., "R001", "R002", etc.)
- `question_id`: **MUST EXACTLY MATCH** question IDs in your survey YAML config
  - ⚠️ **CRITICAL**: If your config has `q1_would_subscribe`, your CSV must use `q1_would_subscribe` (exact match, case-sensitive)
  - System will reject CSV if question IDs don't match
- `ground_truth`: The human's rating/answer
  - For Yes/No questions: 1=No, 2=Yes
  - For Likert-5: 1-5 (1=lowest, 5=highest)
  - For Likert-7: 1-7 (1=lowest, 7=highest)
  - For multiple choice: 1, 2, 3, etc. (matches option order in config)

### Question ID Matching

**Example Survey Config:**
```yaml
questions:
  - id: "q1_would_subscribe"
  - id: "q2_subscription_likelihood"
  - id: "q3_platform_trust"
```

**Your CSV MUST use these exact IDs:**
```csv
respondent_id,question_id,ground_truth
R001,q1_would_subscribe,2
R001,q2_subscription_likelihood,4
R001,q3_platform_trust,5
```

**WRONG (will be rejected):**
```csv
respondent_id,question_id,ground_truth
R001,q1,2                           ❌ Wrong ID
R001,Q2_Subscription_Likelihood,4   ❌ Wrong case
R001,platform_trust,5               ❌ Missing prefix
```

### How It Works

1. **Define Personas in Survey Config**: Edit `config/mixed_survey_config.yaml` and add personas:
   ```yaml
   personas:
     - "A 35-year-old tech entrepreneur. High income, plays lottery occasionally."
     - "A 68-year-old retired teacher. Fixed income, never played online lottery."
   ```

2. **Collect Human Data**: Survey real humans with the same personas and record their responses

3. **Create Ground Truth CSV**: Format like `sample_ground_truth.csv`

4. **Upload in UI**:
   - Go to Run Experiment page
   - Section D: Ground Truth Data
   - Select "Upload Real Human Data"
   - Upload your CSV file

5. **Run Pipeline**:
   - System loads personas from survey config
   - System loads your ground truth data
   - System generates LLM responses using the same personas
   - System compares LLM responses against your human ground truth

### Example Workflow

```
Survey Config (mixed_survey_config.yaml):
  - Question q1: "Would you subscribe?"
  - Question q2: "How likely are you to subscribe?" (Likert-5)
  - Persona 1: "Tech entrepreneur, high income"
  - Persona 2: "Retired teacher, fixed income"

Human Data Collection:
  - Show survey to person matching Persona 1 → R001
    - q1: Yes (=2)
    - q2: Very likely (=5)
  - Show survey to person matching Persona 2 → R002
    - q1: No (=1)
    - q2: Unlikely (=2)

Ground Truth CSV:
  respondent_id,question_id,ground_truth
  R001,q1,2
  R001,q2,5
  R002,q1,1
  R002,q2,2

LLM Generation:
  - System generates responses for Persona 1 and Persona 2
  - Applies SSR to convert text to ratings
  - Compares against your human ground truth
```

### Tips

- Match respondent IDs to personas in your data collection
- Ensure all question IDs match your survey config exactly
- Include all questions for each respondent
- Ground truth ratings must be within valid range for each question type
