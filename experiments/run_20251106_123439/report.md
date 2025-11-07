# Ground Truth Comparison Report
## Human vs LLM Semantic Similarity Rating Performance

---

**Report Generated:** 2025-11-06 12:34:47

**Survey:** Product Evaluation - Single Category Example

**Description:** Standard product evaluation survey for a meal kit delivery service

**Total Questions:** 8

**Sample Size per Question:** 50

---

## Executive Summary

### Key Results

- **Human Response Accuracy:** 100.0%
- **LLM Response Accuracy:** 93.5%
- **Accuracy Gap:** 6.5% (Human superior)

### What This Means

Human-style responses show **moderately better** alignment with ground truth (6.5% advantage). The SSR method performs well on both response styles but has a slight preference for direct language.

---

## Methodology

### Semantic Similarity Rating (SSR)

The SSR methodology converts textual survey responses into probability distributions over Likert scale points using semantic similarity:

1. **Text Response Collection:** Gather open-ended textual responses to survey questions
2. **Semantic Encoding:** Convert both responses and scale labels (e.g., 'Very likely', 'Unlikely') into high-dimensional vector representations using sentence transformers
3. **Similarity Computation:** Calculate cosine similarity between the response vector and each scale label vector
4. **Probability Distribution:** Convert similarity scores to probabilities using softmax transformation
5. **Prediction:** The scale point with highest probability is the predicted rating (mode)

### Ground Truth Comparison

To evaluate SSR performance, we:

1. **Generate Ground Truth:** Create actual ratings for each respondent based on their profile
2. **Generate Responses:** Create textual responses that align with ground truth ratings
   - **Human-style:** Direct, opinionated (e.g., 'Definitely yes!')
   - **LLM-style:** Hedged, nuanced (e.g., 'While I appreciate... I might consider...')
3. **Apply SSR:** Convert text to probability distributions
4. **Compare:** Evaluate how well SSR predictions match ground truth

---

## Overall Performance Comparison

### Accuracy Metrics

| Metric | Human | LLM | Better | Interpretation |
|--------|-------|-----|--------|----------------|
| **Mode Accuracy** | 100.0% | 93.5% | Human | % of predictions exactly matching ground truth |
| **Top-2 Accuracy** | 100.0% | 99.5% | Human | % where true answer is in top 2 predictions |
| **Mean Absolute Error** | 0.000 | 0.714 | Human | Average distance from true rating (lower is better) |
| **Prob at Truth** | 1.000 | 0.509 | Human | Avg probability assigned to true answer (higher = more confident) |

### What These Metrics Mean

**Mode Accuracy** measures how often SSR's top prediction exactly matches the ground truth. This is the strictest metric—you either get it right or wrong.

- **Result:** Human responses achieve 100.0% vs LLM's 93.5%
- **Implication:** Direct language is 6.5% more likely to be correctly classified

**Top-2 Accuracy** is more forgiving—it counts predictions as correct if the true answer is in the top 2 most likely ratings. This reflects SSR's probabilistic nature.

- **Result:** Human 100.0% vs LLM 99.5%
- **Implication:** Both methods rarely miss completely—SSR captures the right range

**Mean Absolute Error (MAE)** measures average distance from the true rating. For a 5-point scale, an MAE of 0.5 means predictions are off by half a point on average.

- **Result:** Human MAE = 0.000, LLM MAE = 0.714
- **Implication:** Human responses are highly accurate (within ~0.3 points)

**Probability at Truth** shows how confident SSR is when it assigns probability to the correct answer. Higher values mean more confident, accurate predictions.

- **Result:** Human 1.000 vs LLM 0.509
---

## Question-by-Question Analysis

### Purchase Intent

**Question:** How likely would you be to subscribe to this meal kit service?

**Type:** Likert 5

**Scale:** 5-point (Definitely would not subscribe to Definitely would subscribe)

#### Performance

| Metric | Human | LLM | Difference |
|--------|-------|-----|------------|
| Mode Accuracy | 100.0% | 100.0% | +0.0% |
| MAE | 0.000 | 0.563 | -0.563 |
| Prob at Truth | 1.000 | 0.465 | +0.535 |

#### Interpretation

**5-point Likert scales** are more challenging because SSR must differentiate between 5 similar options. Human responses still achieve strong accuracy (100.0%), indicating clear semantic distinctions.

#### Prediction Patterns

- **Human correct predictions:** 50 / 50 responses
- **LLM correct predictions:** 50 / 50 responses


---

### Value For Money

**Question:** The pricing for this service represents good value for money

**Type:** Likert 5

**Scale:** 5-point (Strongly disagree to Strongly agree)

#### Performance

| Metric | Human | LLM | Difference |
|--------|-------|-----|------------|
| Mode Accuracy | 100.0% | 96.0% | +4.0% |
| MAE | 0.000 | 0.780 | -0.780 |
| Prob at Truth | 1.000 | 0.480 | +0.520 |

#### Interpretation

**5-point Likert scales** are more challenging because SSR must differentiate between 5 similar options. Human responses still achieve strong accuracy (100.0%), indicating clear semantic distinctions.

#### Prediction Patterns

- **Human correct predictions:** 50 / 50 responses
- **LLM correct predictions:** 48 / 50 responses


---

### Likeability

**Question:** Overall, how much do you like or dislike this meal kit concept?

**Type:** Likert 7

**Scale:** 7-point (Dislike it a great deal to Like it a great deal)

#### Performance

| Metric | Human | LLM | Difference |
|--------|-------|-----|------------|
| Mode Accuracy | 100.0% | 92.0% | +8.0% |
| MAE | 0.000 | 1.048 | -1.048 |
| Prob at Truth | 1.000 | 0.367 | +0.633 |

#### Interpretation

**7-point Likert scales** are more challenging because SSR must differentiate between 7 similar options. Human responses still achieve strong accuracy (100.0%), indicating clear semantic distinctions.

#### Prediction Patterns

- **Human correct predictions:** 49 / 49 responses
- **LLM correct predictions:** 46 / 50 responses


---

### Uniqueness

**Question:** This meal kit service is unique and different from other meal delivery options

**Type:** Likert 5

**Scale:** 5-point (Strongly disagree to Strongly agree)

#### Performance

| Metric | Human | LLM | Difference |
|--------|-------|-----|------------|
| Mode Accuracy | 100.0% | 96.0% | +4.0% |
| MAE | 0.000 | 0.938 | -0.938 |
| Prob at Truth | 1.000 | 0.487 | +0.513 |

#### Interpretation

**5-point Likert scales** are more challenging because SSR must differentiate between 5 similar options. Human responses still achieve strong accuracy (100.0%), indicating clear semantic distinctions.

#### Prediction Patterns

- **Human correct predictions:** 50 / 50 responses
- **LLM correct predictions:** 48 / 50 responses


---

### Relevance

**Question:** This meal kit service is relevant to people like me

**Type:** Likert 5

**Scale:** 5-point (Strongly disagree to Strongly agree)

#### Performance

| Metric | Human | LLM | Difference |
|--------|-------|-----|------------|
| Mode Accuracy | 100.0% | 92.0% | +8.0% |
| MAE | 0.000 | 0.753 | -0.753 |
| Prob at Truth | 1.000 | 0.504 | +0.496 |

#### Interpretation

**5-point Likert scales** are more challenging because SSR must differentiate between 5 similar options. Human responses still achieve strong accuracy (100.0%), indicating clear semantic distinctions.

#### Prediction Patterns

- **Human correct predictions:** 50 / 50 responses
- **LLM correct predictions:** 46 / 50 responses


---

### Recommendation

**Question:** How likely would you be to recommend this meal kit service to a friend?

**Type:** Likert 5

**Scale:** 5-point (Definitely would not recommend to Definitely would recommend)

#### Performance

| Metric | Human | LLM | Difference |
|--------|-------|-----|------------|
| Mode Accuracy | 100.0% | 92.0% | +8.0% |
| MAE | 0.000 | 0.655 | -0.655 |
| Prob at Truth | 1.000 | 0.437 | +0.563 |

#### Interpretation

**5-point Likert scales** are more challenging because SSR must differentiate between 5 similar options. Human responses still achieve strong accuracy (100.0%), indicating clear semantic distinctions.

#### Prediction Patterns

- **Human correct predictions:** 50 / 50 responses
- **LLM correct predictions:** 46 / 50 responses


---

### Current Meal Kit User

**Question:** Do you currently use any meal kit delivery service?

**Type:** Yes No

**Scale:** Binary (No / Yes)

#### Performance

| Metric | Human | LLM | Difference |
|--------|-------|-----|------------|
| Mode Accuracy | 100.0% | 80.0% | +20.0% |
| MAE | 0.000 | 0.200 | -0.200 |
| Prob at Truth | 1.000 | 0.800 | +0.200 |

#### Interpretation

**Binary questions** are typically easier to classify because there are only two options. Human responses achieve near-perfect accuracy (100.0%), suggesting direct yes/no statements align perfectly with semantic similarity.

LLM responses show 20.0% lower accuracy, likely due to hedging language that makes binary classification ambiguous.

#### Prediction Patterns

- **Human correct predictions:** 50 / 50 responses
- **LLM correct predictions:** 40 / 50 responses


---

### Most Important Feature

**Question:** Which feature of this meal kit service is most important to you?

**Type:** Multiple Choice

**Scale:** 6 options

#### Performance

| Metric | Human | LLM | Difference |
|--------|-------|-----|------------|
| Mode Accuracy | 100.0% | 100.0% | +0.0% |
| MAE | 0.000 | 0.777 | -0.777 |
| Prob at Truth | 1.000 | 0.532 | +0.468 |

#### Interpretation

**Multiple choice questions** depend heavily on how distinct the options are semantically. Both response styles achieve excellent accuracy (100.0%), suggesting the options are semantically well-separated.

#### Prediction Patterns

- **Human correct predictions:** 48 / 48 responses
- **LLM correct predictions:** 50 / 50 responses


---

## Key Findings

### 1. Response Style Significantly Impacts SSR Accuracy

Direct, opinionated responses (human-style) outperform hedged, nuanced responses (LLM-style) by **6.5%** on average. This suggests:

- SSR relies on clear semantic alignment between response and scale labels
- Hedging language ("I might", "perhaps", "considering") dilutes semantic signal
- Confident statements map more precisely to scale extremes

### 2. Question Type Matters

**Yes/No questions** achieve the highest accuracy (100.0% for human responses). Binary choices provide clear semantic boundaries that SSR handles well.

**Likert scales** show more variation (100.0% accuracy). The challenge increases with more scale points due to semantic overlap between adjacent options.

### 3. Prediction Errors are Typically Adjacent

When SSR makes mistakes, they're usually off by just one scale point. This indicates:

- The semantic embedding space preserves ordinal relationships
- SSR captures the general sentiment even when missing the exact rating
- Top-2 accuracy is significantly higher than mode accuracy

### 4. SSR Maintains Appropriate Uncertainty

The average probability assigned to the true answer is 100.0%, meaning SSR typically distributes probability across 3-4 plausible ratings rather than being overly confident. This probabilistic approach:

- Captures the inherent ambiguity in textual responses
- Provides richer information than forced single-choice classification
- Enables downstream uncertainty-aware analysis

---

## Metric Interpretation Guide

### Accuracy Metrics

**Mode Accuracy**
- **90-100%:** Excellent - SSR reliably predicts exact ratings
- **70-90%:** Good - Most predictions are correct, some adjacent errors
- **50-70%:** Fair - Captures general sentiment but misses specifics
- **<50%:** Poor - Below random chance for many scales

**Top-2 Accuracy**
- **95-100%:** Excellent - SSR rarely completely misses the range
- **80-95%:** Good - Occasional multi-point errors
- **<80%:** Concerning - SSR may be missing key semantic patterns

### Error Metrics

**Mean Absolute Error (MAE)**
- **<0.3:** Excellent - Predictions very close to truth
- **0.3-0.5:** Good - Within half a point on average
- **0.5-1.0:** Fair - Off by about one scale point
- **>1.0:** Poor - Missing by multiple points

### Probabilistic Metrics

**Probability at Truth**
- **>0.5:** Very confident predictions (concentrated on true answer)
- **0.3-0.5:** Moderate confidence (true answer is top choice but not dominant)
- **0.2-0.3:** Distributed predictions (spreading probability across options)
- **<0.2:** High uncertainty (uniform-like distribution)

**KL Divergence**
- Measures how different the predicted distribution is from empirical reality
- **<0.05:** Very close match to observed distribution
- **0.05-0.10:** Reasonable alignment
- **>0.10:** Systematic bias in predictions

---

## Recommendations

### For Practitioners Using SSR

2. **Choose Appropriate Question Types**
   - Use binary (yes/no) questions when precision is critical
   - Limit Likert scales to 5 points to maximize discrimination
   - Ensure multiple-choice options are semantically distinct

3. **Leverage Probabilistic Outputs**
   - Don't just use the mode—consider the full distribution
   - High entropy predictions indicate genuine ambiguity
   - Use expected values for continuous analysis

4. **Validate on Your Domain**
   - Collect ground truth for a subset of responses
   - Measure accuracy specific to your question wording
   - Adjust temperature parameter if needed (higher = more uncertainty)

### For Survey Design

1. **Write Clear Scale Labels**
   - Use semantically distinct terms (e.g., 'Never' vs 'Rarely' vs 'Often')
   - Avoid similar-sounding options (e.g., 'Good' vs 'Fine')
   - Test that labels span the semantic space evenly

2. **Provide Context**
   - Frame questions clearly so responses stay on-topic
   - Include examples of extreme positions
   - Clarify what each scale point represents

3. **Test with Both Response Styles**
   - Pilot with both direct and hedged responses
   - If LLM performance is poor, scale labels may be ambiguous
   - Refine questions where human-LLM gap is largest

### When to Use SSR

**Best Use Cases:**
- Converting qualitative feedback to quantitative metrics
- Analyzing open-ended survey responses at scale
- Situations where you need probabilistic ratings
- Comparing human and AI-generated responses

**Limitations to Consider:**
- Requires semantically clear scale labels
- May struggle with highly technical or domain-specific language
- Cultural and linguistic variations can affect embeddings
- Not suitable for non-ordinal categories

---

## Appendix

### Technical Details

- **Embedding Model:** sentence-transformers (all-MiniLM-L6-v2)
- **Similarity Metric:** Cosine similarity
- **Normalization:** Softmax with temperature = 1.0
- **Evaluation:** Mode accuracy, MAE, RMSE, probabilistic metrics

### References

- Paper: *LLMs Reproduce Human Purchase Intent via Semantic Similarity Elicitation* (arXiv:2510.08338v2)
- Repository: https://github.com/pymc-labs/semantic-similarity-rating
- Documentation: See README.md for full pipeline details

---

*Report generated by SSR Pipeline*
