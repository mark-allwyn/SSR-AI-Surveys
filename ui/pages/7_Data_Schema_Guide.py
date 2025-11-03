"""Data Schema & Architecture Guide - Technical Documentation for Data Professionals."""

import streamlit as st
from pathlib import Path
import sys
import pandas as pd
import plotly.graph_objects as go
from plotly.subplots import make_subplots

# Add parent directory to path
sys.path.insert(0, str(Path(__file__).parent.parent.parent))

from ui.components.metrics_cards import success_message, warning_message, error_message

# Brand colors
brand_colors = {
    'teal_blue': '#367588',
    'turquoise': '#40E0D0',
    'atomic_orange': '#FF6E3A',
    'cornflower_blue': '#6495ED',
    'electric_lime': '#CCFF00',
    'teal_dark': '#2C5F6F',
    'light_grey': '#D3D3D3'
}

st.set_page_config(page_title="Data Schema Guide", page_icon="", layout="wide")

# Custom CSS for brand colors
st.markdown(f"""
<style>
    /* Primary buttons */
    .stButton > button[kind="primary"] {{
        background-color: {brand_colors['teal_blue']} !important;
        border-color: {brand_colors['teal_blue']} !important;
        color: white !important;
    }}
    .stButton > button[kind="primary"]:hover {{
        background-color: {brand_colors['teal_dark']} !important;
        border-color: {brand_colors['teal_dark']} !important;
    }}
    /* Code blocks */
    .stCodeBlock {{
        background-color: #f5f5f5 !important;
    }}
    /* Tables */
    .dataframe {{
        font-size: 12px !important;
    }}
</style>
""", unsafe_allow_html=True)

st.title(" Data Schema & Architecture Guide")

st.markdown("""
**Technical documentation for data professionals implementing SSR in their workflows.**

This guide explains the data formats, schemas, and architecture of the SSR pipeline,
including how data flows through the system and what formats are required at each stage.
""")

# ======================
# Navigation
# ======================
st.markdown("---")
section = st.radio(
    "**Navigate to Section:**",
    options=[
        " System Architecture",
        " Ground Truth Data Schema",
        " Survey Configuration Schema",
        " Output Data Formats",
        " Data Flow Diagram",
        " Example Datasets"
    ],
    horizontal=True
)

st.markdown("---")

# ======================
# Section 1: System Architecture
# ======================
if section == " System Architecture":
    st.header(" System Architecture Overview")

    st.markdown("""
    The SSR Pipeline consists of three main components that work together to convert
    natural language survey responses into probability distributions over Likert scales.
    """)

    # Architecture diagram using Plotly
    fig = go.Figure()

    # Define boxes for each component
    components = [
        {"name": "Ground Truth Data", "x": 0.15, "y": 0.8, "color": brand_colors['cornflower_blue']},
        {"name": "Survey Config", "x": 0.15, "y": 0.5, "color": brand_colors['cornflower_blue']},
        {"name": "SSR Model", "x": 0.5, "y": 0.65, "color": brand_colors['teal_blue']},
        {"name": "Evaluation Metrics", "x": 0.85, "y": 0.8, "color": brand_colors['atomic_orange']},
        {"name": "Output Reports", "x": 0.85, "y": 0.5, "color": brand_colors['atomic_orange']},
    ]

    # Add boxes
    for comp in components:
        fig.add_shape(
            type="rect",
            x0=comp["x"]-0.12, y0=comp["y"]-0.08,
            x1=comp["x"]+0.12, y1=comp["y"]+0.08,
            fillcolor=comp["color"],
            line=dict(color="white", width=2),
        )
        fig.add_annotation(
            x=comp["x"], y=comp["y"],
            text=f"<b>{comp['name']}</b>",
            showarrow=False,
            font=dict(size=14, color="white"),
        )

    # Add arrows
    arrows = [
        {"x0": 0.27, "y0": 0.8, "x1": 0.38, "y1": 0.73},  # Ground Truth -> SSR
        {"x0": 0.27, "y0": 0.5, "x1": 0.38, "y1": 0.57},  # Config -> SSR
        {"x0": 0.62, "y0": 0.73, "x1": 0.73, "y1": 0.8},  # SSR -> Metrics
        {"x0": 0.62, "y0": 0.57, "x1": 0.73, "y1": 0.5},  # SSR -> Reports
    ]

    for arrow in arrows:
        fig.add_annotation(
            x=arrow["x1"], y=arrow["y1"],
            ax=arrow["x0"], ay=arrow["y0"],
            xref='x', yref='y',
            axref='x', ayref='y',
            showarrow=True,
            arrowhead=2,
            arrowsize=1.5,
            arrowwidth=2,
            arrowcolor=brand_colors['teal_dark']
        )

    fig.update_layout(
        showlegend=False,
        xaxis=dict(showgrid=False, showticklabels=False, zeroline=False, range=[0, 1]),
        yaxis=dict(showgrid=False, showticklabels=False, zeroline=False, range=[0.3, 1]),
        height=400,
        plot_bgcolor='white',
        margin=dict(l=20, r=20, t=20, b=20)
    )

    st.plotly_chart(fig, use_container_width=True)

    # Component descriptions
    col1, col2 = st.columns(2)

    with col1:
        st.markdown("""
        ### Input Components

        **1. Ground Truth Data (CSV)**
        - Contains human Likert scale ratings (numeric only)
        - Required columns: `respondent_id`, `question_id`, `ground_truth`

        **2. Survey Configuration (YAML)**
        - Defines questions, scale types, and labels
        - Specifies personas for LLM generation
        - Controls experiment parameters
        """)

    with col2:
        st.markdown("""
        ### Processing & Output

        **3. SSR Model**
        - Converts text responses to probability distributions
        - Uses OpenAI embeddings + semantic similarity
        - Implements paper methodology (arXiv:2510.08338v2)

        **4. Evaluation & Reports**
        - Calculates accuracy, MAE, RMSE, KL divergence
        - Generates comparison reports (Ground Truth vs SSR)
        - Outputs JSON distributions and CSV results
        """)

    st.info("""
    **Key Insight:** The system is designed to be **modular**. You can:
    - Use existing ground truth data (no LLM generation needed)
    - Generate synthetic responses via LLMs
    - Mix human and LLM responses for validation
    """)

# ======================
# Section 2: Ground Truth Schema
# ======================
elif section == " Ground Truth Data Schema":
    st.header(" Ground Truth Data Schema")

    st.markdown("""
    The ground truth CSV contains **human Likert scale ratings** used for validation.

    **Purpose:** Compare LLM-generated responses (via SSR) against real human ratings to validate accuracy.

    **You can either:**
    - Upload your own ground truth data (real human survey results)
    - Let the system generate synthetic ground truth for testing
    """)

    # Required schema
    st.subheader("Required Columns")

    schema_df = pd.DataFrame([
        {
            "Column": "respondent_id",
            "Type": "string/int",
            "Description": "Unique identifier for each respondent",
            "Example": "R001, R002, R003",
            "Required": " Yes"
        },
        {
            "Column": "question_id",
            "Type": "string",
            "Description": "Identifier matching survey config questions",
            "Example": "recommend, quality, satisfaction",
            "Required": " Yes"
        },
        {
            "Column": "ground_truth",
            "Type": "int",
            "Description": "Human-selected rating on Likert scale",
            "Example": "1, 2, 3, 4, 5",
            "Required": " Yes"
        },
    ])

    st.dataframe(schema_df, use_container_width=True, hide_index=True)

    # Optional columns
    st.subheader("Optional Columns")

    optional_df = pd.DataFrame([
        {
            "Column": "timestamp",
            "Type": "datetime",
            "Description": "When response was collected",
            "Example": "2024-01-15 14:30:00",
            "Use Case": "Time-series analysis"
        },
        {
            "Column": "demographics_*",
            "Type": "various",
            "Description": "Any demographic columns (age, gender, etc.)",
            "Example": "age, gender, income_bracket",
            "Use Case": "Segmentation and subgroup analysis"
        },
    ])

    st.dataframe(optional_df, use_container_width=True, hide_index=True)

    # Example data
    st.markdown("---")
    st.subheader(" Example Ground Truth Data")

    example_gt = pd.DataFrame([
        {"respondent_id": "R001", "question_id": "recommend", "ground_truth": 5},
        {"respondent_id": "R001", "question_id": "quality", "ground_truth": 5},
        {"respondent_id": "R002", "question_id": "recommend", "ground_truth": 3},
        {"respondent_id": "R002", "question_id": "quality", "ground_truth": 3},
        {"respondent_id": "R003", "question_id": "recommend", "ground_truth": 2},
        {"respondent_id": "R003", "question_id": "quality", "ground_truth": 2},
    ])

    st.dataframe(example_gt, use_container_width=True, hide_index=True)

    # Data quality requirements
    st.markdown("---")
    st.subheader(" Data Quality Requirements")

    col1, col2 = st.columns(2)

    with col1:
        st.markdown("""
        **Required:**
        -  No missing values in required columns
        -  `ground_truth` values must match scale definition in config
        -  `question_id` must exist in survey config
        -  `respondent_id` should be consistent for each person
        """)

    with col2:
        st.markdown("""
        **Recommended:**
        -  At least 50+ responses per question for robust analysis
        -  Multiple ratings per scale point (avoid skewed data)
        -  Consistent respondent IDs across all questions
        -  Representative sample of your target population
        """)

    st.warning("""
    **Common Issues:**
    -  **Rating out of range**: e.g., ground_truth=6 on a 5-point scale → Will cause errors
    -  **Mismatched question IDs**: e.g., `q1` in data but `recommend` in config → Will be skipped
    -  **Missing values**: Null or empty cells in required columns → Will cause errors
    """)

# ======================
# Section 3: Survey Config Schema
# ======================
elif section == " Survey Configuration Schema":
    st.header(" Survey Configuration Schema (YAML)")

    st.markdown("""
    The survey configuration file (YAML format) defines the structure of your survey,
    including questions, scales, and optional personas for LLM generation.
    """)

    # Full example
    st.subheader(" Complete Example Configuration")

    example_yaml = """# Survey Configuration Example
name: "Product Feedback Survey"
description: "Evaluating customer sentiment about our new product"

# Optional: Context/Stimulus shown to respondents
context: |
  Please consider our new SmartWidget Pro when answering these questions.
  The SmartWidget Pro features:
  - Advanced AI integration
  - 10-hour battery life
  - Price: $299

# Question definitions
questions:
  - id: recommend
    text: "How likely are you to recommend this product to others?"
    type: likert_5
    scale:
      1: "Very unlikely"
      2: "Unlikely"
      3: "Neutral"
      4: "Likely"
      5: "Very likely"

  - id: quality
    text: "How would you rate the quality of this product?"
    type: likert_5
    scale:
      1: "Very poor"
      2: "Poor"
      3: "Fair"
      4: "Good"
      5: "Excellent"

  - id: value
    text: "Do you think this product offers good value for money?"
    type: yes_no
    scale:
      1: "No"
      2: "Yes"

# Optional: Personas for LLM generation
personas:
  mode: descriptions  # or 'attributes'
  descriptions:
    - "A 35-year-old tech enthusiast who values innovation and quality."
    - "A 50-year-old budget-conscious shopper focused on value."
    - "A 28-year-old early adopter excited about new products."
    - "A 65-year-old skeptical customer who needs convincing."

# Optional: Experiment settings
settings:
  llm_provider: openai
  llm_model: gpt-4
  temperature: 0.7
  n_respondents: 100
"""

    st.code(example_yaml, language="yaml")

    # Schema breakdown
    st.markdown("---")
    st.subheader(" Schema Breakdown")

    # Top-level fields
    st.markdown("### Top-Level Fields")

    top_level_df = pd.DataFrame([
        {
            "Field": "name",
            "Type": "string",
            "Required": "Optional",
            "Description": "Survey name for identification"
        },
        {
            "Field": "description",
            "Type": "string",
            "Required": "Optional",
            "Description": "Survey description/purpose"
        },
        {
            "Field": "context",
            "Type": "string (multiline)",
            "Required": "Optional",
            "Description": "Context/stimulus information shown to respondents before questions"
        },
        {
            "Field": "questions",
            "Type": "list",
            "Required": " Required",
            "Description": "List of question objects (see below)"
        },
        {
            "Field": "personas",
            "Type": "object",
            "Required": "Optional",
            "Description": "Persona definitions for LLM generation"
        },
        {
            "Field": "settings",
            "Type": "object",
            "Required": "Optional",
            "Description": "Experiment settings and parameters"
        },
    ])

    st.dataframe(top_level_df, use_container_width=True, hide_index=True)

    # Question object
    st.markdown("---")
    st.markdown("### Question Object Schema")

    question_df = pd.DataFrame([
        {
            "Field": "id",
            "Type": "string",
            "Required": " Required",
            "Description": "Unique identifier (must match ground truth data)",
            "Example": "recommend, q1, likeability"
        },
        {
            "Field": "text",
            "Type": "string",
            "Required": " Required",
            "Description": "The actual question text",
            "Example": "How likely are you to recommend?"
        },
        {
            "Field": "type",
            "Type": "string",
            "Required": " Required",
            "Description": "Scale type: likert_5, likert_7, likert_6, yes_no, custom",
            "Example": "likert_5"
        },
        {
            "Field": "scale",
            "Type": "dict",
            "Required": " Required",
            "Description": "Mapping of numeric ratings to text labels",
            "Example": "{1: 'Disagree', 2: 'Neutral', 3: 'Agree'}"
        },
    ])

    st.dataframe(question_df, use_container_width=True, hide_index=True)

    # Supported scale types
    st.markdown("---")
    st.subheader(" Supported Scale Types")

    scale_cols = st.columns(3)

    with scale_cols[0]:
        st.markdown("""
        **likert_5** (5-point scale)
        ```yaml
        scale:
          1: "Strongly disagree"
          2: "Disagree"
          3: "Neutral"
          4: "Agree"
          5: "Strongly agree"
        ```
        """)

    with scale_cols[1]:
        st.markdown("""
        **likert_7** (7-point scale)
        ```yaml
        scale:
          1: "Strongly disagree"
          2: "Disagree"
          3: "Somewhat disagree"
          4: "Neutral"
          5: "Somewhat agree"
          6: "Agree"
          7: "Strongly agree"
        ```
        """)

    with scale_cols[2]:
        st.markdown("""
        **yes_no** (Binary)
        ```yaml
        scale:
          1: "No"
          2: "Yes"
        ```

        **custom** (Any range)
        ```yaml
        scale:
          1: "Very poor"
          2: "Poor"
          3: "Fair"
          4: "Good"
          5: "Very good"
          6: "Excellent"
        ```
        """)

    # Personas section
    st.markdown("---")
    st.subheader(" Personas Configuration (Optional)")

    st.markdown("""
    Used for LLM-based response generation. Define diverse respondent profiles:
    """)

    personas_yaml = """personas:
  mode: descriptions
  descriptions:
    - "A 35-year-old tech entrepreneur in San Francisco. High income, environmentally conscious."
    - "A 68-year-old retired teacher in rural Iowa. Fixed income, cautious about change."
    - "A 28-year-old graduate student. Passionate about sustainability and social issues."
    - "A 45-year-old small business owner. Pragmatic, family-oriented, moderate income."
"""

    st.code(personas_yaml, language="yaml")

    st.info("""
    **Note:** Personas are only needed if you're using the LLM generation feature to create
    synthetic survey responses. If you already have ground truth data, you can omit this section.
    """)

# ======================
# Section 4: Output Formats
# ======================
elif section == " Output Data Formats":
    st.header(" Output Data Formats")

    st.markdown("""
    The SSR pipeline generates multiple output files in the experiment folder (`experiments/run_TIMESTAMP/`).
    Each file serves a specific purpose for analysis and validation.
    """)

    # Output files table
    st.subheader(" Output Files Overview")

    output_files_df = pd.DataFrame([
        {
            "File": "ground_truth.csv",
            "Format": "CSV",
            "Description": "Copy of input ground truth data",
            "Use Case": "Reference and reproducibility"
        },
        {
            "File": "llm_distributions.json",
            "Format": "JSON",
            "Description": "SSR probability distributions for each response",
            "Use Case": "Detailed analysis, custom metrics"
        },
        {
            "File": "report.txt",
            "Format": "Text",
            "Description": "Human-readable comparison report",
            "Use Case": "Quick review, sharing with stakeholders"
        },
        {
            "File": "report.md",
            "Format": "Markdown",
            "Description": "Formatted report with tables and metrics",
            "Use Case": "Documentation, presentations"
        },
        {
            "File": "results_summary.json",
            "Format": "JSON",
            "Description": "Aggregated metrics and statistics",
            "Use Case": "Automated analysis, dashboards"
        },
    ])

    st.dataframe(output_files_df, use_container_width=True, hide_index=True)

    # LLM Distributions JSON
    st.markdown("---")
    st.subheader(" llm_distributions.json Schema")

    st.markdown("""
    This file contains the probability distributions generated by SSR for each response.
    It's the core output for downstream analysis.
    """)

    json_example = """{
  "respondent_id": {
    "question_id": {
      "distribution": [0.05, 0.15, 0.25, 0.35, 0.20],
      "mode": 4,
      "expected_value": 3.45,
      "entropy": 1.52,
      "text_response": "I think this product is pretty good overall.",
      "ground_truth_rating": 4,
      "similarities": [0.72, 0.78, 0.85, 0.91, 0.82]
    }
  }
}

// Example with real data:
{
  "R001": {
    "recommend": {
      "distribution": [0.02, 0.08, 0.15, 0.35, 0.40],
      "mode": 5,
      "expected_value": 4.03,
      "entropy": 1.31,
      "text_response": "Absolutely love it! Would definitely recommend.",
      "ground_truth_rating": 5,
      "similarities": [0.65, 0.71, 0.78, 0.87, 0.93]
    },
    "quality": {
      "distribution": [0.05, 0.12, 0.28, 0.38, 0.17],
      "mode": 4,
      "expected_value": 3.50,
      "entropy": 1.45,
      "text_response": "Quality is good, no major issues.",
      "ground_truth_rating": 4,
      "similarities": [0.68, 0.74, 0.82, 0.88, 0.79]
    }
  }
}
"""

    st.code(json_example, language="json")

    # Field descriptions
    st.markdown("### Field Descriptions")

    json_fields_df = pd.DataFrame([
        {
            "Field": "distribution",
            "Type": "array[float]",
            "Description": "Probability distribution over scale points (sums to 1.0)",
            "Example": "[0.05, 0.15, 0.25, 0.35, 0.20]"
        },
        {
            "Field": "mode",
            "Type": "int",
            "Description": "Most likely rating (argmax of distribution)",
            "Example": "4"
        },
        {
            "Field": "expected_value",
            "Type": "float",
            "Description": "Weighted average of distribution (more robust than mode)",
            "Example": "3.45"
        },
        {
            "Field": "entropy",
            "Type": "float",
            "Description": "Shannon entropy (uncertainty measure, lower = more certain)",
            "Example": "1.52"
        },
        {
            "Field": "text_response",
            "Type": "string",
            "Description": "Original text response",
            "Example": "I think this is pretty good"
        },
        {
            "Field": "ground_truth_rating",
            "Type": "int",
            "Description": "Actual rating from ground truth data",
            "Example": "4"
        },
        {
            "Field": "similarities",
            "Type": "array[float]",
            "Description": "Raw cosine similarities before normalization",
            "Example": "[0.72, 0.78, 0.85, 0.91, 0.82]"
        },
    ])

    st.dataframe(json_fields_df, use_container_width=True, hide_index=True)

    # Results summary JSON
    st.markdown("---")
    st.subheader(" results_summary.json Schema")

    summary_example = """{
  "experiment_id": "run_20240115_143022",
  "timestamp": "2024-01-15T14:30:22",
  "config": {
    "survey_name": "Product Feedback Survey",
    "n_questions": 3,
    "n_respondents": 100,
    "n_responses": 300
  },
  "overall_metrics": {
    "mode_accuracy": 0.72,
    "expected_value_mae": 0.45,
    "expected_value_rmse": 0.63,
    "kl_divergence": 0.28
  },
  "question_metrics": {
    "recommend": {
      "mode_accuracy": 0.75,
      "mae": 0.42,
      "rmse": 0.58,
      "kl_divergence": 0.25,
      "n_responses": 100
    },
    "quality": {
      "mode_accuracy": 0.68,
      "mae": 0.51,
      "rmse": 0.71,
      "kl_divergence": 0.32,
      "n_responses": 100
    }
  }
}
"""

    st.code(summary_example, language="json")

    st.info("""
    **Use Cases for Output Files:**
    - **llm_distributions.json**: Load into Python/R for custom analysis, visualizations
    - **results_summary.json**: Power dashboards, compare experiments programmatically
    - **report.txt/md**: Share with stakeholders, include in documentation
    - **ground_truth.csv**: Verify data integrity, reproduce results
    """)

# ======================
# Section 5: Data Flow
# ======================
elif section == " Data Flow Diagram":
    st.header(" Data Flow Through the Pipeline")

    st.markdown("""
    This diagram shows how data flows through the SSR pipeline from input to output,
    including all transformations and intermediate steps.
    """)

    # Create detailed flow diagram
    fig = go.Figure()

    # Define flow steps
    steps = [
        # Stage 1: Input
        {"name": "Ground Truth CSV", "x": 0.1, "y": 0.9, "stage": "input"},
        {"name": "Survey Config YAML", "x": 0.3, "y": 0.9, "stage": "input"},

        # Stage 2: Loading
        {"name": "Data Validation", "x": 0.2, "y": 0.75, "stage": "process"},

        # Stage 3: Processing
        {"name": "Text Embedding\n(OpenAI API)", "x": 0.2, "y": 0.6, "stage": "process"},
        {"name": "Semantic Similarity\n(Cosine)", "x": 0.5, "y": 0.6, "stage": "process"},
        {"name": "Normalization\n(Paper Method)", "x": 0.8, "y": 0.6, "stage": "process"},

        # Stage 4: Analysis
        {"name": "Probability\nDistributions", "x": 0.35, "y": 0.45, "stage": "intermediate"},
        {"name": "Metric Calculation", "x": 0.65, "y": 0.45, "stage": "intermediate"},

        # Stage 5: Output
        {"name": "LLM Distributions\nJSON", "x": 0.2, "y": 0.25, "stage": "output"},
        {"name": "Results Summary\nJSON", "x": 0.5, "y": 0.25, "stage": "output"},
        {"name": "Reports\n(TXT/MD)", "x": 0.8, "y": 0.25, "stage": "output"},
    ]

    # Color mapping
    stage_colors = {
        "input": brand_colors['cornflower_blue'],
        "process": brand_colors['teal_blue'],
        "intermediate": brand_colors['turquoise'],
        "output": brand_colors['atomic_orange']
    }

    # Add boxes
    for step in steps:
        fig.add_shape(
            type="rect",
            x0=step["x"]-0.08, y0=step["y"]-0.05,
            x1=step["x"]+0.08, y1=step["y"]+0.05,
            fillcolor=stage_colors[step["stage"]],
            line=dict(color="white", width=2),
        )
        fig.add_annotation(
            x=step["x"], y=step["y"],
            text=f"<b>{step['name']}</b>",
            showarrow=False,
            font=dict(size=10, color="white"),
        )

    # Add flow arrows
    flows = [
        # Input to validation
        {"from": 0, "to": 2},
        {"from": 1, "to": 2},
        # Validation to embedding
        {"from": 2, "to": 3},
        # Embedding to similarity
        {"from": 3, "to": 4},
        # Similarity to normalization
        {"from": 4, "to": 5},
        # Normalization to distributions
        {"from": 5, "to": 6},
        # Distributions to metrics
        {"from": 6, "to": 7},
        # To outputs
        {"from": 6, "to": 8},
        {"from": 7, "to": 9},
        {"from": 7, "to": 10},
    ]

    for flow in flows:
        from_step = steps[flow["from"]]
        to_step = steps[flow["to"]]
        fig.add_annotation(
            x=to_step["x"], y=to_step["y"]+0.05,
            ax=from_step["x"], ay=from_step["y"]-0.05,
            xref='x', yref='y',
            axref='x', ayref='y',
            showarrow=True,
            arrowhead=2,
            arrowsize=1,
            arrowwidth=1.5,
            arrowcolor=brand_colors['teal_dark']
        )

    # Add stage labels
    stage_labels = [
        {"text": "1. INPUT", "x": 0.2, "y": 0.98},
        {"text": "2. VALIDATION", "x": 0.2, "y": 0.83},
        {"text": "3. PROCESSING", "x": 0.5, "y": 0.68},
        {"text": "4. ANALYSIS", "x": 0.5, "y": 0.53},
        {"text": "5. OUTPUT", "x": 0.5, "y": 0.33},
    ]

    for label in stage_labels:
        fig.add_annotation(
            x=label["x"], y=label["y"],
            text=f"<b>{label['text']}</b>",
            showarrow=False,
            font=dict(size=12, color=brand_colors['teal_dark']),
        )

    fig.update_layout(
        showlegend=False,
        xaxis=dict(showgrid=False, showticklabels=False, zeroline=False, range=[0, 1]),
        yaxis=dict(showgrid=False, showticklabels=False, zeroline=False, range=[0.15, 1.05]),
        height=700,
        plot_bgcolor='white',
        margin=dict(l=20, r=20, t=20, b=20)
    )

    st.plotly_chart(fig, use_container_width=True)

    # Detailed step descriptions
    st.markdown("---")
    st.subheader(" Step-by-Step Breakdown")

    with st.expander("**Stage 1: INPUT** - Data Loading", expanded=True):
        st.markdown("""
        **1. Ground Truth CSV**
        - Read CSV file with pandas
        - Contains: respondent_id, question_id, response, rating

        **2. Survey Config YAML**
        - Parse YAML configuration
        - Extract: questions, scales, personas (optional)
        """)

    with st.expander("**Stage 2: VALIDATION** - Data Quality Checks"):
        st.markdown("""
        **Data Validation Steps:**
        -  Check all required columns exist
        -  Verify ratings are within scale range
        -  Match question_ids between data and config
        -  Check for missing/null values
        -  Validate scale definitions match data

        **Output:** Clean, validated dataset ready for processing
        """)

    with st.expander("**Stage 3: PROCESSING** - SSR Computation"):
        st.markdown("""
        **3.1 Text Embedding (OpenAI API)**
        - Convert text responses to embeddings using `text-embedding-3-small`
        - Convert scale labels to embeddings
        - Batch processing for efficiency
        - Results: Dense vectors (1536 dimensions)

        **3.2 Semantic Similarity (Cosine)**
        - Compute cosine similarity between response embedding and each scale label embedding
        - Results: Raw similarity scores [0, 1]

        **3.3 Normalization (Paper Method)**
        - Subtract minimum similarity from all scores
        - Apply temperature scaling (default: 1.0)
        - Normalize to sum to 1.0
        - Results: Probability distribution
        """)

    with st.expander("**Stage 4: ANALYSIS** - Metrics & Evaluation"):
        st.markdown("""
        **4.1 Probability Distributions**
        - Store full distribution for each response
        - Calculate mode (argmax), expected value (weighted avg)
        - Compute entropy (uncertainty measure)

        **4.2 Metric Calculation**
        - **Mode Accuracy**: % where SSR mode == ground truth rating
        - **MAE**: Mean Absolute Error between expected value and ground truth
        - **RMSE**: Root Mean Square Error
        - **KL Divergence**: Distributional distance
        - **Top-2 Accuracy**: % where ground truth in top 2 probabilities
        - **Prob at Truth**: Probability mass at ground truth rating
        """)

    with st.expander("**Stage 5: OUTPUT** - Report Generation"):
        st.markdown("""
        **5.1 LLM Distributions JSON**
        - Complete probability distributions for each response
        - Includes: distribution, mode, expected_value, entropy, similarities
        - Format: Nested dict by respondent_id and question_id

        **5.2 Results Summary JSON**
        - Aggregated metrics across all questions
        - Per-question metrics breakdown
        - Experiment metadata (timestamp, config)

        **5.3 Reports (TXT/MD)**
        - Human-readable comparison tables
        - Ground Truth vs SSR metrics side-by-side
        - Question-level and overall statistics
        """)

# ======================
# Section 6: Example Datasets
# ======================
elif section == " Example Datasets":
    st.header(" Example Datasets & Templates")

    st.markdown("""
    Download these templates to get started quickly with your own data.
    Each example demonstrates a different use case.
    """)

    # Example 1: Simple Likert-5 Survey
    st.subheader(" Simple Product Feedback (Likert-5)")

    st.markdown("**Use Case:** Basic product satisfaction survey with 5-point scales")

    col1, col2 = st.columns(2)

    with col1:
        st.markdown("**ground_truth.csv**")
        simple_gt = pd.DataFrame([
            {"respondent_id": "R001", "question_id": "recommend", "ground_truth": 5},
            {"respondent_id": "R001", "question_id": "quality", "ground_truth": 5},
            {"respondent_id": "R002", "question_id": "recommend", "ground_truth": 3},
            {"respondent_id": "R002", "question_id": "quality", "ground_truth": 3},
            {"respondent_id": "R003", "question_id": "recommend", "ground_truth": 2},
            {"respondent_id": "R003", "question_id": "quality", "ground_truth": 2},
        ])
        st.dataframe(simple_gt, use_container_width=True, hide_index=True)

        csv1 = simple_gt.to_csv(index=False)
        st.download_button(" Download CSV", csv1, "simple_ground_truth.csv", "text/csv")

    with col2:
        st.markdown("**survey_config.yaml**")
        simple_yaml = """name: "Product Feedback Survey"
description: "Evaluating customer satisfaction with our product"

context: |
  Please think about our SmartWidget Pro when answering these questions.
  The product features wireless connectivity, 12-hour battery life, and costs $199.

questions:
  - id: recommend
    text: "How likely are you to recommend?"
    type: likert_5
    scale:
      1: "Very unlikely"
      2: "Unlikely"
      3: "Neutral"
      4: "Likely"
      5: "Very likely"

  - id: quality
    text: "How would you rate the quality?"
    type: likert_5
    scale:
      1: "Very poor"
      2: "Poor"
      3: "Fair"
      4: "Good"
      5: "Excellent"
"""
        st.code(simple_yaml, language="yaml")
        st.download_button(" Download YAML", simple_yaml, "simple_config.yaml", "text/yaml")

    st.markdown("---")

    # Example 2: Mixed Scale Types
    st.subheader(" Mixed Scale Types (Binary + Likert)")

    st.markdown("**Use Case:** Survey with different question types (yes/no, ratings, etc.)")

    mixed_gt = pd.DataFrame([
        {"respondent_id": "R001", "question_id": "purchase", "ground_truth": 2},
        {"respondent_id": "R001", "question_id": "satisfaction", "ground_truth": 5},
        {"respondent_id": "R002", "question_id": "purchase", "ground_truth": 1},
        {"respondent_id": "R002", "question_id": "satisfaction", "ground_truth": 2},
    ])

    st.dataframe(mixed_gt, use_container_width=True, hide_index=True)

    mixed_yaml = """name: "Mixed Scale Survey"
description: "Customer purchase intent and satisfaction evaluation"

context: |
  Consider our new EcoBottle - a reusable water bottle with temperature control.
  Price: $49.99. Available in 5 colors. 24-hour hot/cold retention.

questions:
  - id: purchase
    text: "Would you purchase this product?"
    type: yes_no
    scale:
      1: "No"
      2: "Yes"

  - id: satisfaction
    text: "How satisfied are you?"
    type: likert_5
    scale:
      1: "Very dissatisfied"
      2: "Dissatisfied"
      3: "Neutral"
      4: "Satisfied"
      5: "Very satisfied"
"""

    col1, col2 = st.columns(2)
    with col1:
        csv3 = mixed_gt.to_csv(index=False)
        st.download_button(" Download CSV", csv3, "mixed_scales_ground_truth.csv", "text/csv", use_container_width=True)
    with col2:
        st.download_button(" Download YAML", mixed_yaml, "mixed_scales_config.yaml", "text/yaml", use_container_width=True)

    st.markdown("---")

    # Quick start guide
    st.success("""
    ###  Quick Start Guide

    1. **Download** one of the template pairs above (CSV + YAML)
    2. **Modify** the CSV with your ground truth data (human Likert scale ratings only)
    3. **Update** the YAML with your question text and scale labels
    4. **Upload** both files via the "Run Experiment" page
    5. The system will generate LLM text responses and apply SSR to predict ratings
    6. **View** comparison results (LLM+SSR vs. Ground Truth) in the Results Dashboard

    ** Tip:** Start with the "Simple Product Feedback" template if you're new to SSR.
    """)

    st.info("""
    **Need Help?**
    - Check the Live Demo page to see SSR in action on a single text response
    - Review the data quality requirements in the "Ground Truth Data Schema" section
    - Ensure your ground_truth values match the scale definitions in the config file
    - Remember: Ground truth = numeric ratings only (LLM generates the text responses)
    """)

# Footer
st.markdown("---")
st.markdown("""
<div style='text-align: center; color: #666; padding: 20px;'>
    <p><b>SSR Pipeline Technical Documentation</b></p>
    <p>For questions or support, refer to the project documentation or contact the development team.</p>
    <p style='font-size: 0.9em; margin-top: 10px;'>
        Based on: <i>"LLMs Reproduce Human Purchase Intent via Semantic Similarity Elicitation"</i> (arXiv:2510.08338v2)
    </p>
</div>
""", unsafe_allow_html=True)
