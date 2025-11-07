"""
S.A.G.E Home - Streamlined Dashboard
Main entry point with quick start wizard and intuitive navigation.
"""

import streamlit as st
from pathlib import Path
import sys
import pandas as pd

# Add parent directory to path for imports
sys.path.insert(0, str(Path(__file__).parent.parent))

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

# Page configuration
st.set_page_config(
    page_title="S.A.G.E - Home",
    page_icon="chart_with_upwards_trend",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS
st.markdown(f"""
<style>
    /* Main header styling */
    .main-header {{
        font-size: 3rem;
        font-weight: bold;
        color: {brand_colors['teal_blue']};
        margin-bottom: 0.5rem;
        text-align: center;
    }}
    .sub-header {{
        font-size: 1.3rem;
        color: #666;
        margin-bottom: 3rem;
        text-align: center;
    }}

    /* Quick action cards */
    .action-card {{
        background: linear-gradient(135deg, #f5f7fa 0%, #c3cfe2 100%);
        padding: 2rem;
        border-radius: 1rem;
        border: 2px solid {brand_colors['teal_blue']};
        text-align: center;
        transition: transform 0.2s, box-shadow 0.2s;
        cursor: pointer;
        min-height: 160px;
        display: flex;
        flex-direction: column;
        justify-content: space-between;
    }}
    .action-card:hover {{
        transform: translateY(-5px);
        box-shadow: 0 10px 25px rgba(54, 117, 136, 0.3);
    }}
    .action-icon {{
        font-size: 3rem;
        margin-bottom: 1rem;
    }}
    .action-title {{
        font-size: 1.5rem;
        font-weight: bold;
        color: {brand_colors['teal_blue']};
        margin-bottom: 0.5rem;
    }}
    .action-desc {{
        color: #666;
        font-size: 1rem;
        min-height: 3em;
    }}

    /* Primary buttons */
    .stButton > button[kind="primary"] {{
        background-color: {brand_colors['teal_blue']} !important;
        border-color: {brand_colors['teal_blue']} !important;
        color: white !important;
        font-weight: bold;
        padding: 0.75rem 2rem;
        font-size: 1.1rem;
        width: 100% !important;
        min-height: 3rem;
    }}
    .stButton > button[kind="primary"]:hover {{
        background-color: {brand_colors['teal_dark']} !important;
        border-color: {brand_colors['teal_dark']} !important;
        transform: translateY(-2px);
        box-shadow: 0 5px 15px rgba(54, 117, 136, 0.4);
    }}

    /* Ensure all buttons have consistent sizing */
    .stButton > button {{
        width: 100% !important;
        min-height: 3rem;
        padding: 0.75rem 2rem;
        font-size: 1.1rem;
    }}

    /* Metric cards */
    .metric-card {{
        background-color: white;
        padding: 1.5rem;
        border-radius: 0.75rem;
        border-left: 5px solid {brand_colors['turquoise']};
        box-shadow: 0 2px 8px rgba(0,0,0,0.1);
    }}

    /* Wizard steps */
    .wizard-step {{
        background-color: #f8f9fa;
        padding: 1.5rem;
        border-radius: 0.5rem;
        border-left: 4px solid {brand_colors['cornflower_blue']};
        margin-bottom: 1rem;
    }}
    .wizard-step-number {{
        display: inline-block;
        width: 2rem;
        height: 2rem;
        background-color: {brand_colors['cornflower_blue']};
        color: white;
        border-radius: 50%;
        text-align: center;
        line-height: 2rem;
        font-weight: bold;
        margin-right: 1rem;
    }}
</style>
""", unsafe_allow_html=True)

# Initialize session state
if 'persona_config' not in st.session_state:
    st.session_state.persona_config = {
        'age_groups': ["18-25", "26-35", "36-45", "46-55", "56-65", "65+"],
        'income_brackets': ["<$30k", "$30k-$50k", "$50k-$75k", "$75k-$100k", "$100k-$150k", ">$150k"],
        'env_consciousness': ["Not concerned", "Slightly concerned", "Moderately concerned",
                             "Very concerned", "Extremely concerned"]
    }

# Header
st.markdown('<div class="main-header">S.A.G.E</div>', unsafe_allow_html=True)
st.markdown('<div class="sub-header">Survey Analytics and Generation Engine</div>', unsafe_allow_html=True)

st.markdown("---")

# Main Action Cards
st.markdown("### Quick Actions")

col1, col2, col3, col4 = st.columns(4)

with col1:
    st.markdown("""
    <div class="action-card">
        <div class="action-title">Survey Preview</div>
        <div class="action-desc">Take surveys and collect real user responses</div>
    </div>
    """, unsafe_allow_html=True)
    if st.button("Take Survey", key="survey_preview", use_container_width=True, type="primary"):
        st.switch_page("pages/2_Survey_Preview.py")

with col2:
    st.markdown("""
    <div class="action-card">
        <div class="action-title">Run Experiment</div>
        <div class="action-desc">Process survey data with SSR and compare against ground truth</div>
    </div>
    """, unsafe_allow_html=True)
    if st.button("Start Experiment", key="run_exp", use_container_width=True, type="primary"):
        st.switch_page("pages/3_Run_Experiment.py")

with col3:
    st.markdown("""
    <div class="action-card">
        <div class="action-title">View Results</div>
        <div class="action-desc">Explore interactive dashboards and detailed metrics</div>
    </div>
    """, unsafe_allow_html=True)
    if st.button("Open Results", key="view_results", use_container_width=True, type="primary"):
        st.switch_page("pages/4_Results_Dashboard.py")

with col4:
    st.markdown("""
    <div class="action-card">
        <div class="action-title">Live Demo</div>
        <div class="action-desc">Test SSR on individual text responses in real-time</div>
    </div>
    """, unsafe_allow_html=True)
    if st.button("Try Demo", key="live_demo", use_container_width=True, type="primary"):
        st.switch_page("pages/6_Live_Demo.py")

st.markdown("---")

# Experiments Overview
st.markdown("### Experiments Overview")

experiments_dir = Path("experiments")
if not experiments_dir.exists():
    experiments_dir = Path("../experiments")

if experiments_dir.exists():
    experiment_folders = sorted(experiments_dir.glob("run_*"), reverse=True)

    if experiment_folders:
        # Summary metrics
        col1, col2, col3, col4 = st.columns(4)

        with col1:
            st.metric("Total Experiments", len(experiment_folders))

        with col2:
            latest = experiment_folders[0]
            timestamp = latest.name.replace("run_", "")
            formatted_time = f"{timestamp[4:6]}/{timestamp[6:8]} {timestamp[9:11]}:{timestamp[11:13]}"
            st.metric("Latest Run", formatted_time)

        with col3:
            # Count total responses across all experiments
            total_responses = 0
            for folder in experiment_folders:
                gt_file = folder / "ground_truth.csv"
                if gt_file.exists():
                    df = pd.read_csv(gt_file)
                    total_responses += len(df)
            st.metric("Total Responses", f"{total_responses:,}")

        with col4:
            # Count total questions
            latest_gt = latest / "ground_truth.csv"
            if latest_gt.exists():
                df = pd.read_csv(latest_gt)
                n_questions = df['question_id'].nunique()
                st.metric("Last Experiment Questions", n_questions)

        # Recent experiments table
        st.markdown("#### Recent Experiments")

        recent_data = []
        for folder in experiment_folders[:10]:  # Show top 10
            timestamp = folder.name.replace("run_", "")
            formatted_time = f"{timestamp[:4]}-{timestamp[4:6]}-{timestamp[6:8]} {timestamp[9:11]}:{timestamp[11:13]}"

            gt_file = folder / "ground_truth.csv"
            n_resp = "N/A"
            n_questions = "N/A"
            if gt_file.exists():
                df = pd.read_csv(gt_file)
                n_resp = len(df)
                n_questions = df['question_id'].nunique()

            recent_data.append({
                "Timestamp": formatted_time,
                "Responses": n_resp,
                "Questions": n_questions,
                "Folder": folder.name
            })

        if recent_data:
            recent_df = pd.DataFrame(recent_data)
            st.dataframe(recent_df, use_container_width=True, hide_index=True)

        # Quick view button for latest
        if st.button("View Latest Results", type="primary"):
            st.session_state.selected_experiment = str(experiment_folders[0])
            st.switch_page("pages/4_Results_Dashboard.py")
    else:
        st.info("No experiments yet! Click 'Start Experiment' above to run your first analysis.")
else:
    st.info("No experiments yet! Click 'Start Experiment' above to run your first analysis.")

# Advanced Options (collapsed)
st.markdown("---")
with st.expander("Advanced Options"):
    col1, col2, col3 = st.columns(3)

    with col1:
        if st.button("Settings", use_container_width=True):
            st.switch_page("pages/7_Settings.py")

    with col2:
        if st.button("Data Schema Guide", use_container_width=True):
            st.switch_page("pages/8_Data_Schema_Guide.py")

    with col3:
        if st.button("Compare Experiments", use_container_width=True):
            st.switch_page("pages/5_Compare_Experiments.py")

# Footer with what is SSR
st.markdown("---")
with st.expander("What is Semantic Similarity Rating (SSR)?"):
    st.markdown("""
    SSR uses semantic similarity to convert natural language survey responses into probability distributions over Likert scales.

    **Example:**
    - **Text response:** *"I'm very interested in this product!"*
    - **SSR output:** `[0.05, 0.10, 0.20, 0.30, 0.35]` (probabilities for ratings 1-5)
    - **Mode:** 5 (most likely rating)
    - **Expected value:** 3.85 (weighted average)

    **Key Benefits:**
    - Preserves uncertainty and nuance in responses
    - Enables richer statistical analysis
    - Validated methodology (90%+ accuracy)
    - Works with any embedding model (OpenAI, sentence-transformers, etc.)

    **Based on:** arXiv:2510.08338v2 - "LLMs Reproduce Human Purchase Intent via Semantic Similarity Elicitation"
    """)
