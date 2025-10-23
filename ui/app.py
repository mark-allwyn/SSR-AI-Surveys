"""
Streamlit UI for SSR Pipeline
Main entry point for the web interface.
"""

import streamlit as st
from pathlib import Path
import sys

# Add parent directory to path for imports
sys.path.insert(0, str(Path(__file__).parent.parent))

# Page configuration
st.set_page_config(
    page_title="SSR Pipeline",
    page_icon="chart_with_upwards_trend",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS
st.markdown("""
<style>
    .main-header {
        font-size: 2.5rem;
        font-weight: bold;
        color: #1f77b4;
        margin-bottom: 0.5rem;
    }
    .sub-header {
        font-size: 1.2rem;
        color: #666;
        margin-bottom: 2rem;
    }
    .metric-card {
        background-color: #f0f2f6;
        padding: 1.5rem;
        border-radius: 0.5rem;
        border-left: 4px solid #1f77b4;
    }
    .success-box {
        background-color: #d4edda;
        border: 1px solid #c3e6cb;
        border-radius: 0.25rem;
        padding: 1rem;
        margin: 1rem 0;
    }
    .warning-box {
        background-color: #fff3cd;
        border: 1px solid #ffeaa7;
        border-radius: 0.25rem;
        padding: 1rem;
        margin: 1rem 0;
    }
    .error-box {
        background-color: #f8d7da;
        border: 1px solid #f5c6cb;
        border-radius: 0.25rem;
        padding: 1rem;
        margin: 1rem 0;
    }
</style>
""", unsafe_allow_html=True)

# Initialize session state
if 'api_key' not in st.session_state:
    st.session_state.api_key = None

if 'selected_experiment' not in st.session_state:
    st.session_state.selected_experiment = None

if 'persona_config' not in st.session_state:
    st.session_state.persona_config = {
        'age_groups': ["18-25", "26-35", "36-45", "46-55", "56-65", "65+"],
        'income_brackets': ["<$30k", "$30k-$50k", "$50k-$75k", "$75k-$100k", "$100k-$150k", ">$150k"],
        'env_consciousness': ["Not concerned", "Slightly concerned", "Moderately concerned",
                             "Very concerned", "Extremely concerned"]
    }

# Sidebar
with st.sidebar:
    st.image("https://via.placeholder.com/200x80/1f77b4/ffffff?text=SSR+Pipeline", use_container_width=True)

    st.markdown("---")

    # Navigation
    st.markdown("### Navigation")
    st.markdown("Use the pages in the sidebar to navigate through the app.")

    st.markdown("---")

    # API Status
    st.markdown("### API Status")
    if st.session_state.api_key:
        st.success("API Key Configured")
    else:
        st.warning("API Key Not Set")
        if st.button("Configure API Key", use_container_width=True):
            st.switch_page("pages/5_Settings.py")

    st.markdown("---")

    # Quick Info
    st.markdown("### About")
    st.markdown("""
    **SSR Pipeline** implements Semantic Similarity Rating from
    [arXiv:2510.08338v2](https://arxiv.org/abs/2510.08338v2)

    Convert textual survey responses to probability distributions.
    """)

    st.markdown("---")
    st.markdown("Made with Streamlit")

# Main content
st.markdown('<div class="main-header">SSR Pipeline</div>', unsafe_allow_html=True)
st.markdown('<div class="sub-header">Semantic Similarity Rating for Survey Analysis</div>', unsafe_allow_html=True)

# Welcome section
st.markdown("""
## Welcome to the SSR Pipeline

This tool helps you analyze survey responses using Semantic Similarity Rating (SSR),
a methodology that converts textual responses into probability distributions over Likert scales.

### Getting Started

1. **Configure API Key**: Go to Settings to add your OpenAI API key
2. **Run an Experiment**: Use the Run Experiment page to process survey data
3. **View Results**: Analyze the results and compare human vs LLM responses
4. **Try Live Demo**: Test SSR on individual text responses

### What is SSR?

SSR uses semantic similarity to convert text like *"I'm very interested!"* into probability
distributions like `[0.05, 0.10, 0.20, 0.30, 0.35]` across Likert scale options.

**Key Features:**
- ✅ Paper-exact implementation (OpenAI text-embedding-3-small)
- ✅ Multiple question types (yes/no, Likert-5, Likert-7, multiple choice)
- ✅ Ground truth evaluation with 7+ metrics
- ✅ Human vs LLM response comparison
- ✅ Customizable persona parameters
- ✅ Automated reporting
""")

# Quick stats from experiments
experiments_dir = Path("experiments")
if experiments_dir.exists():
    experiment_folders = sorted(experiments_dir.glob("run_*"), reverse=True)

    if experiment_folders:
        st.markdown("---")
        st.markdown("### Recent Activity")

        col1, col2, col3 = st.columns(3)

        with col1:
            st.metric("Total Experiments", len(experiment_folders))

        with col2:
            latest = experiment_folders[0]
            timestamp = latest.name.replace("run_", "")
            # Format timestamp
            formatted_time = f"{timestamp[:4]}-{timestamp[4:6]}-{timestamp[6:8]} {timestamp[9:11]}:{timestamp[11:13]}"
            st.metric("Latest Experiment", formatted_time)

        with col3:
            # Count total ground truth entries
            latest_gt = latest / "ground_truth.csv"
            if latest_gt.exists():
                import pandas as pd
                df = pd.read_csv(latest_gt)
                n_responses = len(df)
                st.metric("Last Run Responses", n_responses)

        # Recent experiments table
        st.markdown("#### Recent Experiments")

        recent_data = []
        for folder in experiment_folders[:5]:
            timestamp = folder.name.replace("run_", "")
            formatted_time = f"{timestamp[:4]}-{timestamp[4:6]}-{timestamp[6:8]} {timestamp[9:11]}:{timestamp[11:13]}"

            gt_file = folder / "ground_truth.csv"
            n_resp = "N/A"
            if gt_file.exists():
                import pandas as pd
                df = pd.read_csv(gt_file)
                n_resp = len(df)

            recent_data.append({
                "Timestamp": formatted_time,
                "Folder": folder.name,
                "Responses": n_resp
            })

        if recent_data:
            import pandas as pd
            recent_df = pd.DataFrame(recent_data)
            st.dataframe(recent_df, use_container_width=True, hide_index=True)
    else:
        st.info("No experiments yet! Head to the Run Experiment page to get started.")
else:
    st.info("No experiments yet! Head to the Run Experiment page to get started.")

# Call to action
st.markdown("---")
col1, col2, col3, col4 = st.columns(4)

with col1:
    if st.button("Run Experiment", use_container_width=True, type="primary"):
        st.switch_page("pages/2_Run_Experiment.py")

with col2:
    if st.button("View Results", use_container_width=True):
        st.switch_page("pages/3_View_Results.py")

with col3:
    if st.button("Live Demo", use_container_width=True):
        st.switch_page("pages/4_Live_Demo.py")

with col4:
    if st.button("Settings", use_container_width=True):
        st.switch_page("pages/5_Settings.py")
