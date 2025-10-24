"""
Streamlit UI for SSR Pipeline
Main entry point for the web interface.
"""

import streamlit as st
from pathlib import Path
import sys

# Add parent directory to path for imports
sys.path.insert(0, str(Path(__file__).parent.parent))

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
    page_title="Home - S.A.G.E",
    page_icon="chart_with_upwards_trend",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS
st.markdown(f"""
<style>
    .main-header {{
        font-size: 2.5rem;
        font-weight: bold;
        color: {brand_colors['teal_blue']};
        margin-bottom: 0.5rem;
    }}
    .sub-header {{
        font-size: 1.2rem;
        color: #666;
        margin-bottom: 2rem;
    }}
    .metric-card {{
        background-color: #f0f2f6;
        padding: 1.5rem;
        border-radius: 0.5rem;
        border-left: 4px solid {brand_colors['teal_blue']};
    }}
    .success-box {{
        background-color: rgba(64, 224, 208, 0.15);
        border: 1px solid {brand_colors['turquoise']};
        border-radius: 0.25rem;
        padding: 1rem;
        margin: 1rem 0;
        color: {brand_colors['turquoise']};
    }}
    .warning-box {{
        background-color: rgba(204, 255, 0, 0.15);
        border: 1px solid {brand_colors['electric_lime']};
        border-radius: 0.25rem;
        padding: 1rem;
        margin: 1rem 0;
        color: {brand_colors['electric_lime']};
    }}
    .error-box {{
        background-color: rgba(255, 110, 58, 0.15);
        border: 1px solid {brand_colors['atomic_orange']};
        border-radius: 0.25rem;
        padding: 1rem;
        margin: 1rem 0;
        color: {brand_colors['atomic_orange']};
    }}

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
    /* Secondary buttons */
    .stButton > button[kind="secondary"] {{
        background-color: white !important;
        border-color: {brand_colors['teal_blue']} !important;
        color: {brand_colors['teal_blue']} !important;
    }}
    .stButton > button[kind="secondary"]:hover {{
        background-color: {brand_colors['light_grey']} !important;
    }}

    /* Radio buttons */
    .stRadio > label > div[data-testid="stMarkdownContainer"] > p {{
        color: {brand_colors['teal_blue']};
    }}
    .stRadio > div[role="radiogroup"] > label > div[data-baseweb="radio"] > div:first-child {{
        background-color: white !important;
        border-color: {brand_colors['teal_blue']} !important;
    }}
    .stRadio > div[role="radiogroup"] > label > div[data-baseweb="radio"] > div:first-child:after {{
        background-color: {brand_colors['teal_blue']} !important;
    }}

    /* Checkboxes */
    .stCheckbox > label > div[data-baseweb="checkbox"] > div {{
        border-color: {brand_colors['teal_blue']} !important;
    }}
    .stCheckbox > label > div[data-baseweb="checkbox"] > div[data-checked="true"] {{
        background-color: {brand_colors['teal_blue']} !important;
    }}

    /* Sliders */
    .stSlider > div > div > div > div {{
        background-color: {brand_colors['teal_blue']} !important;
    }}
    .stSlider > div > div > div > div > div {{
        background-color: {brand_colors['teal_blue']} !important;
    }}
    .stSlider > div > div > div > div > div > div {{
        background-color: {brand_colors['teal_blue']} !important;
        border-color: {brand_colors['teal_blue']} !important;
    }}

    /* Number input */
    .stNumberInput > div > div > input:focus {{
        border-color: {brand_colors['teal_blue']} !important;
        box-shadow: 0 0 0 0.2rem {brand_colors['teal_blue']}33 !important;
    }}

    /* Selectbox */
    .stSelectbox > div > div > div:focus {{
        border-color: {brand_colors['teal_blue']} !important;
        box-shadow: 0 0 0 0.2rem {brand_colors['teal_blue']}33 !important;
    }}
</style>
""", unsafe_allow_html=True)

# Initialize session state
if 'selected_experiment' not in st.session_state:
    st.session_state.selected_experiment = None

if 'persona_config' not in st.session_state:
    st.session_state.persona_config = {
        'age_groups': ["18-25", "26-35", "36-45", "46-55", "56-65", "65+"],
        'income_brackets': ["<$30k", "$30k-$50k", "$50k-$75k", "$75k-$100k", "$100k-$150k", ">$150k"],
        'env_consciousness': ["Not concerned", "Slightly concerned", "Moderately concerned",
                             "Very concerned", "Extremely concerned"]
    }

# Sidebar (empty - just for Streamlit navigation)
with st.sidebar:
    pass

# Main content
st.markdown('<div class="main-header">S.A.G.E</div>', unsafe_allow_html=True)
st.markdown('<div class="sub-header">Survey Analytics and Generation Engine</div>', unsafe_allow_html=True)

# Welcome section
st.markdown("""
S.A.G.E (Survey Analytics and Generation Engine) helps you analyze survey responses using Semantic Similarity Rating (SSR),
a methodology that converts textual responses into probability distributions over Likert scales.

### Getting Started

1. **Run an Experiment**: Use the Run Experiment page to process survey data
2. **View Results Dashboard**: Analyze interactive visualizations and detailed metrics
3. **Try Live Demo**: Test SSR on individual text responses

### What is SSR?

SSR uses semantic similarity to convert text like *"I'm very interested!"* into probability
distributions like `[0.05, 0.10, 0.20, 0.30, 0.35]` across Likert scale options.

**Key Features:**
- Multiple question types (yes/no, Likert-5, Likert-7, multiple choice)
- Ground truth evaluation with 7+ metrics
- Human vs LLM response comparison
- Customizable persona parameters
- Automated reporting
- OpenAI text-embedding-3-small model
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
col1, col2, col3 = st.columns(3)

with col1:
    if st.button("Run Experiment", use_container_width=True, type="primary"):
        st.switch_page("pages/2_Run_Experiment.py")

with col2:
    if st.button("Results Dashboard", use_container_width=True):
        st.switch_page("pages/3_Results_Dashboard.py")

with col3:
    if st.button("Live Demo", use_container_width=True):
        st.switch_page("pages/4_Live_Demo.py")
