"""Settings page for experiment management."""

import streamlit as st
from pathlib import Path
import sys
import pandas as pd

# Add parent directory to path
sys.path.insert(0, str(Path(__file__).parent.parent.parent))

from ui.utils.data_loader import get_all_experiments, get_experiment_info, delete_experiment
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

st.set_page_config(page_title="Settings", page_icon="", layout="wide")

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

    /* Selectbox */
    .stSelectbox > div > div > div:focus {{
        border-color: {brand_colors['teal_blue']} !important;
        box-shadow: 0 0 0 0.2rem {brand_colors['teal_blue']}33 !important;
    }}
</style>
""", unsafe_allow_html=True)

st.title("Settings & Configuration")

# Tabs for different settings
tab1, tab2 = st.tabs(["Experiment Management", "About"])

# ======================
# Tab 1: Experiment Management
# ======================
with tab1:
    st.header("Experiment Management")

    # Get all experiments
    experiments = get_all_experiments()

    if not experiments:
        st.info("No experiments found. Run an experiment to see results here.")
    else:
        st.markdown(f"**Total Experiments:** {len(experiments)}")

        st.markdown("---")

        # Display experiments in a table
        experiment_data = []
        for exp_path in experiments:
            info = get_experiment_info(exp_path)
            experiment_data.append({
                "Timestamp": info['timestamp'],
                "Folder": info['folder'],
                "Respondents": info['n_respondents'],
                "Questions": info['n_questions'],
                "Responses": info['n_responses']
            })

        df = pd.DataFrame(experiment_data)
        st.dataframe(df, use_container_width=True, hide_index=True)

        st.markdown("---")

        # Delete experiments
        st.subheader("Delete Experiments")
        warning_message("Deletion is permanent and cannot be undone!")

        # Select experiment to delete
        experiment_options = {info['timestamp']: path for path, info in
                             [(exp, get_experiment_info(exp)) for exp in experiments]}

        selected_to_delete = st.selectbox(
            "Select experiment to delete",
            options=list(experiment_options.keys()),
            index=None,
            placeholder="Choose an experiment..."
        )

        if selected_to_delete:
            exp_path = experiment_options[selected_to_delete]
            exp_info = get_experiment_info(exp_path)

            st.warning(f"""
            **You are about to delete:**
            - Folder: {exp_info['folder']}
            - Timestamp: {exp_info['timestamp']}
            - Responses: {exp_info['n_responses']}

            This will permanently delete all files including ground truth, reports, and visualizations.
            """)

            col1, col2 = st.columns(2)

            with col1:
                confirm = st.checkbox("I understand this action cannot be undone")

            with col2:
                if st.button("Delete Experiment", type="primary", disabled=not confirm):
                    if delete_experiment(exp_path):
                        success_message(f"Experiment {exp_info['folder']} deleted successfully")
                        st.rerun()
                    else:
                        error_message("Failed to delete experiment")

        # Bulk delete
        st.markdown("---")
        st.subheader("Bulk Delete")

        if len(experiments) > 1:
            num_to_keep = st.number_input(
                "Keep most recent N experiments, delete the rest",
                min_value=0,
                max_value=len(experiments),
                value=min(5, len(experiments)),
                help="Keeps the N most recent experiments and deletes all older ones"
            )

            experiments_to_delete = experiments[num_to_keep:]

            if experiments_to_delete:
                st.warning(f"This will delete {len(experiments_to_delete)} experiment(s)")

                col1, col2 = st.columns(2)

                with col1:
                    bulk_confirm = st.checkbox("I confirm bulk deletion")

                with col2:
                    if st.button("Delete Old Experiments", type="primary", disabled=not bulk_confirm):
                        deleted_count = 0
                        for exp in experiments_to_delete:
                            if delete_experiment(exp):
                                deleted_count += 1

                        success_message(f"Deleted {deleted_count} experiment(s)")
                        st.rerun()

# ======================
# Tab 2: About
# ======================
with tab2:
    st.header("About S.A.G.E")

    st.markdown("""
    ## Semantic Similarity Rating

    This implementation follows the methodology from the paper:

    **"LLMs Reproduce Human Purchase Intent via Semantic Similarity Elicitation"**

    [arXiv:2510.08338v2](https://arxiv.org/abs/2510.08338v2)

    ### Features

    - OpenAI text-embedding-3-small model
    - Paper's normalization method (subtract min + proportional)
    - Multiple question types supported
    - Ground truth evaluation with 7+ metrics
    - Human vs LLM response comparison
    - Customizable persona parameters
    - Automated reporting (PNG, TXT, MD)

    ### Technology Stack

    - **Backend**: Python 3.10+
    - **UI**: Streamlit
    - **Embeddings**: OpenAI API
    - **Analysis**: pandas, numpy, scikit-learn
    - **Visualization**: matplotlib, seaborn, plotly

    ### Citation

    If you use this tool in your research, please cite the original paper:

    ```bibtex
    @article{ssr2024,
      title={LLMs Reproduce Human Purchase Intent via Semantic Similarity Elicitation},
      journal={arXiv preprint arXiv:2510.08338v2},
      year={2024}
    }
    ```

    ### License

    This project is provided for research and educational purposes.
    """)
