"""Run Experiment page - configure and execute the SSR pipeline."""

import streamlit as st
from pathlib import Path
import sys
import os
import subprocess
from datetime import datetime
import json

# Add parent directory to path
sys.path.insert(0, str(Path(__file__).parent.parent.parent))

from ui.utils.data_loader import load_survey_config, get_available_surveys
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

st.set_page_config(
    page_title="Run Experiment",
    page_icon="",
    layout="wide"
)

# Custom CSS for brand colors - Applied after page load to override Streamlit defaults
st.markdown(f"""
<style>
    /* Override Streamlit's default red color variables */
    :root {{
        --primary-color: {brand_colors['teal_blue']} !important;
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

    /* Checkboxes - More specific selectors */
    .stCheckbox input[type="checkbox"]:checked + div {{
        background-color: {brand_colors['teal_blue']} !important;
        border-color: {brand_colors['teal_blue']} !important;
    }}
    .stCheckbox input[type="checkbox"] + div {{
        border-color: {brand_colors['teal_blue']} !important;
    }}
    [data-baseweb="checkbox"] {{
        border-color: {brand_colors['teal_blue']} !important;
    }}
    [data-baseweb="checkbox"] > div:first-child {{
        border-color: {brand_colors['teal_blue']} !important;
    }}
    [data-baseweb="checkbox"][data-checked="true"] > div:first-child {{
        background-color: {brand_colors['teal_blue']} !important;
        border-color: {brand_colors['teal_blue']} !important;
    }}

    /* Sliders - More specific selectors */
    [data-baseweb="slider"] {{
        background-color: transparent !important;
    }}
    [data-baseweb="slider"] [data-testid="stThumb"] {{
        background-color: {brand_colors['teal_blue']} !important;
        border: 2px solid white !important;
    }}
    [data-baseweb="slider"] [data-testid="stThumbValue"] {{
        color: {brand_colors['teal_blue']} !important;
    }}
    [data-baseweb="slider"] > div:first-child {{
        background: linear-gradient(to right,
            {brand_colors['teal_blue']} 0%,
            {brand_colors['teal_blue']} var(--value),
            {brand_colors['light_grey']} var(--value),
            {brand_colors['light_grey']} 100%) !important;
    }}
    .stSlider > div > div > div[role="slider"] {{
        background-color: {brand_colors['teal_blue']} !important;
    }}
    /* Slider track */
    input[type="range"]::-webkit-slider-thumb {{
        background-color: {brand_colors['teal_blue']} !important;
    }}
    input[type="range"]::-moz-range-thumb {{
        background-color: {brand_colors['teal_blue']} !important;
    }}
    input[type="range"]::-webkit-slider-runnable-track {{
        background: linear-gradient(to right, {brand_colors['teal_blue']} 0%, {brand_colors['light_grey']} 0%) !important;
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

    /* File uploader */
    .stFileUploader > div > button {{
        border-color: {brand_colors['teal_blue']} !important;
        color: {brand_colors['teal_blue']} !important;
    }}
</style>
""", unsafe_allow_html=True)

st.title(" Run Experiment")

# Check API key
api_key = os.getenv("OPENAI_API_KEY") or st.session_state.get('api_key')

if not api_key:
    error_message("OpenAI API key not configured. Please set OPENAI_API_KEY environment variable.")
    st.stop()

# ======================
# Section A: Survey Configuration
# ======================
st.header(" Survey Configuration")

available_surveys = get_available_surveys()

if not available_surveys:
    error_message("No survey configurations found in config/ folder")
    st.stop()

# Survey selector
selected_survey = st.selectbox(
    "Select Survey",
    options=available_surveys,
    index=0 if available_surveys else None,
    help="Choose a survey configuration file"
)

if selected_survey:
    # Load and display survey preview
    survey_config = load_survey_config(selected_survey)

    if survey_config and 'survey' in survey_config:
        survey_data = survey_config['survey']

        col1, col2 = st.columns([2, 1])

        with col1:
            st.subheader(survey_data.get('name', 'Survey'))
            st.markdown(f"**Description:** {survey_data.get('description', 'N/A')}")

        with col2:
            n_questions = len(survey_data.get('questions', []))
            sample_size = survey_data.get('sample_size', 50)
            st.metric("Questions", n_questions)
            st.metric("Default Sample Size", sample_size)

        # Survey questions preview
        with st.expander(" View Survey Questions", expanded=False):
            questions = survey_data.get('questions', [])

            for i, q in enumerate(questions, 1):
                st.markdown(f"**{i}. {q.get('id')}** - {q.get('type')}")
                st.markdown(f"*{q.get('text')}*")

                # Show scale/options
                if 'scale' in q:
                    scale_preview = ", ".join([f"{k}: {v}" for k, v in q['scale'].items()])
                    st.caption(f"Scale: {scale_preview}")
                elif 'options' in q:
                    options_preview = ", ".join(q['options'])
                    st.caption(f"Options: {options_preview}")

                st.markdown("---")

# ======================
# Section B: Experiment Settings
# ======================
st.header(" Experiment Settings")

col1, col2 = st.columns(2)

with col1:
    n_respondents = st.slider(
        "Number of Respondents",
        min_value=10,
        max_value=200,
        value=50,
        step=10,
        help="Number of synthetic respondents to generate"
    )

with col2:
    random_seed = st.number_input(
        "Random Seed",
        min_value=0,
        max_value=10000,
        value=100,
        help="Seed for reproducibility"
    )

# Response styles
st.subheader("Response Styles to Test")

col1, col2 = st.columns(2)

with col1:
    test_human = st.checkbox(
        " Human-style responses",
        value=True,
        help="Direct, opinionated responses (e.g., 'Definitely yes!')"
    )

with col2:
    test_llm = st.checkbox(
        " LLM-style responses",
        value=True,
        help="Hedged, nuanced responses (e.g., 'I would say that...')"
    )

if not test_human and not test_llm:
    warning_message("Please select at least one response style to test")

# ======================
# Section C: Persona Configuration
# ======================
st.header(" Persona Configuration")

st.markdown("""
Personas are defined in the survey YAML config file.
These personas are used to generate LLM responses and should match the personas
used in human ground truth data collection.
""")

# Load personas from survey config (if survey was loaded)
if selected_survey and survey_config and 'survey' in survey_config:
    survey_personas = survey_config['survey'].get('personas', [])

    if survey_personas:
        # Show persona pool from config
        with st.expander(f" Persona Pool from Config ({len(survey_personas)} personas)", expanded=False):
            st.markdown("**Personas defined in survey config:**")
            for i, persona in enumerate(survey_personas, 1):
                st.markdown(f"{i}. {persona}")

            st.info(f"Edit {selected_survey} to modify personas")

        # Show current configuration summary
        st.markdown("**Current Configuration:**")
        col1, col2 = st.columns(2)

        with col1:
            st.metric("Persona Pool Size", len(survey_personas))

        with col2:
            st.metric("Selection Method", "Random")
    else:
        warning_message("No personas defined in survey config. Using default personas.")

# ======================
# Section D: Ground Truth Data
# ======================
st.header(" Ground Truth Data")

st.markdown("""
You can either upload real human ground truth data or generate artificial ground truth.
""")

# Ground truth option
ground_truth_option = st.radio(
    "Ground Truth Source",
    ["Generate Artificial Ground Truth", "Upload Real Human Data"],
    help="Choose whether to generate random ground truth or upload real human survey responses"
)

uploaded_ground_truth = None

if ground_truth_option == "Upload Real Human Data":
    st.markdown("**Upload CSV File:**")
    st.markdown("**Required format:**")
    st.markdown("- Columns: `respondent_id`, `question_id`, `ground_truth`")
    st.markdown(f"- `question_id` **must exactly match** question IDs from survey config:")

    if survey_config and 'survey' in survey_config:
        survey_questions = [q['id'] for q in survey_config['survey'].get('questions', [])]
        st.code(", ".join(survey_questions))

    st.warning("IMPORTANT: question_id values in your CSV must exactly match the question IDs shown above.")

    uploaded_file = st.file_uploader(
        "Choose ground truth CSV file",
        type=['csv'],
        help="Upload a CSV with ground truth ratings from human respondents"
    )

    if uploaded_file:
        # Preview uploaded file
        import pandas as pd
        import tempfile
        import shutil

        # Read and validate
        try:
            df = pd.read_csv(uploaded_file)

            # Check required columns
            required_cols = ['respondent_id', 'question_id', 'ground_truth']
            if all(col in df.columns for col in required_cols):
                # Validate question IDs match survey config
                csv_questions = set(df['question_id'].unique())
                survey_questions = set([q['id'] for q in survey_config['survey'].get('questions', [])])

                # Check for mismatches
                extra_questions = csv_questions - survey_questions
                missing_questions = survey_questions - csv_questions

                if extra_questions:
                    error_message(f"CSV contains question IDs not in survey config: {', '.join(extra_questions)}")
                elif missing_questions:
                    warning_message(f"CSV is missing some questions from survey: {', '.join(missing_questions)}")
                else:
                    success_message(f"Valid ground truth file uploaded: {len(df)} ratings")

                # Show preview regardless (allow partial data)
                with st.expander("Preview Ground Truth Data"):
                    st.dataframe(df.head(10))
                    st.markdown(f"**Total rows:** {len(df)}")
                    st.markdown(f"**Unique respondents:** {df['respondent_id'].nunique()}")
                    st.markdown(f"**Questions in CSV:** {', '.join(sorted(df['question_id'].unique()))}")
                    st.markdown(f"**Questions in Config:** {', '.join(sorted(survey_questions))}")

                    if extra_questions or missing_questions:
                        if extra_questions:
                            st.error(f"Extra questions: {', '.join(extra_questions)}")
                        if missing_questions:
                            st.warning(f"Missing questions: {', '.join(missing_questions)}")

                # Save to temp file for pipeline (even if warning)
                if not extra_questions:  # Only proceed if no invalid question IDs
                    temp_dir = Path("temp")
                    temp_dir.mkdir(exist_ok=True)
                    temp_path = temp_dir / f"ground_truth_{datetime.now().strftime('%Y%m%d_%H%M%S')}.csv"
                    df.to_csv(temp_path, index=False)
                    uploaded_ground_truth = str(temp_path)

            else:
                error_message(f"CSV must have columns: {required_cols}. Found: {list(df.columns)}")

        except Exception as e:
            error_message(f"Error reading CSV: {str(e)}")
else:
    st.info("Artificial ground truth will be generated based on persona descriptions")

# ======================
# Section E: SSR Configuration
# ======================
st.header(" SSR Configuration")

st.markdown("**Paper-exact settings** (following arXiv:2510.08338v2)")

col1, col2, col3 = st.columns(3)

with col1:
    st.metric("Model", "text-embedding-3-small")
    st.caption("OpenAI embedding model")

with col2:
    temperature = st.slider(
        "Temperature",
        min_value=0.1,
        max_value=3.0,
        value=1.0,
        step=0.1,
        help="Controls distribution spread (1.0 = paper default)"
    )

with col3:
    st.metric("Normalization", "Paper Method")
    st.caption("Subtract min + proportional")

# ======================
# Section F: Run Experiment
# ======================
st.markdown("---")
st.header(" Execute Pipeline")

# Preview experiment configuration
with st.expander(" Experiment Summary", expanded=True):
    st.markdown(f"""
    **Survey:** {survey_data.get('name', 'Unknown')}
    - Questions: {len(survey_data.get('questions', []))}
    - Respondents: {n_respondents}
    - Random Seed: {random_seed}

    **Response Styles:**
    - Human-style: {'' if test_human else ''}
    - LLM-style: {'' if test_llm else ''}

    **Persona Pool:**
    - Pool Size: {len(survey_personas) if survey_personas else 0} unique personas
    - Selection: Random sampling with replacement
    - Source: Survey YAML config

    **Ground Truth:**
    - Source: {"Uploaded Human Data" if uploaded_ground_truth else "Artificial Generation"}
    {"- File: " + uploaded_ground_truth if uploaded_ground_truth else ""}

    **SSR Settings:**
    - Model: text-embedding-3-small
    - Temperature: {temperature}
    - Normalization: Paper method

    **Expected Outputs:**
    - Ground truth CSV ({n_respondents * len(survey_data.get('questions', []))} ratings)
    - Visual report (PNG)
    - Text metrics (TXT)
    - Comprehensive report (MD)
    """)

# Run button
run_button = st.button(
    " Run Experiment",
    type="primary",
    use_container_width=True,
    disabled=not (test_human or test_llm)
)

if run_button:
    # Create a progress container
    progress_container = st.container()

    with progress_container:
        st.markdown("###  Running Experiment...")

        progress_bar = st.progress(0)
        status_text = st.empty()

        # Create temporary Python script to run the pipeline
        # (We'll use subprocess to run ground_truth_pipeline.py)

        try:
            # Step 1: Preparing
            status_text.text("Step 1/8: Preparing environment...")
            progress_bar.progress(10)

            # Set environment variable for API key
            env = os.environ.copy()
            if api_key:
                env['OPENAI_API_KEY'] = api_key

            # Step 2: Running pipeline
            status_text.text("Step 2/8: Initializing pipeline...")
            progress_bar.progress(20)

            # Build persona configuration (empty, will use survey config)
            persona_config = {}

            # Build command with persona config, optional ground truth, and survey config path
            persona_config_json = json.dumps(persona_config)
            cmd = [
                sys.executable,
                "ground_truth_pipeline.py",
                persona_config_json
            ]

            # Add ground truth path if uploaded (must be second argument)
            if uploaded_ground_truth:
                cmd.append(uploaded_ground_truth)
            else:
                # Add empty string as placeholder if no ground truth
                cmd.append("")

            # Add survey config path (third argument)
            cmd.append(selected_survey)

            # Run the pipeline as subprocess
            status_text.text("Step 3/8: Loading survey...")
            progress_bar.progress(30)

            status_text.text("Step 4/8: Generating profiles...")
            progress_bar.progress(40)

            status_text.text("Step 5/8: Creating ground truth...")
            progress_bar.progress(50)

            status_text.text("Step 6/8: Generating responses...")
            progress_bar.progress(60)

            status_text.text("Step 7/8: Applying SSR (this may take a few minutes)...")
            progress_bar.progress(70)

            # Execute pipeline
            process = subprocess.Popen(
                cmd,
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
                text=True,
                env=env,
                cwd=str(Path(__file__).parent.parent.parent)
            )

            # Wait for completion
            stdout, stderr = process.communicate()

            if process.returncode == 0:
                status_text.text("Step 8/8: Generating reports...")
                progress_bar.progress(90)

                # Find the latest experiment folder
                experiments_dir = Path("experiments")
                if experiments_dir.exists():
                    experiment_folders = sorted(experiments_dir.glob("run_*"), reverse=True)
                    if experiment_folders:
                        latest_experiment = experiment_folders[0]

                        progress_bar.progress(100)
                        status_text.text(" Experiment completed successfully!")

                        success_message(f"Experiment completed! Results saved to: {latest_experiment.name}")

                        # Show quick results
                        st.markdown("###  Quick Results")

                        # Load ground truth
                        gt_file = latest_experiment / "ground_truth.csv"
                        if gt_file.exists():
                            import pandas as pd
                            gt_df = pd.read_csv(gt_file)

                            col1, col2, col3 = st.columns(3)

                            with col1:
                                st.metric("Total Responses", len(gt_df))

                            with col2:
                                n_resp = gt_df['respondent_id'].nunique()
                                st.metric("Respondents", n_resp)

                            with col3:
                                n_q = gt_df['question_id'].nunique()
                                st.metric("Questions", n_q)

                        # Store experiment path for navigation
                        st.session_state.selected_experiment = latest_experiment
                        st.session_state.latest_experiment_path = str(latest_experiment)

                        # Action buttons
                        st.markdown("---")
                        col1, col2 = st.columns(2)

                        with col1:
                            # Use page_link for navigation
                            st.page_link("pages/3_Results_Dashboard.py", label="View Results")

                        with col2:
                            if st.button("Run Another", use_container_width=True, key="run_another_btn"):
                                st.rerun()

                    else:
                        error_message("Experiment completed but no results folder found")
                else:
                    error_message("Experiments directory not found")

            else:
                progress_bar.progress(0)
                status_text.text("")
                error_message(f"Pipeline execution failed with return code {process.returncode}")

                # Show error details
                with st.expander("Error Details"):
                    st.code(stderr)
                    st.code(stdout)

        except Exception as e:
            progress_bar.progress(0)
            status_text.text("")
            error_message(f"Error running experiment: {str(e)}")

            with st.expander("Error Details"):
                import traceback
                st.code(traceback.format_exc())

# ======================
# Tips Section
# ======================
st.markdown("---")
st.markdown("###  Tips")

with st.expander("How to choose parameters"):
    st.markdown("""
    **Number of Respondents:**
    - Start with 50 for quick tests
    - Use 100-200 for more robust results
    - Higher numbers = longer processing time but better statistics

    **Random Seed:**
    - Use the same seed to reproduce results
    - Change seed to get different synthetic respondents

    **Temperature:**
    - 1.0 (default): Balanced distributions as per paper
    - <1.0: More peaked, confident predictions
    - >1.0: More spread, uncertain predictions

    **Persona Configuration:**
    - More categories = more diversity
    - Environmental consciousness directly influences ground truth generation
    - Profiles are sampled randomly from your categories
    """)

with st.expander("Expected processing time"):
    st.markdown("""
    Processing time depends on:
    - Number of respondents
    - Number of questions
    - OpenAI API latency

    **Typical times:**
    - 50 respondents, 6 questions: ~2-3 minutes
    - 100 respondents, 6 questions: ~4-5 minutes
    - 200 respondents, 6 questions: ~8-10 minutes

    The SSR step (applying embeddings) takes the most time.
    """)
