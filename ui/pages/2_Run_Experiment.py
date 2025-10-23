"""Run Experiment page - configure and execute the SSR pipeline."""

import streamlit as st
from pathlib import Path
import sys
import os
import subprocess
from datetime import datetime

# Add parent directory to path
sys.path.insert(0, str(Path(__file__).parent.parent.parent))

from ui.utils.data_loader import load_survey_config, get_available_surveys
from ui.components.metrics_cards import success_message, warning_message, error_message

st.set_page_config(page_title="Run Experiment", page_icon="", layout="wide")

st.title(" Run Experiment")

# Check API key
api_key = os.getenv("OPENAI_API_KEY") or st.session_state.get('api_key')

if not api_key:
    error_message(" OpenAI API key not configured. Please go to Settings to add your API key.")
    if st.button("Go to Settings"):
        st.switch_page("pages/5__Settings.py")
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
Configure the persona parameters for generating synthetic respondents.
These determine the diversity of your simulated survey respondents.
""")

# Get persona config from session state or use defaults
if 'persona_config' not in st.session_state:
    st.session_state.persona_config = {
        'age_groups': ["18-25", "26-35", "36-45", "46-55", "56-65", "65+"],
        'income_brackets': ["<$30k", "$30k-$50k", "$50k-$75k", "$75k-$100k", "$100k-$150k", ">$150k"],
        'env_consciousness': ["Not concerned", "Slightly concerned", "Moderately concerned",
                             "Very concerned", "Extremely concerned"]
    }

# Use expanders for persona configuration
with st.expander(" Age Groups", expanded=False):
    age_groups_text = st.text_area(
        "Age Groups (one per line)",
        value="\n".join(st.session_state.persona_config['age_groups']),
        height=150,
        key="age_groups_input"
    )

with st.expander(" Income Brackets", expanded=False):
    income_brackets_text = st.text_area(
        "Income Brackets (one per line)",
        value="\n".join(st.session_state.persona_config['income_brackets']),
        height=150,
        key="income_brackets_input"
    )

with st.expander(" Environmental Consciousness", expanded=False):
    env_consciousness_text = st.text_area(
        "Environmental Consciousness Levels (one per line)",
        value="\n".join(st.session_state.persona_config['env_consciousness']),
        height=150,
        key="env_consciousness_input",
        help="These levels influence ground truth generation"
    )

# Parse persona inputs for experiment
age_groups = [line.strip() for line in age_groups_text.split('\n') if line.strip()]
income_brackets = [line.strip() for line in income_brackets_text.split('\n') if line.strip()]
env_consciousness = [line.strip() for line in env_consciousness_text.split('\n') if line.strip()]

# Show current configuration summary
st.markdown("**Current Configuration:**")
col1, col2, col3 = st.columns(3)

with col1:
    st.metric("Age Groups", len(age_groups))

with col2:
    st.metric("Income Brackets", len(income_brackets))

with col3:
    st.metric("Consciousness Levels", len(env_consciousness))

# ======================
# Section D: SSR Configuration
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
# Section E: Run Experiment
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

    **Persona Diversity:**
    - Age Groups: {len(age_groups)} categories
    - Income Brackets: {len(income_brackets)} categories
    - Environmental Consciousness: {len(env_consciousness)} levels

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

            # Build command
            cmd = [
                sys.executable,
                "ground_truth_pipeline.py"
            ]

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

                        # Action buttons
                        col1, col2 = st.columns(2)

                        with col1:
                            if st.button(" View Results", use_container_width=True, type="primary"):
                                st.session_state.selected_experiment = latest_experiment
                                st.switch_page("pages/3__View_Results.py")

                        with col2:
                            if st.button(" Run Another", use_container_width=True):
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
