"""Settings page for API configuration and experiment management."""

import streamlit as st
from pathlib import Path
import sys
import os
from dotenv import load_dotenv, set_key

# Add parent directory to path
sys.path.insert(0, str(Path(__file__).parent.parent.parent))

from ui.utils.data_loader import get_all_experiments, get_experiment_info, delete_experiment
from ui.components.metrics_cards import success_message, warning_message, error_message

st.set_page_config(page_title="Settings", page_icon="", layout="wide")

st.title(" Settings & Configuration")

# Load environment variables
load_dotenv()

# Tabs for different settings
tab1, tab2, tab3, tab4 = st.tabs([" API Configuration", " Persona Defaults", "🗂 Experiment Management", " About"])

# ======================
# Tab 1: API Configuration
# ======================
with tab1:
    st.header("API Configuration")

    st.markdown("""
    The SSR pipeline requires an OpenAI API key to use the `text-embedding-3-small` model
    for semantic similarity computations.
    """)

    # Check current API key status
    current_key = os.getenv("OPENAI_API_KEY") or st.session_state.get('api_key')

    if current_key:
        st.success(" API Key is configured")
        masked_key = current_key[:8] + "..." + current_key[-4:] if len(current_key) > 12 else "***"
        st.code(f"Current Key: {masked_key}")
    else:
        warning_message("No API key configured. Please enter your OpenAI API key below.")

    st.markdown("---")

    # API Key input
    col1, col2 = st.columns([3, 1])

    with col1:
        api_key_input = st.text_input(
            "OpenAI API Key",
            value="",
            type="password",
            placeholder="sk-...",
            help="Get your API key from https://platform.openai.com/api-keys"
        )

    with col2:
        save_to_env = st.checkbox(
            "Save to .env",
            value=True,
            help="Save the API key to .env file for persistence"
        )

    if st.button(" Save API Key", type="primary"):
        if api_key_input.startswith("sk-"):
            # Save to session state
            st.session_state.api_key = api_key_input

            # Save to .env if requested
            if save_to_env:
                env_path = Path(".env")
                if not env_path.exists():
                    env_path.touch()

                set_key(str(env_path), "OPENAI_API_KEY", api_key_input)
                success_message("API key saved to session and .env file")
            else:
                success_message("API key saved to session (this session only)")

            st.rerun()
        else:
            error_message("Invalid API key format. OpenAI keys start with 'sk-'")

    # Test API key
    if current_key:
        st.markdown("---")
        st.subheader("Test API Connection")

        if st.button(" Test Connection"):
            with st.spinner("Testing API connection..."):
                try:
                    from openai import OpenAI
                    client = OpenAI(api_key=current_key)

                    # Test with a simple embedding request
                    response = client.embeddings.create(
                        model="text-embedding-3-small",
                        input="test"
                    )

                    if response.data:
                        success_message(" API connection successful! Embedding model is accessible.")
                except Exception as e:
                    error_message(f"API connection failed: {str(e)}")

    # Clear API key
    if current_key:
        st.markdown("---")
        if st.button(" Clear API Key", type="secondary"):
            st.session_state.api_key = None
            warning_message("API key cleared from session. Reload page to use .env key if available.")
            st.rerun()

# ======================
# Tab 2: Persona Defaults
# ======================
with tab2:
    st.header("Persona Configuration")

    st.markdown("""
    Define persona descriptions for synthetic respondents.
    Enter each persona as a natural language description, one per line.
    These will be used to generate survey responses just like human surveys.
    """)

    st.markdown("---")

    # Initialize persona descriptions if not present
    if 'persona_descriptions' not in st.session_state:
        st.session_state.persona_descriptions = [
            "A 35-year-old tech entrepreneur in San Francisco. Values innovation and efficiency. Early adopter of new technology. High income, environmentally conscious.",
            "A 68-year-old retired teacher living in rural Iowa. Fixed income, cautious about change. Prefers traditional methods. Not very tech-savvy.",
            "A 28-year-old graduate student in environmental science. Very passionate about climate change. Low income but highly educated. Socially progressive.",
            "A 45-year-old small business owner in suburban Texas. Moderate income, family-oriented. Pragmatic about environmental issues. Politically independent.",
            "A 52-year-old nurse in an urban hospital. Middle income, works long hours. Concerned about healthcare costs. Values work-life balance."
        ]

    # Persona descriptions input
    st.subheader("Persona Descriptions")
    st.markdown("Enter persona descriptions, one per line. The system will randomly select from these when generating respondents.")

    persona_text = st.text_area(
        "Persona Descriptions (one per line)",
        value="\n".join(st.session_state.persona_descriptions),
        height=300,
        help="Each line is one persona description. Be as detailed or brief as you like.",
        placeholder="Example:\nA 35-year-old tech entrepreneur. High income, environmentally conscious.\nA retired teacher on fixed income. Cautious about change."
    )

    # Show current count
    current_personas = [line.strip() for line in persona_text.split('\n') if line.strip()]
    st.info(f"Current persona pool: {len(current_personas)} persona(s)")

    st.markdown("---")

    # Save button
    if st.button(" Save Persona Configuration", type="primary"):
        personas = [line.strip() for line in persona_text.split('\n') if line.strip()]

        if not personas:
            error_message("You must provide at least one persona description")
        else:
            st.session_state.persona_descriptions = personas
            success_message(f"Saved {len(personas)} persona description(s) for this session")
            st.rerun()

    # Reset to defaults
    if st.button(" Reset to Default Personas"):
        st.session_state.persona_descriptions = [
            "A 35-year-old tech entrepreneur in San Francisco. Values innovation and efficiency. Early adopter of new technology. High income, environmentally conscious.",
            "A 68-year-old retired teacher living in rural Iowa. Fixed income, cautious about change. Prefers traditional methods. Not very tech-savvy.",
            "A 28-year-old graduate student in environmental science. Very passionate about climate change. Low income but highly educated. Socially progressive.",
            "A 45-year-old small business owner in suburban Texas. Moderate income, family-oriented. Pragmatic about environmental issues. Politically independent.",
            "A 52-year-old nurse in an urban hospital. Middle income, works long hours. Concerned about healthcare costs. Values work-life balance."
        ]
        success_message("Reset to default persona descriptions")
        st.rerun()

    st.markdown("---")

    # Tips
    with st.expander("Tips for Writing Personas"):
        st.markdown("""
        **Good persona descriptions include:**
        - Age and location
        - Occupation and income level
        - Values and priorities
        - Attitudes relevant to your survey topic
        - Lifestyle or behavioral traits

        **Examples:**

        **For a healthcare survey:**
        - "A 30-year-old software engineer with employer health insurance. Generally healthy, rarely visits doctors. Prefers telemedicine."
        - "A 65-year-old retiree with multiple chronic conditions. On Medicare. Visits specialists regularly. Concerned about medication costs."

        **For a product survey:**
        - "A budget-conscious college student. Shops at discount stores. Values price over brand. Active on social media."
        - "A wealthy executive who values quality and convenience. Willing to pay premium for best products. Shops online frequently."

        **Keep it natural - write them as you would describe real people in your target audience.**
        """)

# ======================
# Tab 3: Experiment Management
# ======================
with tab3:
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

        import pandas as pd
        df = pd.DataFrame(experiment_data)
        st.dataframe(df, use_container_width=True, hide_index=True)

        st.markdown("---")

        # Delete experiments
        st.subheader("Delete Experiments")
        warning_message(" Deletion is permanent and cannot be undone!")

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
                if st.button(" Delete Experiment", type="primary", disabled=not confirm):
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
                value=5,
                help="Keeps the N most recent experiments and deletes all older ones"
            )

            experiments_to_delete = experiments[num_to_keep:]

            if experiments_to_delete:
                st.warning(f"This will delete {len(experiments_to_delete)} experiment(s)")

                col1, col2 = st.columns(2)

                with col1:
                    bulk_confirm = st.checkbox("I confirm bulk deletion")

                with col2:
                    if st.button(" Delete Old Experiments", type="primary", disabled=not bulk_confirm):
                        deleted_count = 0
                        for exp in experiments_to_delete:
                            if delete_experiment(exp):
                                deleted_count += 1

                        success_message(f"Deleted {deleted_count} experiment(s)")
                        st.rerun()

# ======================
# Tab 4: About
# ======================
with tab4:
    st.header("About SSR Pipeline")

    st.markdown("""
    ## Semantic Similarity Rating

    This implementation follows the methodology from the paper:

    **"LLMs Reproduce Human Purchase Intent via Semantic Similarity Elicitation"**

     [arXiv:2510.08338v2](https://arxiv.org/abs/2510.08338v2)

    ### Features

    - ✅ Paper-exact implementation
    - ✅ OpenAI text-embedding-3-small model
    - ✅ Paper's normalization method (subtract min + proportional)
    - ✅ Multiple question types supported
    - ✅ Ground truth evaluation with 7+ metrics
    - ✅ Human vs LLM response comparison
    - ✅ Customizable persona parameters
    - ✅ Automated reporting (PNG, TXT, MD)

    ### Technology Stack

    - **Backend**: Python 3.10+
    - **UI**: Streamlit
    - **Embeddings**: OpenAI API
    - **Analysis**: pandas, numpy, scikit-learn
    - **Visualization**: matplotlib, seaborn, plotly

    ### Version Information

    - **UI Version**: 1.0.0 (MVP)
    - **Pipeline Version**: 1.0.0

    ### Repository

    🔗 [GitHub Repository](https://github.com/mark-allwyn/SSR-AI-Surveys)

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

    ---

    Made with  using Streamlit
    """)
