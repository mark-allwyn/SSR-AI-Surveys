"""Survey Preview - See what respondents would experience."""

import streamlit as st
from pathlib import Path
import sys

# Add parent directory to path
sys.path.insert(0, str(Path(__file__).parent.parent.parent))

from ui.utils.data_loader import load_survey_config, get_available_surveys
import pandas as pd
from datetime import datetime

st.set_page_config(
    page_title="Survey Preview",
    page_icon="📋",
    layout="wide"
)

# Header
st.title("Survey Preview")
st.markdown("See what the survey experience looks like for respondents")
st.markdown("---")

# Survey selection
available_surveys = get_available_surveys()

if not available_surveys:
    st.error("No survey configurations found in the config/ directory")
    st.stop()

survey_options = {survey.stem: survey for survey in available_surveys}

selected_survey = st.selectbox(
    "Select Survey to Preview",
    options=list(survey_options.keys()),
    help="Choose which survey configuration to preview"
)

survey_config = load_survey_config(str(survey_options[selected_survey]))

if not survey_config or 'survey' not in survey_config:
    st.error("Could not load survey configuration")
    st.stop()

survey_data = survey_config['survey']

# Initialize session state for responses
if 'current_question_idx' not in st.session_state:
    st.session_state.current_question_idx = 0
if 'responses' not in st.session_state:
    st.session_state.responses = {}
if 'survey_started' not in st.session_state:
    st.session_state.survey_started = False
if 'current_survey' not in st.session_state:
    st.session_state.current_survey = selected_survey
if 'demographics_collected' not in st.session_state:
    st.session_state.demographics_collected = False
if 'user_demographics' not in st.session_state:
    st.session_state.user_demographics = {}

# Check if survey changed - reset state if it did
if st.session_state.current_survey != selected_survey:
    st.session_state.current_survey = selected_survey
    st.session_state.current_question_idx = 0
    st.session_state.responses = {}
    st.session_state.survey_started = False
    st.session_state.demographics_collected = False
    st.session_state.user_demographics = {}

# Demographics collection (before survey)
if not st.session_state.demographics_collected:
    st.header("Before You Begin")
    st.markdown("Please provide some basic information about yourself. This helps us analyze the survey results.")

    with st.form("demographics_form"):
        st.subheader("Your Information")

        col1, col2 = st.columns(2)

        with col1:
            respondent_id = st.text_input(
                "Respondent ID (optional)",
                placeholder="Leave blank for auto-generated ID",
                help="A unique identifier for you"
            )

            gender = st.selectbox(
                "Gender",
                options=["Male", "Female", "Non-binary", "Prefer not to say", "Other"]
            )

            age_group = st.selectbox(
                "Age Group",
                options=["18-24", "25-34", "35-44", "45-54", "55-64", "65+"]
            )

        with col2:
            occupation = st.selectbox(
                "Occupation",
                options=[
                    "Professional",
                    "Manager",
                    "Technical",
                    "Service",
                    "Sales",
                    "Administrative",
                    "Student",
                    "Retired",
                    "Creative",
                    "Other"
                ]
            )

            persona_group = st.text_input(
                "Persona Group (optional)",
                value="Survey Respondent",
                help="A category that describes you as a respondent"
            )

        submitted = st.form_submit_button("Continue to Survey", type="primary", use_container_width=True)

        if submitted:
            # Generate respondent ID if not provided
            if not respondent_id:
                timestamp = datetime.now().strftime("%Y%m%d%H%M%S")
                respondent_id = f"USER_{timestamp}"

            st.session_state.user_demographics = {
                'respondent_id': respondent_id,
                'gender': gender,
                'age_group': age_group,
                'occupation': occupation,
                'persona_group': persona_group
            }
            st.session_state.demographics_collected = True
            st.rerun()

# Survey intro (after demographics collected)
elif not st.session_state.survey_started:
    st.header(survey_data.get('name', 'Survey'))
    st.markdown(f"**{survey_data.get('description', '')}**")

    # Show intro information
    col1, col2, col3 = st.columns(3)

    questions = survey_data.get('questions', [])
    categories = survey_data.get('categories', [])

    with col1:
        st.metric("Total Questions", len(questions))

    with col2:
        if categories:
            st.metric("Categories", len(categories))
        else:
            st.metric("Question Types", len(set(q.get('type') for q in questions)))

    with col3:
        est_time = len(questions) * 0.5  # 30 seconds per question
        st.metric("Estimated Time", f"{int(est_time)} min")

    st.markdown("---")

    # Show categories if multi-category
    if categories:
        st.subheader("What You'll Be Evaluating")

        for category in categories:
            st.markdown(f"### {category.get('name')}")
            st.markdown(f"*{category.get('description')}*")

            if category.get('context'):
                with st.container():
                    st.info(category.get('context'))
    else:
        # Single category - show global context
        if survey_data.get('context'):
            st.subheader("Survey Context")
            with st.container():
                st.info(survey_data.get('context'))

    st.markdown("---")

    if st.button("Start Survey", type="primary", use_container_width=True):
        st.session_state.survey_started = True
        st.session_state.current_question_idx = 0
        st.session_state.responses = {}
        st.rerun()

# Survey questions (after starting)
else:
    questions = survey_data.get('questions', [])
    current_idx = st.session_state.current_question_idx

    # Progress bar
    progress = min((current_idx / len(questions)), 1.0)
    st.progress(progress)
    st.markdown(f"**Question {current_idx + 1} of {len(questions)}**")

    if current_idx < len(questions):
        question = questions[current_idx]

        # Show category-specific context if applicable
        categories = survey_data.get('categories', [])
        if categories:
            q_category = question.get('category')
            categories_compared = question.get('categories_compared', [])

            if categories_compared:
                # Comparative question - show both categories
                st.markdown("**Comparison Question**")

                cols = st.columns(2)
                for i, cat_id in enumerate(categories_compared):
                    category = next((c for c in categories if c.get('id') == cat_id), None)
                    if category:
                        with cols[i]:
                            st.markdown(f"**{category.get('name')}**")
                            st.info(category.get('context', ''))

            elif q_category:
                # Single category question
                category = next((c for c in categories if c.get('id') == q_category), None)
                if category:
                    st.markdown(f"**{category.get('name')}**")
                    st.info(category.get('context', ''))
        else:
            # Single category survey (no categories defined) - show global context
            global_context = survey_data.get('context')
            if global_context:
                st.info(global_context)

        # Question text
        st.subheader(question.get('text'))

        # Answer options based on question type
        q_type = question.get('type')
        q_id = question.get('id')

        # Create unique key for this question
        response_key = f"response_{current_idx}_{q_id}"

        if q_type in ['likert_5', 'likert_7', 'preference_scale']:
            scale = question.get('scale', {})
            options = [f"{k}. {v}" for k, v in sorted(scale.items())]

            response = st.radio(
                "Select your response:",
                options=options,
                key=response_key,
                label_visibility="collapsed"
            )

            if response:
                st.session_state.responses[q_id] = response

        elif q_type == 'yes_no':
            response = st.radio(
                "Select your response:",
                options=["No", "Yes"],
                key=response_key,
                label_visibility="collapsed",
                horizontal=True
            )

            if response:
                st.session_state.responses[q_id] = response

        elif q_type == 'multiple_choice':
            options = question.get('options', [])
            response = st.radio(
                "Select your response:",
                options=options,
                key=response_key,
                label_visibility="collapsed"
            )

            if response:
                st.session_state.responses[q_id] = response

        # Navigation buttons
        st.markdown("---")
        col1, col2, col3 = st.columns([1, 1, 1])

        with col1:
            if current_idx > 0:
                if st.button("Previous", use_container_width=True):
                    st.session_state.current_question_idx -= 1
                    st.rerun()

        with col2:
            if st.button("Restart Survey", use_container_width=True):
                st.session_state.survey_started = False
                st.session_state.current_question_idx = 0
                st.session_state.responses = {}
                st.rerun()

        with col3:
            # Check if current question is answered
            current_answered = q_id in st.session_state.responses

            if current_idx < len(questions) - 1:
                if st.button("Next", type="primary", disabled=not current_answered, use_container_width=True):
                    st.session_state.current_question_idx += 1
                    st.rerun()
            else:
                if st.button("Complete Survey", type="primary", disabled=not current_answered, use_container_width=True):
                    st.session_state.current_question_idx += 1
                    st.rerun()

    else:
        # Survey complete
        st.success("Survey Complete!")
        st.balloons()

        st.markdown("### Thank you for completing the survey!")
        st.markdown(f"You answered **{len(st.session_state.responses)}** out of **{len(questions)}** questions.")

        # Save responses to CSV
        if 'responses_saved' not in st.session_state:
            st.session_state.responses_saved = False

        if not st.session_state.responses_saved:
            try:
                # Create responses dataframe
                responses_data = []
                demographics = st.session_state.user_demographics

                for question in questions:
                    q_id = question.get('id')
                    response = st.session_state.responses.get(q_id)

                    if response:
                        # Extract numeric rating from response string (e.g., "1. Strongly disagree" -> 1)
                        try:
                            if isinstance(response, str) and '. ' in response:
                                rating_value = int(response.split('.')[0])
                            elif response in ["No", "Yes"]:
                                rating_value = 1 if response == "No" else 2
                            elif isinstance(response, (int, float)):
                                rating_value = int(response)
                            else:
                                # Try to extract number from string
                                rating_value = int(response)
                        except (ValueError, AttributeError) as e:
                            st.warning(f"Could not parse response for {q_id}: {response}")
                            continue

                        # Determine category
                        category = question.get('category', 'general')
                        if not category:
                            category = 'general'

                        responses_data.append({
                            'respondent_id': demographics['respondent_id'],
                            'question_id': q_id,
                            'ground_truth': rating_value,
                            'category': category,
                            'gender': demographics['gender'],
                            'age_group': demographics['age_group'],
                            'persona_group': demographics['persona_group'],
                            'occupation': demographics['occupation']
                        })

                if responses_data:
                    df = pd.DataFrame(responses_data)

                    # Create survey_responses directory if it doesn't exist
                    responses_dir = Path("survey_responses")
                    responses_dir.mkdir(exist_ok=True)

                    # Save with timestamp
                    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
                    filename = f"survey_responses/response_{demographics['respondent_id']}_{timestamp}.csv"
                    df.to_csv(filename, index=False)

                    st.session_state.responses_saved = True
                    st.success(f"Your responses have been saved to: `{filename}`")
                else:
                    st.error("No valid responses to save. Please check that you answered all questions.")
                    st.write("Debug - Responses in session state:", st.session_state.responses)
            except Exception as e:
                st.error(f"Error saving responses: {str(e)}")
                st.write("Debug info:")
                st.write("- Demographics:", st.session_state.user_demographics)
                st.write("- Responses:", st.session_state.responses)
                st.write("- Questions count:", len(questions))

        # Show responses
        with st.expander("View Your Responses"):
            for i, question in enumerate(questions, 1):
                q_id = question.get('id')
                response = st.session_state.responses.get(q_id, "Not answered")
                st.markdown(f"**{i}. {question.get('text')}**")
                st.markdown(f"*Response:* {response}")
                st.markdown("---")

        st.markdown("---")

        col1, col2 = st.columns(2)

        with col1:
            if st.button("Take Survey Again", use_container_width=True):
                st.session_state.survey_started = False
                st.session_state.current_question_idx = 0
                st.session_state.responses = {}
                st.session_state.demographics_collected = False
                st.session_state.user_demographics = {}
                st.session_state.responses_saved = False
                st.rerun()

        with col2:
            if st.button("Back to Home", type="primary", use_container_width=True):
                st.switch_page("1_Home.py")
