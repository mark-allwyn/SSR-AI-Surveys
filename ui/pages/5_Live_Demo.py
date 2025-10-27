"""Live Demo page - test SSR on individual responses interactively."""

import streamlit as st
from pathlib import Path
import sys
import os
import numpy as np
import plotly.graph_objects as go

# Add parent directory to path
sys.path.insert(0, str(Path(__file__).parent.parent.parent))

from src.ssr_model import SemanticSimilarityRater, RatingDistribution
from src.survey import Question
from src.llm_client import Response
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

st.set_page_config(page_title="Live Demo", page_icon="", layout="wide")

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
    /* Regular buttons (no type specified) */
    .stButton > button:not([kind]) {{
        background-color: white !important;
        border-color: {brand_colors['teal_blue']} !important;
        color: {brand_colors['teal_blue']} !important;
    }}
    .stButton > button:not([kind]):hover {{
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
    .stSlider [data-baseweb="slider"] {{
        background-color: transparent !important;
    }}
    .stSlider [data-baseweb="slider"] > div {{
        background-color: transparent !important;
    }}
    .stSlider [data-baseweb="slider"] > div > div {{
        background-color: {brand_colors['light_grey']} !important;
    }}
    .stSlider [data-baseweb="slider"] > div > div > div {{
        background-color: {brand_colors['teal_blue']} !important;
    }}
    .stSlider [data-baseweb="slider"] [role="slider"] {{
        background-color: {brand_colors['teal_blue']} !important;
        border: 2px solid white !important;
        box-shadow: 0 2px 4px rgba(0,0,0,0.2) !important;
    }}
    .stSlider [data-baseweb="slider"] [role="slider"]:hover {{
        background-color: {brand_colors['teal_dark']} !important;
    }}

    /* Text input */
    .stTextInput > div > div > input:focus {{
        border-color: {brand_colors['teal_blue']} !important;
        box-shadow: 0 0 0 0.2rem {brand_colors['teal_blue']}33 !important;
    }}

    /* Text area */
    .stTextArea > div > div > textarea:focus {{
        border-color: {brand_colors['teal_blue']} !important;
        box-shadow: 0 0 0 0.2rem {brand_colors['teal_blue']}33 !important;
    }}

    /* Selectbox */
    .stSelectbox > div > div > div:focus {{
        border-color: {brand_colors['teal_blue']} !important;
        box-shadow: 0 0 0 0.2rem {brand_colors['teal_blue']}33 !important;
    }}

    /* Download button */
    .stDownloadButton > button {{
        background-color: {brand_colors['teal_blue']} !important;
        border-color: {brand_colors['teal_blue']} !important;
        color: white !important;
    }}
</style>
""", unsafe_allow_html=True)

st.title(" Live Demo - Interactive SSR Playground")

st.markdown("""
Test Semantic Similarity Rating on individual text responses in real-time.
Enter any text and see how SSR converts it to a probability distribution.
""")

# Check API key
api_key = os.getenv("OPENAI_API_KEY") or st.session_state.get('api_key')

if not api_key:
    error_message(" OpenAI API key not configured. Please go to Settings to add your API key.")
    if st.button("Go to Settings"):
        st.switch_page("pages/5__Settings.py")
    st.stop()

# ======================
# Section A: Question Setup
# ======================
st.header(" Question Configuration")

# Question type selector
question_types = {
    "Yes/No (Binary)": "yes_no",
    "Likert 5-point": "likert_5",
    "Likert 7-point": "likert_7",
    "Custom Scale": "custom"
}

selected_type = st.selectbox(
    "Question Type",
    options=list(question_types.keys())
)

question_type = question_types[selected_type]

# Question text
question_text = st.text_input(
    "Question Text",
    value="How likely are you to recommend this product?",
    help="Enter the survey question"
)

# Scale configuration
st.subheader("Scale Labels")

if question_type == "yes_no":
    scale_labels = {1: "No", 2: "Yes"}
    st.info("Binary scale: No / Yes")

elif question_type == "likert_5":
    scale_labels = {
        1: st.text_input("Label 1", value="Strongly disagree"),
        2: st.text_input("Label 2", value="Disagree"),
        3: st.text_input("Label 3", value="Neutral"),
        4: st.text_input("Label 4", value="Agree"),
        5: st.text_input("Label 5", value="Strongly agree")
    }

elif question_type == "likert_7":
    scale_labels = {
        1: st.text_input("Label 1", value="Strongly disagree"),
        2: st.text_input("Label 2", value="Disagree"),
        3: st.text_input("Label 3", value="Somewhat disagree"),
        4: st.text_input("Label 4", value="Neutral"),
        5: st.text_input("Label 5", value="Somewhat agree"),
        6: st.text_input("Label 6", value="Agree"),
        7: st.text_input("Label 7", value="Strongly agree")
    }

else:  # custom
    num_options = st.slider("Number of Options", min_value=2, max_value=10, value=5)

    scale_labels = {}
    for i in range(1, num_options + 1):
        scale_labels[i] = st.text_input(f"Label {i}", value=f"Option {i}")

# Display scale preview
with st.expander(" Scale Preview"):
    for point, label in scale_labels.items():
        st.markdown(f"**{point}.** {label}")

st.markdown("---")

# ======================
# Section B: Text Input
# ======================
st.header(" Enter Response")

# Example responses
examples = {
    "Positive (enthusiastic)": "I absolutely love this! It's exactly what I've been looking for. Definitely would recommend!",
    "Positive (moderate)": "This is pretty good. I'm quite satisfied with it and would likely recommend it.",
    "Neutral": "It's okay. Nothing special but not bad either. I'm neutral about it.",
    "Negative (moderate)": "Not really what I expected. Somewhat disappointed, probably wouldn't recommend.",
    "Negative (strong)": "Very disappointed with this. Definitely would not recommend to anyone.",
    "Hedged/LLM-style": "While I appreciate certain aspects of this product, considering various factors and taking into account different perspectives, I would say that it seems moderately acceptable, though there are nuances to consider."
}

# Initialize session state for text response
if 'live_demo_text' not in st.session_state:
    st.session_state.live_demo_text = ""

col1, col2 = st.columns([3, 1])

with col2:
    st.markdown("**Example Responses:**")
    for example_name, example_text in examples.items():
        if st.button(example_name, use_container_width=True, key=f"example_{example_name}"):
            st.session_state.live_demo_text = example_text

with col1:
    text_response = st.text_area(
        "Text Response",
        value=st.session_state.live_demo_text,
        height=150,
        placeholder="Enter a textual response to the question...",
        help="Type any text response you want to test"
    )
    # Update session state when user types
    if text_response != st.session_state.live_demo_text:
        st.session_state.live_demo_text = text_response

# Character counter
if text_response:
    st.caption(f" {len(text_response)} characters")

st.markdown("---")

# ======================
# Section C: SSR Configuration
# ======================
st.header(" SSR Settings")

col1, col2 = st.columns(2)

with col1:
    temperature = st.slider(
        "Temperature",
        min_value=0.1,
        max_value=3.0,
        value=1.0,
        step=0.1,
        help="Controls distribution spread. Lower = more peaked, Higher = more spread"
    )

with col2:
    st.metric("Model", "text-embedding-3-small")
    st.caption("OpenAI embedding model")

# Temperature explanation
with st.expander(" What does temperature do?"):
    st.markdown("""
    **Temperature** controls how spread out the probability distribution is:

    - **Low (0.1-0.5):** More peaked distributions, confident predictions
    - **Default (1.0):** Balanced, as per paper
    - **High (2.0-3.0):** More spread distributions, uncertain predictions

    **Example:** If the response is "I strongly agree":
    - Low temp: `[0.0, 0.0, 0.0, 0.2, 0.8]` (very confident)
    - Default: `[0.0, 0.05, 0.15, 0.35, 0.45]` (balanced)
    - High temp: `[0.05, 0.15, 0.25, 0.30, 0.25]` (spread out)
    """)

st.markdown("---")

# ======================
# Section D: Process & Results
# ======================
st.header(" Apply SSR")

process_button = st.button(
    " Process Response",
    type="primary",
    use_container_width=True,
    disabled=not text_response
)

if process_button and text_response:
    with st.spinner("Computing semantic similarities..."):
        try:
            # Initialize SSR
            rater = SemanticSimilarityRater(
                model_name="text-embedding-3-small",
                temperature=temperature,
                normalize_method="paper",
                use_openai=True
            )

            # Create question object
            if question_type == "yes_no":
                question = Question(
                    id="demo_question",
                    text=question_text,
                    type=question_type
                )
            elif question_type in ["likert_5", "likert_7", "custom"]:
                from src.survey import LikertScale
                scale = LikertScale(
                    scale_type=question_type,
                    labels=scale_labels
                )
                question = Question(
                    id="demo_question",
                    text=question_text,
                    type=question_type,
                    scale=scale
                )
            else:
                raise ValueError(f"Unknown question type: {question_type}")

            # Create response object
            response = Response(
                respondent_id="demo",
                question_id="demo_question",
                text_response=text_response,
                respondent_profile={}
            )

            # Apply SSR
            distribution = rater.rate_response(response, question)

            success_message(" SSR processing complete!")

            st.markdown("---")

            # ======================
            # Results Display
            # ======================
            st.header(" Results")

            # Metrics
            col1, col2, col3, col4 = st.columns(4)

            with col1:
                mode_label = scale_labels[distribution.mode]
                st.metric("Predicted Rating (Mode)", f"{distribution.mode}: {mode_label}")

            with col2:
                st.metric("Expected Value", f"{distribution.expected_value:.2f}")

            with col3:
                max_prob = distribution.distribution.max()
                st.metric("Confidence", f"{max_prob:.1%}")

            with col4:
                st.metric("Entropy", f"{distribution.entropy:.2f}")

            # Probability Distribution Chart
            st.subheader("Probability Distribution")

            labels_list = [scale_labels[i] for i in sorted(scale_labels.keys())]
            probabilities = distribution.distribution

            fig = go.Figure()

            colors = [brand_colors['cornflower_blue'] if i != distribution.mode - 1 else brand_colors['atomic_orange']
                     for i in range(len(probabilities))]

            fig.add_trace(go.Bar(
                x=labels_list,
                y=probabilities,
                text=[f"{p:.1%}" for p in probabilities],
                textposition='outside',
                marker_color=colors,
                hovertemplate='<b>%{x}</b><br>Probability: %{y:.1%}<extra></extra>'
            ))

            fig.update_layout(
                title="Probability Distribution Over Scale Points",
                xaxis_title="Scale Label",
                yaxis_title="Probability",
                yaxis_tickformat='.0%',
                showlegend=False,
                height=500
            )

            st.plotly_chart(fig, use_container_width=True)

            # Detailed probability table
            with st.expander(" Detailed Probabilities"):
                prob_data = []
                for i, (point, label) in enumerate(sorted(scale_labels.items()), start=0):
                    prob = probabilities[i]
                    is_mode = (point == distribution.mode)

                    prob_data.append({
                        "Rating": point,
                        "Label": label,
                        "Probability": f"{prob:.4f}",
                        "Percentage": f"{prob:.2%}",
                        "Mode": "" if is_mode else ""
                    })

                import pandas as pd
                prob_df = pd.DataFrame(prob_data)
                st.dataframe(prob_df, use_container_width=True, hide_index=True)

            # Interpretation
            st.markdown("---")
            st.subheader(" Interpretation")

            max_prob = probabilities.max()

            if max_prob > 0.7:
                confidence_level = "very confident"
                confidence_color = brand_colors['electric_lime']
            elif max_prob > 0.5:
                confidence_level = "moderately confident"
                confidence_color = brand_colors['atomic_orange']
            else:
                confidence_level = "uncertain"
                confidence_color = brand_colors['atomic_orange']

            st.markdown(f"""
            The SSR model is **<span style="color:{confidence_color}">{confidence_level}</span>** in this prediction.

            - **Most likely rating:** {distribution.mode} ({mode_label})
            - **Confidence:** {max_prob:.1%}
            - **Expected value:** {distribution.expected_value:.2f} (weighted average)
            - **Entropy:** {distribution.entropy:.2f} (lower = more certain, higher = more uncertain)

            The model assigned probabilities across all {len(scale_labels)} scale points based on
            semantic similarity between your response and each scale label.
            """, unsafe_allow_html=True)

            # Raw similarities (advanced)
            with st.expander(" Advanced: Raw Similarity Scores"):
                st.markdown("""
                These are the raw cosine similarity scores **before** normalization.
                They show how semantically similar your response is to each scale label.
                """)

                # We need to recompute to show raw similarities
                # Get embeddings
                response_embedding = rater.get_embeddings([text_response])[0]
                label_embeddings = rater.get_embeddings(labels_list)

                # Compute similarities
                similarities = rater.compute_similarities(response_embedding, label_embeddings)

                sim_data = []
                for i, (point, label) in enumerate(sorted(scale_labels.items())):
                    sim_data.append({
                        "Rating": point,
                        "Label": label,
                        "Raw Similarity": f"{similarities[i]:.4f}"
                    })

                import pandas as pd
                sim_df = pd.DataFrame(sim_data)
                st.dataframe(sim_df, use_container_width=True, hide_index=True)

                st.markdown("""
                **Note:** Higher similarity = response text is more semantically similar to that label.
                SSR then normalizes these using the paper's method (subtract min, divide by sum).
                """)

        except Exception as e:
            error_message(f"Error processing response: {str(e)}")
            with st.expander("Error Details"):
                import traceback
                st.code(traceback.format_exc())

# ======================
# Section E: Batch Test
# ======================
st.markdown("---")
st.header(" Batch Test")

st.markdown("""
Test multiple responses at once. Enter one response per line.
""")

batch_input = st.text_area(
    "Multiple Responses (one per line)",
    height=200,
    placeholder="I love this!\nIt's okay I guess.\nNot for me.",
    help="Enter multiple text responses, one per line"
)

if st.button(" Process Batch", type="secondary"):
    if batch_input:
        responses = [line.strip() for line in batch_input.split('\n') if line.strip()]

        if len(responses) > 20:
            warning_message("Batch processing limited to 20 responses at a time")
            responses = responses[:20]

        with st.spinner(f"Processing {len(responses)} responses..."):
            try:
                # Initialize SSR
                rater = SemanticSimilarityRater(
                    model_name="text-embedding-3-small",
                    temperature=temperature,
                    normalize_method="paper",
                    use_openai=True
                )

                # Create question
                if question_type == "yes_no":
                    question = Question(
                        id="batch_demo",
                        text=question_text,
                        type=question_type
                    )
                elif question_type in ["likert_5", "likert_7", "custom"]:
                    from src.survey import LikertScale
                    scale = LikertScale(
                        scale_type=question_type,
                        labels=scale_labels
                    )
                    question = Question(
                        id="batch_demo",
                        text=question_text,
                        type=question_type,
                        scale=scale
                    )
                else:
                    raise ValueError(f"Unknown question type: {question_type}")

                # Process all responses
                batch_results = []
                for i, text in enumerate(responses):
                    response = Response(
                        respondent_id=f"batch_{i+1}",
                        question_id="batch_demo",
                        text_response=text,
                        respondent_profile={}
                    )

                    dist = rater.rate_response(response, question)

                    batch_results.append({
                        "Response": text[:50] + "..." if len(text) > 50 else text,
                        "Predicted": f"{dist.mode}: {scale_labels[dist.mode]}",
                        "Expected": f"{dist.expected_value:.2f}",
                        "Confidence": f"{dist.distribution.max():.1%}",
                        "Entropy": f"{dist.entropy:.2f}"
                    })

                # Display results
                import pandas as pd
                results_df = pd.DataFrame(batch_results)
                st.dataframe(results_df, use_container_width=True, hide_index=True)

                # Download button
                csv = results_df.to_csv(index=False)
                st.download_button(
                    label=" Download Batch Results",
                    data=csv,
                    file_name="batch_ssr_results.csv",
                    mime="text/csv"
                )

            except Exception as e:
                error_message(f"Error in batch processing: {str(e)}")
    else:
        warning_message("Please enter at least one response")

# ======================
# Tips
# ======================
st.markdown("---")
st.markdown("###  Tips for Using Live Demo")

with st.expander("How to interpret results"):
    st.markdown("""
    **Probability Distribution:**
    - Shows how confident SSR is about each rating option
    - Higher bars = more likely
    - The highlighted bar (orange) is the mode (most likely rating)

    **Expected Value:**
    - Weighted average of all ratings
    - Takes into account probabilities across all options
    - More robust than just the mode

    **Confidence:**
    - Maximum probability assigned to any single option
    - >70%: Very confident
    - 50-70%: Moderately confident
    - <50%: Uncertain (probability spread across options)

    **Entropy:**
    - Measure of uncertainty in the distribution
    - Lower values (close to 0): Very certain, peaked distribution
    - Higher values: More uncertain, spread distribution
    """)

with st.expander("Best practices"):
    st.markdown("""
    **For accurate results:**
    - Use clear, well-defined scale labels
    - Test with responses that vary in sentiment
    - Try different temperatures to see effects
    - Compare hedged vs direct language

    **Example comparisons to try:**
    - "Yes" vs "I would say yes"
    - "Strongly agree" vs "I think I somewhat agree"
    - See how hedging affects the distribution
    """)
