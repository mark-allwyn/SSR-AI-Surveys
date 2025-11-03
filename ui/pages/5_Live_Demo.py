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
from src.survey import Question, LikertScale
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

# Custom CSS
st.markdown(f"""
<style>
    .stButton > button[kind="primary"] {{
        background-color: {brand_colors['teal_blue']} !important;
        border-color: {brand_colors['teal_blue']} !important;
        color: white !important;
    }}
    .stButton > button[kind="primary"]:hover {{
        background-color: {brand_colors['teal_dark']} !important;
    }}
</style>
""", unsafe_allow_html=True)

st.title("Live Demo - Test SSR on Your Own Text")

st.markdown("""
**See how SSR works in real-time.** Enter a text response to any survey question and watch SSR convert it
into a probability distribution over rating scales.
""")

# Check API key
api_key = os.getenv("OPENAI_API_KEY") or st.session_state.get('api_key')

if not api_key:
    error_message("OpenAI API key not configured. Please go to Settings to add your API key.")
    if st.button("Go to Settings"):
        st.switch_page("pages/6_Settings.py")
    st.stop()

st.markdown("---")

# ======================
# Section 1: Context (Optional)
# ======================
st.subheader("Step 1: Context / Stimulus (Optional)")

st.markdown("Provide background information if your question refers to a specific product, concept, or scenario.")

context_examples = {
    "None": "",
    "Product Example": "SmartHome Pro: An AI-powered home automation system that learns your preferences and automatically adjusts lighting, temperature, and security. Price: $299.",
    "Concept Example": "EcoBottle: A reusable water bottle made from 100% recycled ocean plastic. Keeps drinks cold for 24 hours. $34.99.",
}

context_example = st.selectbox("Load Example", options=list(context_examples.keys()))
context_text = st.text_area(
    "Context / Stimulus",
    value=context_examples[context_example],
    height=80,
    placeholder="Optional: Describe the product, concept, or scenario the question is about..."
)

st.markdown("---")

# ======================
# Section 2: Question Configuration
# ======================
st.subheader("Step 2: Configure Question & Scale")

col1, col2 = st.columns([2, 1])

with col1:
    question_text = st.text_input(
        "Question Text",
        value="How likely are you to recommend this product?",
        help="Enter the survey question"
    )

with col2:
    question_types = {
        "5-point Likert": "likert_5",
        "7-point Likert": "likert_7",
        "Yes/No": "yes_no"
    }
    selected_type = st.selectbox("Scale Type", options=list(question_types.keys()))
    question_type = question_types[selected_type]

# Define scale labels based on type
if question_type == "yes_no":
    scale_labels = {1: "No", 2: "Yes"}
elif question_type == "likert_5":
    scale_labels = {
        1: st.text_input("Rating 1", value="Strongly disagree", key="l5_1"),
        2: st.text_input("Rating 2", value="Disagree", key="l5_2"),
        3: st.text_input("Rating 3", value="Neutral", key="l5_3"),
        4: st.text_input("Rating 4", value="Agree", key="l5_4"),
        5: st.text_input("Rating 5", value="Strongly agree", key="l5_5")
    }
else:  # likert_7
    scale_labels = {
        1: st.text_input("Rating 1", value="Strongly disagree", key="l7_1"),
        2: st.text_input("Rating 2", value="Disagree", key="l7_2"),
        3: st.text_input("Rating 3", value="Somewhat disagree", key="l7_3"),
        4: st.text_input("Rating 4", value="Neutral", key="l7_4"),
        5: st.text_input("Rating 5", value="Somewhat agree", key="l7_5"),
        6: st.text_input("Rating 6", value="Agree", key="l7_6"),
        7: st.text_input("Rating 7", value="Strongly agree", key="l7_7")
    }

st.markdown("---")

# ======================
# Section 3: Text Response
# ======================
st.subheader("Step 3: Enter Text Response")

st.markdown("Type or select an example response to see how SSR analyzes it.")

# Example responses
examples = {
    "Custom (type your own)": "",
    "Strong Positive": "I absolutely love this! It exceeded all my expectations and I'd definitely recommend it!",
    "Moderate Positive": "This is pretty good. I'm quite satisfied with it and would likely recommend it.",
    "Neutral/Uncertain": "It's okay. Nothing special but not bad either. I'm neutral about it.",
    "Hedged Negative": "Not really what I expected. Somewhat disappointed, probably wouldn't recommend.",
    "Strong Negative": "Very disappointed with this. Definitely would not recommend to anyone.",
}

example_choice = st.selectbox("Select Example or Type Custom", options=list(examples.keys()))

text_response = st.text_area(
    "Text Response",
    value=examples[example_choice],
    height=100,
    placeholder="Type a text response to the question above..."
)

if text_response:
    st.caption(f"Length: {len(text_response)} characters")

st.markdown("---")

# ======================
# Section 4: Process Button
# ======================
if st.button("Analyze Response with SSR", type="primary", use_container_width=True, disabled=not text_response):
    with st.spinner("Computing semantic similarities..."):
        try:
            # Initialize SSR
            rater = SemanticSimilarityRater(
                model_name="text-embedding-3-small",
                temperature=1.0,
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
            else:
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

            # Create response object
            response = Response(
                respondent_id="demo",
                question_id="demo_question",
                text_response=text_response,
                respondent_profile={}
            )

            # Apply SSR
            distribution = rater.rate_response(response, question)

            success_message("Analysis complete!")

            # ======================
            # Results Display
            # ======================
            st.markdown("---")
            st.header("Results")

            # Key Metrics
            col1, col2, col3, col4 = st.columns(4)

            mode_label = scale_labels[distribution.mode]
            max_prob = distribution.distribution.max()

            with col1:
                st.metric("Most Likely Rating", f"{distribution.mode}: {mode_label}")
            with col2:
                st.metric("Confidence", f"{max_prob:.0%}")
            with col3:
                st.metric("Expected Value", f"{distribution.expected_value:.2f}")
            with col4:
                st.metric("Entropy (Uncertainty)", f"{distribution.entropy:.2f}")

            # Interpretation
            if max_prob > 0.7:
                st.success("**High Confidence:** The response has clear sentiment.")
            elif max_prob > 0.5:
                st.info("**Moderate Confidence:** Some ambiguity in the response.")
            else:
                st.warning("**High Uncertainty:** The response is ambiguous or hedged.")

            st.markdown("---")

            # Probability Distribution Chart
            st.subheader("Probability Distribution")

            labels_list = [scale_labels[i] for i in sorted(scale_labels.keys())]
            probabilities = distribution.distribution

            fig = go.Figure()

            colors = [brand_colors['cornflower_blue'] if i != distribution.mode - 1
                     else brand_colors['atomic_orange']
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
                xaxis_title="Rating",
                yaxis_title="Probability",
                yaxis_tickformat='.0%',
                showlegend=False,
                height=450,
                margin=dict(l=50, r=50, t=50, b=100),
                xaxis=dict(
                    tickangle=-45,
                    tickmode='linear'
                )
            )

            st.plotly_chart(fig, use_container_width=True)

            st.caption("Orange bar = most likely rating. All probabilities sum to 100%.")

            # What this means
            st.markdown("---")
            st.subheader("What This Means")

            st.markdown(f"""
            **SSR analyzed your text response** and computed how semantically similar it is to each rating label.

            - **Most likely rating:** {distribution.mode} ({mode_label}) with {max_prob:.0%} probability
            - **Expected value:** {distribution.expected_value:.2f} (weighted average considering all ratings)
            - **Entropy:** {distribution.entropy:.2f} (lower = more certain, higher = more spread out)

            Instead of forcing a single rating, SSR captures the full range of possibilities.
            This preserves nuance like hesitation, mixed feelings, and confidence levels.
            """)

            # Detailed probabilities table
            with st.expander("View Detailed Probabilities"):
                import pandas as pd
                prob_data = []
                for i, (point, label) in enumerate(sorted(scale_labels.items())):
                    prob = probabilities[i]
                    prob_data.append({
                        "Rating": point,
                        "Label": label,
                        "Probability": f"{prob:.4f}",
                        "Percentage": f"{prob:.1%}"
                    })
                prob_df = pd.DataFrame(prob_data)
                st.dataframe(prob_df, use_container_width=True, hide_index=True)

            # Raw similarities
            with st.expander("Advanced: Raw Similarity Scores"):
                st.markdown("""
                These are the raw cosine similarity scores before normalization.
                They show how semantically similar your text is to each rating label.
                """)

                similarities = distribution.similarities
                sim_data = []
                for i, (point, label) in enumerate(sorted(scale_labels.items())):
                    sim_data.append({
                        "Rating": point,
                        "Label": label,
                        "Similarity": f"{similarities[i]:.4f}"
                    })

                import pandas as pd
                sim_df = pd.DataFrame(sim_data)
                st.dataframe(sim_df, use_container_width=True, hide_index=True)

                st.caption("Higher similarity = more semantically similar to that label")

        except Exception as e:
            error_message(f"Error processing response: {str(e)}")
            with st.expander("Error Details"):
                import traceback
                st.code(traceback.format_exc())

# ======================
# Footer Tips
# ======================
st.markdown("---")
st.markdown("### Tips")

col1, col2 = st.columns(2)

with col1:
    st.markdown("""
    **Try Different Response Styles:**
    - Compare "Yes" vs "I think so"
    - See how "I love it!" differs from "It's pretty good"
    - Test hedged language like "maybe" or "I'm not sure"
    """)

with col2:
    st.markdown("""
    **Understanding Results:**
    - **High confidence (>70%)**: Clear, definitive response
    - **Moderate (50-70%)**: Some ambiguity
    - **Low (<50%)**: Highly uncertain or mixed
    """)
