"""Interactive Dashboard - browser-based visualizations of experiment results."""

import streamlit as st
from pathlib import Path
import sys
import pandas as pd
import plotly.graph_objects as go
import plotly.express as px
from plotly.subplots import make_subplots
import numpy as np

# Add parent directory to path
sys.path.insert(0, str(Path(__file__).parent.parent.parent))

from ui.utils.data_loader import (
    get_all_experiments,
    get_experiment_info,
    load_ground_truth,
    load_text_report,
    parse_text_report
)
from ui.components.metrics_cards import warning_message, error_message

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

st.set_page_config(page_title="Results Dashboard", page_icon="", layout="wide")

# Custom CSS for brand colors on buttons
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

st.title("Results Dashboard")
st.markdown("Interactive visualizations and detailed metrics from your experiments")

# ======================
# Section A: Experiment Selector
# ======================
experiments = get_all_experiments()

if not experiments:
    warning_message("No experiments found. Run an experiment first!")
    if st.button("Go to Run Experiment"):
        st.switch_page("pages/2_Run_Experiment.py")
    st.stop()

# Create experiment options
experiment_options = {}
for exp in experiments:
    info = get_experiment_info(exp)
    experiment_options[f"{info['timestamp']} ({info['n_responses']} responses)"] = exp

# Check if we have a selected experiment from session state
default_index = 0
if 'selected_experiment' in st.session_state and st.session_state.selected_experiment:
    for i, exp in enumerate(experiments):
        if exp == st.session_state.selected_experiment:
            default_index = i
            break

selected_exp_label = st.selectbox(
    "Select Experiment",
    options=list(experiment_options.keys()),
    index=default_index
)

selected_exp_path = experiment_options[selected_exp_label]
exp_info = get_experiment_info(selected_exp_path)

st.markdown("---")

# ======================
# Section B: Load Data
# ======================
text_report = load_text_report(selected_exp_path)
metrics = parse_text_report(text_report) if text_report else {}

if not metrics:
    error_message("Could not load metrics. Please ensure the experiment completed successfully.")
    st.stop()

# Extract metrics
overall_human = metrics.get('overall_human_accuracy', 0)
overall_llm = metrics.get('overall_llm_accuracy', 0)
question_metrics = {k: v for k, v in metrics.items() if k not in ['overall_human_accuracy', 'overall_llm_accuracy']}
question_ids = list(question_metrics.keys())

# ======================
# Section C: Overall Performance
# ======================
st.header("LLM+SSR Overall Performance")

# Calculate additional metrics
human_mae = [question_metrics[q]['human_mae'] for q in question_ids]
llm_mae = [question_metrics[q]['llm_mae'] for q in question_ids]
avg_llm_mae = np.mean(llm_mae)
gap = overall_human - overall_llm

col1, col2, col3 = st.columns(3)

with col1:
    st.metric("Mode Accuracy", f"{overall_llm:.1f}%",
              help="Percentage of responses where LLM+SSR correctly predicted the most likely rating")

with col2:
    st.metric("Average MAE", f"{avg_llm_mae:.3f}",
              help="Mean Absolute Error - average distance between predicted and actual ratings")

with col3:
    st.metric("Gap from Ground Truth", f"{gap:.1f}%",
              delta=f"{-gap:.1f}%",
              delta_color="inverse",
              help="Difference from perfect ground truth accuracy (lower is better)")

st.markdown("---")

# ======================
# Section D: Accuracy by Question
# ======================
st.header("Accuracy by Question")

# Prepare data for bar chart
human_accuracies = [question_metrics[q]['human_accuracy'] for q in question_ids]
llm_accuracies = [question_metrics[q]['llm_accuracy'] for q in question_ids]

fig_accuracy = go.Figure()

fig_accuracy.add_trace(go.Bar(
    name='Ground Truth',
    x=question_ids,
    y=human_accuracies,
    marker_color=brand_colors['teal_blue'],
    text=[f"{v:.1f}%" for v in human_accuracies],
    textposition='outside',
    textfont=dict(size=12, color=brand_colors['teal_blue'], family='Arial Black')
))

fig_accuracy.add_trace(go.Bar(
    name='LLM+SSR',
    x=question_ids,
    y=llm_accuracies,
    marker_color=brand_colors['atomic_orange'],
    text=[f"{v:.1f}%" for v in llm_accuracies],
    textposition='outside',
    textfont=dict(size=12, color=brand_colors['atomic_orange'], family='Arial Black')
))

fig_accuracy.update_layout(
    title=dict(
        text='Mode Accuracy by Question',
        font=dict(size=20, family='Arial Black')
    ),
    xaxis_title='Question',
    yaxis_title='Accuracy (%)',
    barmode='group',
    height=500,
    yaxis=dict(range=[0, 110]),
    legend=dict(
        orientation="h",
        yanchor="bottom",
        y=1.02,
        xanchor="right",
        x=1
    ),
    hovermode='x unified'
)

st.plotly_chart(fig_accuracy, use_container_width=True)

st.markdown("---")

# ======================
# Section E: Confusion Matrix Heatmaps
# ======================
st.header("Confusion Matrix Heatmaps (LLM+SSR)")

st.markdown("**Note:** These heatmaps show how well LLM+SSR predictions match ground truth ratings. Rows represent true ratings, columns represent predicted ratings.")

# Load confusion matrices
import json
cm_file = selected_exp_path / 'confusion_matrices.json'

if cm_file.exists():
    with open(cm_file, 'r') as f:
        confusion_matrices = json.load(f)

    # Display confusion matrices in a grid
    n_questions = len(question_ids)
    cols_per_row = 2
    n_rows = (n_questions + cols_per_row - 1) // cols_per_row

    for row_idx in range(n_rows):
        cols = st.columns(cols_per_row)

        for col_idx in range(cols_per_row):
            q_idx = row_idx * cols_per_row + col_idx

            if q_idx < n_questions:
                question_id = question_ids[q_idx]
                cm = np.array(confusion_matrices[question_id])

                with cols[col_idx]:
                    # Normalize by row for better visualization
                    cm_normalized = cm / (cm.sum(axis=1, keepdims=True) + 1e-10)

                    # Create custom colorscale using brand colors
                    custom_colorscale = [
                        [0, brand_colors['light_grey']],
                        [0.5, brand_colors['cornflower_blue']],
                        [1, brand_colors['teal_dark']]
                    ]

                    # Create heatmap
                    fig = go.Figure(data=go.Heatmap(
                        z=cm_normalized,
                        x=[str(i+1) for i in range(cm.shape[1])],
                        y=[str(i+1) for i in range(cm.shape[0])],
                        colorscale=custom_colorscale,
                        text=cm,  # Show actual counts
                        texttemplate='%{text}',
                        textfont={"size": 12},
                        hovertemplate='True: %{y}<br>Predicted: %{x}<br>Count: %{text}<br>Proportion: %{z:.2f}<extra></extra>',
                        colorbar=dict(title="Proportion")
                    ))

                    fig.update_layout(
                        title=dict(
                            text=f"Q{q_idx+1}: {question_id}",
                            font=dict(size=14)
                        ),
                        xaxis_title="Predicted",
                        yaxis_title="True",
                        height=350,
                        margin=dict(l=60, r=60, t=60, b=60)
                    )

                    st.plotly_chart(fig, use_container_width=True)

else:
    st.warning("Confusion matrices not available. Run a new experiment to generate them.")

st.markdown("---")

# ======================
# Section F: Question-Level Details
# ======================
st.header("Question-Level Performance")

# Display metrics for each question in a clean layout
for i, question_id in enumerate(question_ids):
    q_metrics = question_metrics[question_id]

    st.subheader(f"Q{i+1}: {question_id}")

    col1, col2 = st.columns(2)

    with col1:
        st.metric("LLM+SSR Accuracy", f"{q_metrics['llm_accuracy']:.1f}%",
                  help="Percentage of correct predictions for this question")

    with col2:
        st.metric("LLM+SSR MAE", f"{q_metrics['llm_mae']:.3f}",
                  help="Mean Absolute Error for this question")

    if i < len(question_ids) - 1:
        st.markdown("---")

st.markdown("---")

# ======================
# Section G: Summary Table
# ======================
st.header("Summary Table")

# Create summary dataframe
summary_data = []
for q_id in question_ids:
    q = question_metrics[q_id]
    summary_data.append({
        'Question': q_id,
        'Ground Truth': f"{q['human_accuracy']:.1f}%",
        'LLM+SSR Accuracy': f"{q['llm_accuracy']:.1f}%",
        'GT MAE': f"{q['human_mae']:.3f}",
        'LLM MAE': f"{q['llm_mae']:.3f}",
        'Gap': f"{q['human_accuracy'] - q['llm_accuracy']:.1f}%"
    })

# Add average row
summary_data.append({
    'Question': 'AVERAGE',
    'Ground Truth': f"{overall_human:.1f}%",
    'LLM+SSR Accuracy': f"{overall_llm:.1f}%",
    'GT MAE': f"{np.mean(human_mae):.3f}",
    'LLM MAE': f"{np.mean(llm_mae):.3f}",
    'Gap': f"{overall_human - overall_llm:.1f}%"
})

summary_df = pd.DataFrame(summary_data)

# Style the dataframe
def highlight_winner(row):
    if row['Question'] == 'AVERAGE':
        return [f'background-color: {brand_colors["teal_blue"]}; color: white; font-weight: bold'] * len(row)
    else:
        return [''] * len(row)

styled_df = summary_df.style.apply(highlight_winner, axis=1)

st.dataframe(styled_df, use_container_width=True, hide_index=True)

# ======================
# Section H: Actions
# ======================
st.markdown("---")
st.header("Actions")

col1, col2 = st.columns(2)

with col1:
    if st.button("Run New Experiment", use_container_width=True, type="primary"):
        st.switch_page("pages/2_Run_Experiment.py")

with col2:
    if st.button("Live Demo", use_container_width=True):
        st.switch_page("pages/4_Live_Demo.py")
