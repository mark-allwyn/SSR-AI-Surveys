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
    parse_text_report,
    load_distributions
)
from ui.components.metrics_cards import warning_message, error_message
from ui.utils.metrics_calculator import calculate_radar_metrics

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
# Section C2: Multi-Dimensional Performance Radar Chart
# ======================
st.header("Multi-Dimensional Performance")

# Load distribution data for advanced metrics
distributions_data = load_distributions(selected_exp_path)

# Calculate radar metrics
radar_metrics = calculate_radar_metrics(overall_llm, question_metrics, distributions_data)

# Create radar chart
categories = list(radar_metrics.keys())
values = list(radar_metrics.values())

# Close the radar chart by adding the first value at the end
categories_closed = categories + [categories[0]]
values_closed = values + [values[0]]

fig_radar = go.Figure()

# Add the LLM+SSR performance trace
fig_radar.add_trace(go.Scatterpolar(
    r=values_closed,
    theta=categories_closed,
    fill='toself',
    fillcolor=brand_colors['teal_blue'],
    opacity=0.4,
    line=dict(color=brand_colors['teal_blue'], width=2),
    name='LLM+SSR',
    hovertemplate='%{theta}: %{r:.1f}<extra></extra>'
))

# Optional: Add a benchmark line at 80%
benchmark_values = [80] * len(categories_closed)
fig_radar.add_trace(go.Scatterpolar(
    r=benchmark_values,
    theta=categories_closed,
    line=dict(color=brand_colors['light_grey'], width=1, dash='dash'),
    name='Target (80%)',
    hovertemplate='Target: %{r:.1f}<extra></extra>'
))

fig_radar.update_layout(
    polar=dict(
        radialaxis=dict(
            visible=True,
            range=[0, 100],
            tickfont=dict(size=12),
            gridcolor=brand_colors['light_grey']
        ),
        angularaxis=dict(
            tickfont=dict(size=13, family='Arial Black', color=brand_colors['teal_dark'])
        )
    ),
    showlegend=True,
    legend=dict(
        orientation="h",
        yanchor="bottom",
        y=-0.2,
        xanchor="center",
        x=0.5
    ),
    height=550,
    margin=dict(l=80, r=80, t=60, b=80)
)

st.plotly_chart(fig_radar, use_container_width=True)

# Add explanatory text
with st.expander("What do these dimensions mean?"):
    st.markdown(f"""
    **Accuracy** ({radar_metrics['Accuracy']:.1f}%): Overall percentage of correct predictions

    **Precision** ({radar_metrics['Precision']:.1f}%): How close predictions are to actual values (based on MAE)

    **Consistency** ({radar_metrics['Consistency']:.1f}%): How stable performance is across different questions

    **Confidence** ({radar_metrics['Confidence']:.1f}%): How certain the model is in its predictions (based on entropy)

    **Coverage** ({radar_metrics['Coverage']:.1f}%): Percentage of data points with predictions

    **Calibration** ({radar_metrics['Calibration']:.1f}%): How well predicted probabilities match actual outcomes
    """)

st.markdown("---")

# ======================
# Section D: Accuracy by Question
# ======================
st.header("Accuracy by Question")

# Prepare data for bar chart
human_accuracies = [question_metrics[q]['human_accuracy'] for q in question_ids]
llm_accuracies = [question_metrics[q]['llm_accuracy'] for q in question_ids]
llm_mae_values = [question_metrics[q]['llm_mae'] for q in question_ids]

# Convert MAE to "accuracy-like" score (inverted, higher is better)
# Assuming 5-point scale: MAE of 0 = 100%, MAE of 5 = 0%
max_rating = 5
llm_mae_performance = [(1 - mae/max_rating) * 100 for mae in llm_mae_values]

# Split into multiple charts if more than 10 questions
n_questions = len(question_ids)
questions_per_chart = 10

if n_questions <= questions_per_chart:
    # Single chart for 10 or fewer questions
    fig_accuracy = go.Figure()

    fig_accuracy.add_trace(go.Bar(
        name='Ground Truth (Mode)',
        x=question_ids,
        y=human_accuracies,
        marker_color=brand_colors['teal_blue'],
        text=[f"{v:.1f}%" for v in human_accuracies],
        textposition='outside',
        textfont=dict(size=11, color=brand_colors['teal_blue'], family='Arial Black'),
        hovertemplate='Ground Truth: %{y:.1f}%<extra></extra>'
    ))

    fig_accuracy.add_trace(go.Bar(
        name='LLM Mode Accuracy',
        x=question_ids,
        y=llm_accuracies,
        marker_color=brand_colors['atomic_orange'],
        text=[f"{v:.1f}%" for v in llm_accuracies],
        textposition='outside',
        textfont=dict(size=11, color=brand_colors['atomic_orange'], family='Arial Black'),
        hovertemplate='Mode Accuracy: %{y:.1f}%<extra></extra>'
    ))

    fig_accuracy.add_trace(go.Bar(
        name='LLM Expected Value (MAE-based)',
        x=question_ids,
        y=llm_mae_performance,
        marker_color=brand_colors['turquoise'],
        text=[f"{perf:.1f}%" for perf in llm_mae_performance],
        textposition='outside',
        textfont=dict(size=11, color=brand_colors['turquoise'], family='Arial Black'),
        hovertemplate='MAE Performance: %{y:.1f}%<br>Raw MAE: %{customdata:.2f}<extra></extra>',
        customdata=llm_mae_values
    ))

    fig_accuracy.update_layout(
        title=dict(
            text='Performance Comparison by Question<br><sub>Mode Accuracy vs Expected Value (MAE-based)</sub>',
            font=dict(size=20, family='Arial Black')
        ),
        xaxis_title='Question',
        yaxis_title='Performance (Higher = Better)',
        barmode='group',
        height=550,
        yaxis=dict(range=[0, 115]),
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

    st.caption("**Note:** MAE-based performance = (1 - MAE/5) × 100%. Shows how well the expected value prediction performs (considering full probability distribution).")
else:
    # Multiple charts for more than 10 questions
    n_charts = (n_questions + questions_per_chart - 1) // questions_per_chart

    for chart_idx in range(n_charts):
        start_idx = chart_idx * questions_per_chart
        end_idx = min(start_idx + questions_per_chart, n_questions)

        # Get data for this chart
        chart_question_ids = question_ids[start_idx:end_idx]
        chart_human_accuracies = human_accuracies[start_idx:end_idx]
        chart_llm_accuracies = llm_accuracies[start_idx:end_idx]

        # Create figure for this chunk
        fig_accuracy = go.Figure()

        fig_accuracy.add_trace(go.Bar(
            name='Ground Truth',
            x=chart_question_ids,
            y=chart_human_accuracies,
            marker_color=brand_colors['teal_blue'],
            text=[f"{v:.1f}%" for v in chart_human_accuracies],
            textposition='outside',
            textfont=dict(size=12, color=brand_colors['teal_blue'], family='Arial Black')
        ))

        fig_accuracy.add_trace(go.Bar(
            name='LLM+SSR',
            x=chart_question_ids,
            y=chart_llm_accuracies,
            marker_color=brand_colors['atomic_orange'],
            text=[f"{v:.1f}%" for v in chart_llm_accuracies],
            textposition='outside',
            textfont=dict(size=12, color=brand_colors['atomic_orange'], family='Arial Black')
        ))

        fig_accuracy.update_layout(
            title=dict(
                text=f'Mode Accuracy by Question (Q{start_idx+1}-Q{end_idx})',
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

        # Add spacing between charts (except after the last one)
        if chart_idx < n_charts - 1:
            st.markdown("<br>", unsafe_allow_html=True)

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

    # Primary metrics in top row
    col1, col2, col3, col4 = st.columns(4)

    with col1:
        st.metric("Accuracy", f"{q_metrics['llm_accuracy']:.1f}%",
                  help="Percentage of correct predictions (mode accuracy)")

    with col2:
        st.metric("Top-2 Accuracy", f"{q_metrics.get('llm_top2_accuracy', 0):.1f}%",
                  help="Percentage where true value is in top 2 predictions")

    with col3:
        st.metric("MAE", f"{q_metrics['llm_mae']:.3f}",
                  help="Mean Absolute Error - average distance from true value")

    with col4:
        st.metric("RMSE", f"{q_metrics.get('llm_rmse', 0):.3f}",
                  help="Root Mean Square Error - penalizes larger errors more")

    # Advanced metrics in second row
    col5, col6 = st.columns(2)

    with col5:
        prob_at_truth = q_metrics.get('llm_prob_at_truth', 0)
        st.metric("Prob at Truth", f"{prob_at_truth:.3f}",
                  help="Average probability assigned to the true rating")

    with col6:
        kl_div = q_metrics.get('llm_kl_divergence', 0)
        st.metric("KL Divergence", f"{kl_div:.4f}",
                  help="How different predicted distribution is from ground truth (lower is better)")

    # Add distribution curves if available
    if distributions_data and question_id in distributions_data:
        with st.expander(" View Probability Distributions", expanded=False):
            question_dists = distributions_data[question_id]

            # Aggregate distributions for this question
            from ui.utils.metrics_calculator import aggregate_distribution_stats
            dist_stats = aggregate_distribution_stats(distributions_data, question_id)

            if dist_stats and len(dist_stats['mean_probabilities']) > 0:
                # Create distribution curve
                n_ratings = len(dist_stats['mean_probabilities'])
                rating_labels = [str(r+1) for r in range(n_ratings)]

                fig_dist = go.Figure()

                # Add mean probability distribution
                fig_dist.add_trace(go.Scatter(
                    x=rating_labels,
                    y=dist_stats['mean_probabilities'],
                    mode='lines+markers',
                    name='Mean Probability',
                    line=dict(color=brand_colors['teal_blue'], width=3),
                    marker=dict(size=10, color=brand_colors['teal_blue']),
                    fill='tozeroy',
                    fillcolor=f"rgba(54, 117, 136, 0.2)",
                    hovertemplate='Rating: %{x}<br>Probability: %{y:.3f}<extra></extra>'
                ))

                # Add ground truth distribution if available
                if dist_stats['ground_truth_distribution']:
                    fig_dist.add_trace(go.Bar(
                        x=rating_labels,
                        y=dist_stats['ground_truth_distribution'],
                        name='Ground Truth',
                        marker_color=brand_colors['atomic_orange'],
                        opacity=0.5,
                        hovertemplate='Rating: %{x}<br>Proportion: %{y:.3f}<extra></extra>'
                    ))

                # Color-code by confidence level
                mean_entropy = dist_stats['mean_entropy']
                if mean_entropy < 0.5:
                    confidence_level = "High"
                    confidence_color = brand_colors['electric_lime']
                elif mean_entropy < 1.0:
                    confidence_level = "Medium"
                    confidence_color = brand_colors['cornflower_blue']
                else:
                    confidence_level = "Low"
                    confidence_color = brand_colors['atomic_orange']

                fig_dist.update_layout(
                    title=dict(
                        text=f"Probability Distribution (Confidence: {confidence_level}, Entropy: {mean_entropy:.2f})",
                        font=dict(size=16, color=confidence_color)
                    ),
                    xaxis_title='Rating',
                    yaxis_title='Probability / Proportion',
                    height=350,
                    yaxis=dict(range=[0, max(max(dist_stats['mean_probabilities']), max(dist_stats['ground_truth_distribution']) if dist_stats['ground_truth_distribution'] else 0) * 1.1]),
                    showlegend=True,
                    legend=dict(
                        orientation="h",
                        yanchor="bottom",
                        y=1.02,
                        xanchor="right",
                        x=1
                    ),
                    hovermode='x unified'
                )

                st.plotly_chart(fig_dist, use_container_width=True)

                # Show sample size
                st.caption(f"Based on {dist_stats['n_samples']} respondent predictions")

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
        'Accuracy': f"{q['llm_accuracy']:.1f}%",
        'Top-2': f"{q.get('llm_top2_accuracy', 0):.1f}%",
        'MAE': f"{q['llm_mae']:.3f}",
        'RMSE': f"{q.get('llm_rmse', 0):.3f}",
        'Prob@Truth': f"{q.get('llm_prob_at_truth', 0):.3f}",
        'KL Div': f"{q.get('llm_kl_divergence', 0):.4f}"
    })

# Calculate averages for additional metrics
llm_top2 = [question_metrics[q].get('llm_top2_accuracy', 0) for q in question_ids]
llm_rmse = [question_metrics[q].get('llm_rmse', 0) for q in question_ids]
llm_prob = [question_metrics[q].get('llm_prob_at_truth', 0) for q in question_ids]
llm_kl = [question_metrics[q].get('llm_kl_divergence', 0) for q in question_ids]

# Add average row
summary_data.append({
    'Question': 'AVERAGE',
    'Accuracy': f"{overall_llm:.1f}%",
    'Top-2': f"{np.mean(llm_top2):.1f}%",
    'MAE': f"{np.mean(llm_mae):.3f}",
    'RMSE': f"{np.mean(llm_rmse):.3f}",
    'Prob@Truth': f"{np.mean(llm_prob):.3f}",
    'KL Div': f"{np.mean(llm_kl):.4f}"
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
# Section H: Download Reports
# ======================
st.markdown("---")
st.header("Download Reports")

col1, col2, col3 = st.columns(3)

with col1:
    # Download text report
    txt_file = selected_exp_path / 'report.txt'
    if txt_file.exists():
        with open(txt_file, 'r') as f:
            txt_content = f.read()

        st.download_button(
            label=" Download Text Report",
            data=txt_content,
            file_name=f"report_{exp_info['timestamp']}.txt",
            mime="text/plain",
            use_container_width=True
        )
    else:
        st.warning("Text report not available")

with col2:
    # Download markdown report
    md_file = selected_exp_path / 'report.md'
    if md_file.exists():
        with open(md_file, 'r') as f:
            md_content = f.read()

        st.download_button(
            label=" Download Markdown Report",
            data=md_content,
            file_name=f"report_{exp_info['timestamp']}.md",
            mime="text/markdown",
            use_container_width=True
        )
    else:
        st.warning("Markdown report not available")

with col3:
    # Download ground truth CSV
    csv_file = selected_exp_path / 'ground_truth.csv'
    if csv_file.exists():
        gt_df = pd.read_csv(csv_file)
        csv_content = gt_df.to_csv(index=False)

        st.download_button(
            label=" Download Ground Truth CSV",
            data=csv_content,
            file_name=f"ground_truth_{exp_info['timestamp']}.csv",
            mime="text/csv",
            use_container_width=True
        )
    else:
        st.warning("Ground truth CSV not available")
