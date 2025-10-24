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
    load_distributions,
    group_experiments_by_survey,
    calculate_experiment_metrics
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
# Section C3: Timeline View (Multiple Runs Detection)
# ======================
# Check if there are multiple runs of the same survey
all_experiments = get_all_experiments()
experiment_groups = group_experiments_by_survey(all_experiments)

# Find which group the current experiment belongs to
current_group = None
for fingerprint, group_data in experiment_groups.items():
    if selected_exp_path in group_data['experiments']:
        current_group = group_data
        break

# Only show timeline if there are multiple runs of the same survey
if current_group and len(current_group['experiments']) > 1:
    st.header("Performance Timeline")
    st.markdown(f"**{len(current_group['experiments'])} runs** detected for this survey configuration")

    # Calculate metrics for all experiments in this group
    timeline_data = []
    for exp_path in current_group['experiments']:
        exp_metrics = calculate_experiment_metrics(exp_path)
        if exp_metrics:
            timeline_data.append(exp_metrics)

    # Sort by timestamp
    timeline_data.sort(key=lambda x: x['timestamp'])

    if len(timeline_data) >= 2:
        # Extract data for plotting
        timestamps = [d['timestamp'].strftime('%Y-%m-%d %H:%M') for d in timeline_data]
        accuracies = [d['accuracy'] for d in timeline_data]
        maes = [d['mae'] for d in timeline_data]

        # Create timeline chart for accuracy
        fig_timeline = go.Figure()

        fig_timeline.add_trace(go.Scatter(
            x=timestamps,
            y=accuracies,
            mode='lines+markers',
            name='Accuracy',
            line=dict(color=brand_colors['teal_blue'], width=3),
            marker=dict(size=10, color=brand_colors['teal_blue']),
            hovertemplate='%{x}<br>Accuracy: %{y:.1f}%<extra></extra>'
        ))

        # Add trend line
        from scipy import stats
        x_numeric = list(range(len(accuracies)))
        slope, intercept, r_value, p_value, std_err = stats.linregress(x_numeric, accuracies)
        trend_line = [slope * x + intercept for x in x_numeric]

        fig_timeline.add_trace(go.Scatter(
            x=timestamps,
            y=trend_line,
            mode='lines',
            name='Trend',
            line=dict(color=brand_colors['atomic_orange'], width=2, dash='dash'),
            hovertemplate='Trend: %{y:.1f}%<extra></extra>'
        ))

        fig_timeline.update_layout(
            title=dict(
                text='Accuracy Trend Over Time',
                font=dict(size=20, family='Arial Black')
            ),
            xaxis_title='Experiment Run',
            yaxis_title='Accuracy (%)',
            height=400,
            yaxis=dict(range=[0, 110]),
            hovermode='x unified',
            showlegend=True,
            legend=dict(
                orientation="h",
                yanchor="bottom",
                y=1.02,
                xanchor="right",
                x=1
            )
        )

        st.plotly_chart(fig_timeline, use_container_width=True)

        # Statistical analysis
        col1, col2, col3 = st.columns(3)

        with col1:
            first_accuracy = accuracies[0]
            last_accuracy = accuracies[-1]
            change = last_accuracy - first_accuracy
            st.metric(
                "Overall Change",
                f"{change:+.1f}%",
                delta=f"{change:.1f}%",
                help="Change from first to most recent run"
            )

        with col2:
            avg_accuracy = np.mean(accuracies)
            st.metric(
                "Average Accuracy",
                f"{avg_accuracy:.1f}%",
                help="Mean accuracy across all runs"
            )

        with col3:
            std_accuracy = np.std(accuracies)
            st.metric(
                "Std Deviation",
                f"{std_accuracy:.1f}%",
                help="Consistency across runs (lower is more consistent)"
            )

        # Interpretation
        if slope > 0.5:
            st.success(f"Improving trend detected (R² = {r_value**2:.3f})")
        elif slope < -0.5:
            st.warning(f"Declining trend detected (R² = {r_value**2:.3f})")
        else:
            st.info(f"Stable performance (R² = {r_value**2:.3f})")

    st.markdown("---")

# ======================
# Section D: Accuracy by Question
# ======================
st.header("Accuracy by Question")

# Prepare data for bar chart
human_accuracies = [question_metrics[q]['human_accuracy'] for q in question_ids]
llm_accuracies = [question_metrics[q]['llm_accuracy'] for q in question_ids]

# Split into multiple charts if more than 10 questions
n_questions = len(question_ids)
questions_per_chart = 10

if n_questions <= questions_per_chart:
    # Single chart for 10 or fewer questions
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

    col1, col2 = st.columns(2)

    with col1:
        st.metric("LLM+SSR Accuracy", f"{q_metrics['llm_accuracy']:.1f}%",
                  help="Percentage of correct predictions for this question")

    with col2:
        st.metric("LLM+SSR MAE", f"{q_metrics['llm_mae']:.3f}",
                  help="Mean Absolute Error for this question")

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
