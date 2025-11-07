"""Compare Experiments - Side-by-side comparison of multiple experiment runs."""

import streamlit as st
from pathlib import Path
import sys
import pandas as pd
import plotly.graph_objects as go
import numpy as np
from scipy import stats

# Add parent directory to path
sys.path.insert(0, str(Path(__file__).parent.parent.parent))

from ui.utils.data_loader import (
    get_all_experiments,
    get_experiment_info,
    calculate_experiment_metrics,
    parse_text_report,
    load_text_report,
    load_distributions,
    group_experiments_by_survey
)
from ui.components.metrics_cards import warning_message, error_message, info_message
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

st.set_page_config(page_title="Compare Experiments", page_icon="", layout="wide")

# Custom CSS for brand colors
st.markdown(f"""
<style>
    .stButton > button[kind="primary"] {{
        background-color: {brand_colors['teal_blue']} !important;
        border-color: {brand_colors['teal_blue']} !important;
        color: white !important;
    }}
    .stButton > button[kind="primary"]:hover {{
        background-color: {brand_colors['teal_dark']} !important;
        border-color: {brand_colors['teal_dark']} !important;
    }}
    .stMultiSelect > div > div > div {{
        border-color: {brand_colors['teal_blue']} !important;
    }}
</style>
""", unsafe_allow_html=True)

st.title("Compare Experiments")
st.markdown("Select multiple experiments to compare their performance side-by-side")

# ======================
# Section A: Experiment Selection
# ======================
experiments = get_all_experiments()

if not experiments:
    warning_message("No experiments found. Run an experiment first!")
    if st.button("Go to Run Experiment"):
        st.switch_page("pages/2_Run_Experiment.py")
    st.stop()

# Group experiments by survey
experiment_groups = group_experiments_by_survey(experiments)

# Let user choose between comparing all experiments or filtering by survey
comparison_mode = st.radio(
    "Comparison Mode",
    ["All Experiments", "Same Survey Only"],
    horizontal=True,
    help="Choose whether to compare any experiments or only those with the same survey questions"
)

# Create experiment options
experiment_options = {}
for exp in experiments:
    info = get_experiment_info(exp)
    experiment_options[f"{info['timestamp']} ({info['n_responses']} responses)"] = exp

if comparison_mode == "Same Survey Only":
    # Let user select which survey group
    survey_group_labels = []
    survey_group_map = {}

    for fingerprint, group_data in experiment_groups.items():
        if len(group_data['experiments']) > 1:
            first_exp = group_data['experiments'][0]
            first_info = get_experiment_info(first_exp)
            label = f"{len(group_data['experiments'])} runs of survey with {len(group_data['question_ids'])} questions"
            survey_group_labels.append(label)
            survey_group_map[label] = group_data['experiments']

    if not survey_group_labels:
        warning_message("No survey groups with multiple runs found. Showing all experiments instead.")
        comparison_mode = "All Experiments"
    else:
        selected_group_label = st.selectbox(
            "Select Survey Group",
            options=survey_group_labels
        )
        available_experiments = survey_group_map[selected_group_label]

        # Filter experiment_options to only include experiments from this group
        experiment_options = {k: v for k, v in experiment_options.items() if v in available_experiments}

# Multi-select for experiments
selected_exp_labels = st.multiselect(
    "Select Experiments to Compare (2-5 recommended)",
    options=list(experiment_options.keys()),
    help="Select at least 2 experiments for comparison"
)

if len(selected_exp_labels) < 2:
    info_message("Please select at least 2 experiments to compare.")
    st.stop()

if len(selected_exp_labels) > 6:
    warning_message("Comparing more than 6 experiments may make visualizations crowded. Consider selecting fewer experiments.")

selected_exp_paths = [experiment_options[label] for label in selected_exp_labels]

st.markdown("---")

# ======================
# Section B: Load and Process Data
# ======================
comparison_data = []
color_palette = [
    brand_colors['teal_blue'],
    brand_colors['atomic_orange'],
    brand_colors['cornflower_blue'],
    brand_colors['turquoise'],
    brand_colors['electric_lime'],
    brand_colors['teal_dark']
]

for idx, exp_path in enumerate(selected_exp_paths):
    exp_metrics = calculate_experiment_metrics(exp_path)
    if exp_metrics:
        exp_metrics['color'] = color_palette[idx % len(color_palette)]
        exp_metrics['label'] = selected_exp_labels[idx]
        comparison_data.append(exp_metrics)

if not comparison_data:
    error_message("Could not load metrics for selected experiments.")
    st.stop()

# Sort by timestamp
comparison_data.sort(key=lambda x: x['timestamp'])

# ======================
# Section C: Summary Metrics Comparison
# ======================
st.header("Summary Metrics")

# Create comparison table
metrics_table = []
for data in comparison_data:
    metrics_table.append({
        'Experiment': data['label'].split('(')[0].strip(),
        'Accuracy': f"{data['accuracy']:.1f}%",
        'MAE': f"{data['mae']:.3f}",
        'Respondents': data['n_respondents'],
        'Questions': data['n_questions']
    })

df = pd.DataFrame(metrics_table)
st.dataframe(df, use_container_width=True, hide_index=True)

# Statistical comparison
if len(comparison_data) >= 2:
    accuracies = [d['accuracy'] for d in comparison_data]

    col1, col2, col3 = st.columns(3)

    with col1:
        best_idx = np.argmax(accuracies)
        st.metric(
            "Best Performance",
            comparison_data[best_idx]['label'].split('(')[0].strip(),
            f"{accuracies[best_idx]:.1f}%"
        )

    with col2:
        improvement = accuracies[-1] - accuracies[0]
        st.metric(
            "Change (First → Last)",
            f"{improvement:+.1f}%",
            delta=f"{improvement:.1f}%"
        )

    with col3:
        std_dev = np.std(accuracies)
        st.metric(
            "Variability (Std Dev)",
            f"{std_dev:.1f}%",
            help="Lower values indicate more consistent performance"
        )

st.markdown("---")

# ======================
# Section D: Accuracy Comparison Chart
# ======================
st.header("Accuracy Comparison")

fig_comparison = go.Figure()

timestamps = [d['timestamp'].strftime('%Y-%m-%d\n%H:%M') for d in comparison_data]
accuracies = [d['accuracy'] for d in comparison_data]
maes = [d['mae'] for d in comparison_data]

for idx, data in enumerate(comparison_data):
    fig_comparison.add_trace(go.Scatter(
        x=[timestamps[idx]],
        y=[data['accuracy']],
        mode='markers',
        name=data['label'].split('(')[0].strip(),
        marker=dict(size=15, color=data['color']),
        hovertemplate=f"{data['label']}<br>Accuracy: {data['accuracy']:.1f}%<br>MAE: {data['mae']:.3f}<extra></extra>"
    ))

# Add trend line if chronological
if len(comparison_data) >= 2:
    x_numeric = list(range(len(accuracies)))
    slope, intercept, r_value, p_value, std_err = stats.linregress(x_numeric, accuracies)
    trend_line = [slope * x + intercept for x in x_numeric]

    fig_comparison.add_trace(go.Scatter(
        x=timestamps,
        y=trend_line,
        mode='lines',
        name='Trend',
        line=dict(color=brand_colors['light_grey'], width=2, dash='dash'),
        hovertemplate='Trend: %{y:.1f}%<extra></extra>'
    ))

fig_comparison.update_layout(
    title=dict(
        text='Accuracy Over Time',
        font=dict(size=20, family='Arial Black')
    ),
    xaxis_title='Experiment Date/Time',
    yaxis_title='Accuracy (%)',
    height=450,
    yaxis=dict(range=[min(accuracies) - 5, max(accuracies) + 5]),
    showlegend=True,
    legend=dict(
        orientation="v",
        yanchor="top",
        y=1,
        xanchor="left",
        x=1.02
    ),
    hovermode='closest'
)

st.plotly_chart(fig_comparison, use_container_width=True)

st.markdown("---")

# ======================
# Section E: Radar Chart Comparison
# ======================
st.header("Multi-Dimensional Comparison")

fig_radar_comparison = go.Figure()

# Use comparison_data which is sorted by timestamp
for data in comparison_data:
    exp_path = data['path']

    # Load data for this experiment
    text_report = load_text_report(exp_path)
    if text_report:
        metrics = parse_text_report(text_report)
        overall_llm = metrics.get('overall_llm_accuracy', 0)
        question_metrics = {k: v for k, v in metrics.items()
                          if k not in ['overall_human_accuracy', 'overall_llm_accuracy']}

        distributions_data = load_distributions(exp_path)

        # Calculate radar metrics
        radar_metrics = calculate_radar_metrics(overall_llm, question_metrics, distributions_data)

        categories = list(radar_metrics.keys())
        values = list(radar_metrics.values())

        # Close the radar
        categories_closed = categories + [categories[0]]
        values_closed = values + [values[0]]

        # Add trace
        fig_radar_comparison.add_trace(go.Scatterpolar(
            r=values_closed,
            theta=categories_closed,
            name=data['label'].split('(')[0].strip(),
            line=dict(color=data['color'], width=2),
            fillcolor=data['color'],
            opacity=0.3,
            fill='toself'
        ))

fig_radar_comparison.update_layout(
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
        orientation="v",
        yanchor="top",
        y=1,
        xanchor="left",
        x=1.02
    ),
    height=550,
    margin=dict(l=80, r=150, t=60, b=80)
)

st.plotly_chart(fig_radar_comparison, use_container_width=True)

st.markdown("---")

# ======================
# Section F: Statistical Analysis
# ======================
st.header("Statistical Analysis")

if len(comparison_data) >= 2:
    col1, col2 = st.columns(2)

    with col1:
        st.subheader("Accuracy Statistics")

        # Perform paired t-test if we have enough data
        if len(comparison_data) >= 3:
            # Compare first vs last
            first_acc = comparison_data[0]['accuracy']
            last_acc = comparison_data[-1]['accuracy']

            st.markdown(f"""
            **First Experiment:** {first_acc:.1f}%
            **Last Experiment:** {last_acc:.1f}%
            **Change:** {last_acc - first_acc:+.1f}%
            """)

            # Trend analysis
            x_numeric = list(range(len(accuracies)))
            slope, intercept, r_value, p_value, std_err = stats.linregress(x_numeric, accuracies)

            if p_value < 0.05:
                if slope > 0:
                    st.success(f"✓ Statistically significant improvement detected (p = {p_value:.4f}, R² = {r_value**2:.3f})")
                else:
                    st.error(f"✗ Statistically significant decline detected (p = {p_value:.4f}, R² = {r_value**2:.3f})")
            else:
                st.info(f"No statistically significant trend (p = {p_value:.4f}, R² = {r_value**2:.3f})")
        else:
            st.info("Select at least 3 experiments for trend analysis")

    with col2:
        st.subheader("MAE Statistics")

        maes = [d['mae'] for d in comparison_data]
        avg_mae = np.mean(maes)
        min_mae = min(maes)
        max_mae = max(maes)

        st.markdown(f"""
        **Average MAE:** {avg_mae:.3f}
        **Best (Lowest) MAE:** {min_mae:.3f}
        **Worst (Highest) MAE:** {max_mae:.3f}
        **Range:** {max_mae - min_mae:.3f}
        """)

        best_mae_idx = np.argmin(maes)
        st.success(f"Best precision: {comparison_data[best_mae_idx]['label'].split('(')[0].strip()}")

st.markdown("---")

# ======================
# Section G: Recommendations
# ======================
st.header("Recommendations")

if len(comparison_data) >= 2:
    accuracies = [d['accuracy'] for d in comparison_data]
    maes = [d['mae'] for d in comparison_data]

    # Generate recommendations
    recommendations = []

    # Check for improvement trend
    if len(comparison_data) >= 3:
        x_numeric = list(range(len(accuracies)))
        slope, _, r_value, p_value, _ = stats.linregress(x_numeric, accuracies)

        if p_value < 0.05 and slope > 0:
            recommendations.append("✓ Your experiments show consistent improvement over time. Keep using your current methodology!")
        elif p_value < 0.05 and slope < 0:
            recommendations.append("⚠ Performance is declining. Consider reviewing recent changes to your experimental setup.")

    # Check variability
    std_dev = np.std(accuracies)
    if std_dev > 5:
        recommendations.append(f"⚠ High variability detected (σ = {std_dev:.1f}%). Consider increasing sample size or standardizing methodology.")
    else:
        recommendations.append(f"✓ Low variability (σ = {std_dev:.1f}%) indicates consistent experimental conditions.")

    # Check best performer
    best_idx = np.argmax(accuracies)
    best_exp = comparison_data[best_idx]
    recommendations.append(f"💡 Best performance: {best_exp['label'].split('(')[0].strip()} with {best_exp['accuracy']:.1f}% accuracy. Review this experiment's configuration for optimal settings.")

    # Sample size recommendation
    sample_sizes = [d['n_respondents'] for d in comparison_data]
    if max(sample_sizes) < 50:
        recommendations.append("💡 Consider increasing sample size (50+ respondents) for more reliable results.")

    for rec in recommendations:
        st.markdown(f"- {rec}")
else:
    st.info("Select more experiments for detailed recommendations.")
