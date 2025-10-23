"""View Results page - explore experiment results in detail."""

import streamlit as st
from pathlib import Path
import sys
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
from PIL import Image

# Add parent directory to path
sys.path.insert(0, str(Path(__file__).parent.parent.parent))

from ui.utils.data_loader import (
    get_all_experiments,
    get_experiment_info,
    load_ground_truth,
    load_text_report,
    load_markdown_report,
    parse_text_report
)
from ui.components.metrics_cards import comparison_metrics, warning_message, error_message

st.set_page_config(page_title="View Results", page_icon="", layout="wide")

st.title(" View Results")

# ======================
# Section A: Experiment Selector
# ======================
experiments = get_all_experiments()

if not experiments:
    warning_message("No experiments found. Run an experiment first!")
    if st.button("Go to Run Experiment"):
        st.switch_page("pages/2__Run_Experiment.py")
    st.stop()

# Create experiment options
experiment_options = {}
for exp in experiments:
    info = get_experiment_info(exp)
    experiment_options[f"{info['timestamp']} ({info['n_responses']} responses)"] = exp

# Check if we have a selected experiment from session state
default_index = 0
if 'selected_experiment' in st.session_state and st.session_state.selected_experiment:
    # Find index of selected experiment
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

# Display experiment info
col1, col2, col3, col4 = st.columns(4)

with col1:
    st.metric("Timestamp", exp_info['timestamp'])

with col2:
    st.metric("Respondents", exp_info['n_respondents'])

with col3:
    st.metric("Questions", exp_info['n_questions'])

with col4:
    st.metric("Total Responses", exp_info['n_responses'])

st.markdown("---")

# ======================
# Section B: Load Data
# ======================

# Load text report for metrics
text_report = load_text_report(selected_exp_path)
metrics = parse_text_report(text_report) if text_report else {}

# ======================
# Section C: Overview Metrics
# ======================
st.header("📈 Overall Performance")

if metrics:
    # Get overall metrics
    human_acc = metrics.get('overall_human_accuracy', 0)
    llm_acc = metrics.get('overall_llm_accuracy', 0)

    if human_acc and llm_acc:
        comparison_metrics(human_acc, llm_acc, "Mode Accuracy", ".1f")

        # Calculate winner
        diff = human_acc - llm_acc
        if abs(diff) < 0.1:
            winner_text = " **Result:** Tie (difference < 0.1%)"
        elif diff > 0:
            winner_text = f" **Winner:** Human (+{diff:.1f}%)"
        else:
            winner_text = f" **Winner:** LLM (+{abs(diff):.1f}%)"

        st.markdown(winner_text)
else:
    warning_message("Could not parse metrics from report")

st.markdown("---")

# ======================
# Section D: Visual Report
# ======================
st.header(" Visual Report")

report_png = selected_exp_path / "report.png"
if report_png.exists():
    try:
        image = Image.open(report_png)
        st.image(image, use_container_width=True)

        # Download button
        with open(report_png, "rb") as file:
            st.download_button(
                label=" Download Report Image",
                data=file,
                file_name=f"report_{exp_info['folder']}.png",
                mime="image/png"
            )
    except Exception as e:
        error_message(f"Error loading report image: {str(e)}")
else:
    warning_message("Report image not found")

st.markdown("---")

# ======================
# Section E: Question-by-Question Analysis
# ======================
st.header(" Question-by-Question Analysis")

# Extract question metrics from parsed report
question_metrics = {k: v for k, v in metrics.items() if k not in ['overall_human_accuracy', 'overall_llm_accuracy']}

if question_metrics:
    # Create tabs for each question
    question_ids = list(question_metrics.keys())

    tabs = st.tabs([f"Q{i+1}: {qid}" for i, qid in enumerate(question_ids)])

    for i, (question_id, tab) in enumerate(zip(question_ids, tabs)):
        with tab:
            q_metrics = question_metrics[question_id]

            st.subheader(f"{question_id}")

            # Metrics comparison
            col1, col2, col3, col4 = st.columns(4)

            with col1:
                h_acc = q_metrics.get('human_accuracy', 0)
                st.metric("Human Accuracy", f"{h_acc:.1f}%")

            with col2:
                l_acc = q_metrics.get('llm_accuracy', 0)
                st.metric("LLM Accuracy", f"{l_acc:.1f}%")

            with col3:
                h_mae = q_metrics.get('human_mae', 0)
                st.metric("Human MAE", f"{h_mae:.3f}")

            with col4:
                l_mae = q_metrics.get('llm_mae', 0)
                st.metric("LLM MAE", f"{l_mae:.3f}")

            # Comparison chart
            comparison_data = pd.DataFrame({
                'Response Style': ['Human', 'LLM'],
                'Accuracy': [h_acc, l_acc],
                'MAE': [h_mae, l_mae]
            })

            fig = go.Figure()

            fig.add_trace(go.Bar(
                name='Accuracy (%)',
                x=comparison_data['Response Style'],
                y=comparison_data['Accuracy'],
                text=comparison_data['Accuracy'].round(1),
                textposition='auto',
                marker_color=['#1f77b4', '#ff7f0e']
            ))

            fig.update_layout(
                title=f"{question_id} - Accuracy Comparison",
                yaxis_title="Accuracy (%)",
                showlegend=False,
                height=400
            )

            st.plotly_chart(fig, use_container_width=True)

else:
    warning_message("No question-level metrics available")

st.markdown("---")

# ======================
# Section F: Ground Truth Data
# ======================
st.header(" Ground Truth Data")

ground_truth_df = load_ground_truth(selected_exp_path)

if ground_truth_df is not None:
    st.markdown(f"**Total Entries:** {len(ground_truth_df)}")

    # Filters
    col1, col2 = st.columns(2)

    with col1:
        # Filter by question
        questions = ['All'] + sorted(ground_truth_df['question_id'].unique().tolist())
        selected_question = st.selectbox("Filter by Question", questions)

    with col2:
        # Filter by respondent
        respondents = ['All'] + sorted(ground_truth_df['respondent_id'].unique().tolist())
        selected_respondent = st.selectbox("Filter by Respondent", respondents)

    # Apply filters
    filtered_df = ground_truth_df.copy()

    if selected_question != 'All':
        filtered_df = filtered_df[filtered_df['question_id'] == selected_question]

    if selected_respondent != 'All':
        filtered_df = filtered_df[filtered_df['respondent_id'] == selected_respondent]

    # Display filtered data
    st.dataframe(
        filtered_df,
        use_container_width=True,
        hide_index=True
    )

    # Download button
    csv = filtered_df.to_csv(index=False)
    st.download_button(
        label=" Download Ground Truth CSV",
        data=csv,
        file_name=f"ground_truth_{exp_info['folder']}.csv",
        mime="text/csv"
    )

    # Statistics
    st.markdown("###  Statistics")

    col1, col2, col3 = st.columns(3)

    with col1:
        st.metric("Mean Rating", f"{filtered_df['ground_truth'].mean():.2f}")

    with col2:
        st.metric("Std Deviation", f"{filtered_df['ground_truth'].std():.2f}")

    with col3:
        st.metric("Mode Rating", f"{filtered_df['ground_truth'].mode().iloc[0]}")

    # Distribution histogram
    fig = px.histogram(
        filtered_df,
        x='ground_truth',
        title='Ground Truth Distribution',
        labels={'ground_truth': 'Rating', 'count': 'Frequency'},
        color_discrete_sequence=['#1f77b4']
    )

    fig.update_layout(
        showlegend=False,
        bargap=0.1
    )

    st.plotly_chart(fig, use_container_width=True)

else:
    warning_message("Ground truth data not found")

st.markdown("---")

# ======================
# Section G: Reports
# ======================
st.header(" Detailed Reports")

tab1, tab2 = st.tabs([" Text Report", " Markdown Report"])

# Text Report
with tab1:
    if text_report:
        st.text_area(
            "Text Report",
            value=text_report,
            height=400,
            disabled=True
        )

        st.download_button(
            label=" Download Text Report",
            data=text_report,
            file_name=f"report_{exp_info['folder']}.txt",
            mime="text/plain"
        )
    else:
        warning_message("Text report not found")

# Markdown Report
with tab2:
    md_report = load_markdown_report(selected_exp_path)

    if md_report:
        st.markdown(md_report)

        st.download_button(
            label=" Download Markdown Report",
            data=md_report,
            file_name=f"report_{exp_info['folder']}.md",
            mime="text/markdown"
        )
    else:
        warning_message("Markdown report not found")

# ======================
# Section H: Actions
# ======================
st.markdown("---")
st.header("⚡ Actions")

col1, col2, col3 = st.columns(3)

with col1:
    if st.button(" Run New Experiment", use_container_width=True, type="primary"):
        st.switch_page("pages/2__Run_Experiment.py")

with col2:
    if st.button(" Try Live Demo", use_container_width=True):
        st.switch_page("pages/4__Live_Demo.py")

with col3:
    if st.button(" Settings", use_container_width=True):
        st.switch_page("pages/5__Settings.py")
