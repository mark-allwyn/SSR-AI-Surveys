"""Reusable metric card components."""

import streamlit as st

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


def metric_card(title: str, value: str, subtitle: str = "", delta: str = None, delta_color: str = "normal"):
    """
    Display a metric card with styling.

    Args:
        title: Metric title
        value: Main value to display
        subtitle: Optional subtitle
        delta: Optional delta/change value
        delta_color: Color for delta ("normal", "inverse", "off")
    """
    st.markdown(f"""
    <div class="metric-card">
        <div style="font-size: 0.9rem; color: #666; margin-bottom: 0.5rem;">{title}</div>
        <div style="font-size: 2rem; font-weight: bold; color: {brand_colors['teal_blue']};">{value}</div>
        {f'<div style="font-size: 0.85rem; color: #888;">{subtitle}</div>' if subtitle else ''}
    </div>
    """, unsafe_allow_html=True)

    if delta:
        st.markdown(f"""
        <div style="margin-top: 0.5rem; font-size: 0.9rem; color: {brand_colors['electric_lime'] if delta.startswith('+') else brand_colors['atomic_orange']};">
            {delta}
        </div>
        """, unsafe_allow_html=True)


def comparison_metrics(human_value: float, llm_value: float, metric_name: str, format_str: str = ".1f"):
    """
    Display comparison metrics between human and LLM.

    Args:
        human_value: Human metric value
        llm_value: LLM metric value
        metric_name: Name of the metric
        format_str: Format string for values
    """
    col1, col2, col3 = st.columns(3)

    with col1:
        st.metric(
            f"Human {metric_name}",
            f"{human_value:{format_str}}%",
        )

    with col2:
        st.metric(
            f"LLM {metric_name}",
            f"{llm_value:{format_str}}%",
        )

    with col3:
        diff = human_value - llm_value
        winner = "Human" if diff > 0 else ("LLM" if diff < 0 else "Tie")
        st.metric(
            "Winner",
            winner,
            f"{diff:+.1f}%"
        )


def success_message(message: str):
    """Display a success message box."""
    st.markdown(f"""
    <div class="success-box">
        [OK] {message}
    </div>
    """, unsafe_allow_html=True)


def warning_message(message: str):
    """Display a warning message box."""
    st.markdown(f"""
    <div class="warning-box">
        [WARNING] {message}
    </div>
    """, unsafe_allow_html=True)


def error_message(message: str):
    """Display an error message box."""
    st.markdown(f"""
    <div class="error-box">
        [ERROR] {message}
    </div>
    """, unsafe_allow_html=True)
