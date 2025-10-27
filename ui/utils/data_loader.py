"""Utilities for loading experiment data."""

from pathlib import Path
import pandas as pd
from typing import Dict, List, Optional, Tuple
import yaml


def get_all_experiments() -> List[Path]:
    """Get all experiment folders sorted by date (newest first)."""
    experiments_dir = Path("experiments")
    if not experiments_dir.exists():
        return []

    experiment_folders = sorted(
        experiments_dir.glob("run_*"),
        key=lambda x: x.name,
        reverse=True
    )
    return experiment_folders


def get_experiment_info(experiment_path: Path) -> Dict:
    """Get basic info about an experiment."""
    timestamp = experiment_path.name.replace("run_", "")

    # Format timestamp: YYYYMMDD_HHMMSS -> YYYY-MM-DD HH:MM:SS
    formatted_time = f"{timestamp[:4]}-{timestamp[4:6]}-{timestamp[6:8]} {timestamp[9:11]}:{timestamp[11:13]}:{timestamp[13:15]}"

    info = {
        'folder': experiment_path.name,
        'timestamp': formatted_time,
        'path': experiment_path
    }

    # Load ground truth if exists
    gt_file = experiment_path / "ground_truth.csv"
    if gt_file.exists():
        df = pd.read_csv(gt_file)
        n_respondents = df['respondent_id'].nunique()
        n_questions = df['question_id'].nunique()
        info['n_respondents'] = n_respondents
        info['n_questions'] = n_questions
        info['n_responses'] = len(df)
    else:
        info['n_respondents'] = 0
        info['n_questions'] = 0
        info['n_responses'] = 0

    return info


def load_ground_truth(experiment_path: Path) -> Optional[pd.DataFrame]:
    """Load ground truth CSV from experiment."""
    gt_file = experiment_path / "ground_truth.csv"
    if gt_file.exists():
        return pd.read_csv(gt_file)
    return None


def load_text_report(experiment_path: Path) -> Optional[str]:
    """Load text report from experiment."""
    txt_file = experiment_path / "report.txt"
    if txt_file.exists():
        return txt_file.read_text()
    return None


def load_markdown_report(experiment_path: Path) -> Optional[str]:
    """Load markdown report from experiment."""
    md_file = experiment_path / "report.md"
    if md_file.exists():
        return md_file.read_text()
    return None


def parse_text_report(report_text: str) -> Dict:
    """Parse text report to extract metrics."""
    metrics = {}

    lines = report_text.split('\n')
    current_question = None
    in_overall_section = False

    for i, line in enumerate(lines):
        # Parse overall metrics
        if "Average Mode Accuracy:" in line:
            in_overall_section = True

        if in_overall_section:
            if "Ground Truth" in line and '%' in line:
                # Extract Ground Truth percentage
                human_part = line.split(':')[-1].strip().replace('%', '').strip()
                try:
                    metrics['overall_human_accuracy'] = float(human_part)
                except ValueError:
                    pass
            elif "LLM+SSR" in line and '%' in line:
                # Extract LLM+SSR percentage
                llm_part = line.split(':')[-1].strip().replace('%', '').strip()
                try:
                    metrics['overall_llm_accuracy'] = float(llm_part)
                    in_overall_section = False  # Done with overall section
                except ValueError:
                    pass

        # Parse question-level metrics
        if line.startswith("QUESTION:"):
            current_question = line.split(':')[1].strip()
            metrics[current_question] = {}

        if current_question and "Mode Accuracy:" in line and '|' in line:
            parts = line.split('|')
            if len(parts) >= 2:
                # Extract ground truth accuracy
                human_part = parts[0].split('Ground Truth:')[1].strip().replace('%', '').strip()
                # Extract LLM+SSR accuracy
                llm_part = parts[1].split('LLM+SSR:')[1].strip().replace('%', '').strip()

                try:
                    metrics[current_question]['human_accuracy'] = float(human_part)
                    metrics[current_question]['llm_accuracy'] = float(llm_part)
                except ValueError:
                    pass

        if current_question and "MAE:" in line and '|' in line:
            parts = line.split('|')
            if len(parts) >= 2:
                # Extract ground truth MAE
                human_part = parts[0].split('Ground Truth:')[1].strip()
                # Extract LLM+SSR MAE
                llm_part = parts[1].split('LLM+SSR:')[1].strip()

                try:
                    metrics[current_question]['human_mae'] = float(human_part)
                    metrics[current_question]['llm_mae'] = float(llm_part)
                except ValueError:
                    pass

        if current_question and "RMSE:" in line and '|' in line:
            parts = line.split('|')
            if len(parts) >= 2:
                # Extract ground truth RMSE
                human_part = parts[0].split('Ground Truth:')[1].strip()
                # Extract LLM+SSR RMSE
                llm_part = parts[1].split('LLM+SSR:')[1].strip()

                try:
                    metrics[current_question]['human_rmse'] = float(human_part)
                    metrics[current_question]['llm_rmse'] = float(llm_part)
                except ValueError:
                    pass

        if current_question and "Top-2 Accuracy:" in line and '|' in line:
            parts = line.split('|')
            if len(parts) >= 2:
                # Extract ground truth Top-2 Accuracy
                human_part = parts[0].split('Ground Truth:')[1].strip().replace('%', '').strip()
                # Extract LLM+SSR Top-2 Accuracy
                llm_part = parts[1].split('LLM+SSR:')[1].strip().replace('%', '').strip()

                try:
                    metrics[current_question]['human_top2_accuracy'] = float(human_part)
                    metrics[current_question]['llm_top2_accuracy'] = float(llm_part)
                except ValueError:
                    pass

        if current_question and "Prob at Truth:" in line and '|' in line:
            parts = line.split('|')
            if len(parts) >= 2:
                # Extract ground truth Prob at Truth
                human_part = parts[0].split('Ground Truth:')[1].strip()
                # Extract LLM+SSR Prob at Truth
                llm_part = parts[1].split('LLM+SSR:')[1].strip()

                try:
                    metrics[current_question]['human_prob_at_truth'] = float(human_part)
                    metrics[current_question]['llm_prob_at_truth'] = float(llm_part)
                except ValueError:
                    pass

        if current_question and "KL Divergence:" in line and '|' in line:
            parts = line.split('|')
            if len(parts) >= 2:
                # Extract ground truth KL Divergence
                human_part = parts[0].split('Ground Truth:')[1].strip()
                # Extract LLM+SSR KL Divergence
                llm_part = parts[1].split('LLM+SSR:')[1].strip()

                try:
                    metrics[current_question]['human_kl_divergence'] = float(human_part)
                    metrics[current_question]['llm_kl_divergence'] = float(llm_part)
                except ValueError:
                    pass

    return metrics


def load_survey_config(config_path: str = "config/mixed_survey_config.yaml") -> Dict:
    """Load survey configuration from YAML."""
    config_file = Path(config_path)
    if config_file.exists():
        with open(config_file, 'r') as f:
            return yaml.safe_load(f)
    return {}


def get_available_surveys() -> List[str]:
    """Get list of available survey config files."""
    config_dir = Path("config")
    if not config_dir.exists():
        return []

    survey_files = list(config_dir.glob("*.yaml")) + list(config_dir.glob("*.yml"))
    return [str(f) for f in survey_files]


def delete_experiment(experiment_path: Path) -> bool:
    """Delete an experiment folder and all its contents."""
    import shutil

    try:
        if experiment_path.exists() and experiment_path.is_dir():
            shutil.rmtree(experiment_path)
            return True
        return False
    except Exception as e:
        print(f"Error deleting experiment: {e}")
        return False


def load_distributions(experiment_path: Path) -> Optional[dict]:
    """Load LLM probability distributions from experiment."""
    import json
    dist_file = experiment_path / "llm_distributions.json"
    if dist_file.exists():
        with open(dist_file, 'r') as f:
            return json.load(f)
    return None


def group_experiments_by_survey(experiments: List[Path]) -> Dict[str, List[Path]]:
    """
    Group experiments by survey based on question IDs.

    Returns dict where keys are survey fingerprints (hash of sorted question IDs)
    and values are lists of experiment paths.
    """
    groups = {}

    for exp_path in experiments:
        gt_file = exp_path / "ground_truth.csv"
        if gt_file.exists():
            df = pd.read_csv(gt_file)
            question_ids = tuple(sorted(df['question_id'].unique()))
            fingerprint = str(hash(question_ids))

            if fingerprint not in groups:
                groups[fingerprint] = {
                    'experiments': [],
                    'question_ids': question_ids,
                    'count': 0
                }

            groups[fingerprint]['experiments'].append(exp_path)
            groups[fingerprint]['count'] += 1

    return groups


def calculate_experiment_metrics(experiment_path: Path) -> Optional[Dict]:
    """
    Calculate key metrics for an experiment for comparison/timeline views.

    Returns dict with accuracy, MAE, sample size, and other key metrics.
    """
    text_report = load_text_report(experiment_path)
    if not text_report:
        return None

    metrics = parse_text_report(text_report)
    if not metrics:
        return None

    # Extract overall metrics
    overall_llm = metrics.get('overall_llm_accuracy', 0)
    question_metrics = {k: v for k, v in metrics.items()
                       if k not in ['overall_human_accuracy', 'overall_llm_accuracy']}

    # Calculate averages
    import numpy as np
    llm_maes = [question_metrics[q]['llm_mae'] for q in question_metrics]
    avg_mae = np.mean(llm_maes) if llm_maes else 0

    # Get sample size
    gt_file = experiment_path / "ground_truth.csv"
    n_respondents = 0
    n_questions = len(question_metrics)
    if gt_file.exists():
        df = pd.read_csv(gt_file)
        n_respondents = df['respondent_id'].nunique()

    # Extract timestamp from folder name
    folder_name = experiment_path.name
    timestamp_str = folder_name.replace("run_", "")
    from datetime import datetime
    try:
        timestamp = datetime.strptime(timestamp_str, "%Y%m%d_%H%M%S")
    except:
        timestamp = None

    return {
        'path': experiment_path,
        'folder': folder_name,
        'timestamp': timestamp,
        'accuracy': overall_llm,
        'mae': avg_mae,
        'n_respondents': n_respondents,
        'n_questions': n_questions,
        'question_ids': list(question_metrics.keys())
    }
