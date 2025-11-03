"""Generate ERD diagram for SSR Pipeline database schema."""

import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from matplotlib.patches import FancyBboxPatch, FancyArrowPatch
import numpy as np

# Set up the figure
fig, ax = plt.subplots(1, 1, figsize=(20, 14))
ax.set_xlim(0, 100)
ax.set_ylim(0, 100)
ax.axis('off')

# Colors
header_color = '#2C3E50'
table_color = '#ECF0F1'
pk_color = '#E74C3C'
fk_color = '#3498DB'
text_color = '#2C3E50'

def draw_table(ax, x, y, width, height, name, fields, pks=None, fks=None):
    """Draw a database table with fields."""
    if pks is None:
        pks = []
    if fks is None:
        fks = []

    # Header
    header = FancyBboxPatch(
        (x, y + height - 3), width, 3,
        boxstyle="round,pad=0.1",
        edgecolor=header_color,
        facecolor=header_color,
        linewidth=2
    )
    ax.add_patch(header)

    # Table name
    ax.text(x + width/2, y + height - 1.5, name,
            ha='center', va='center',
            fontsize=10, fontweight='bold',
            color='white', family='monospace')

    # Body
    body = FancyBboxPatch(
        (x, y), width, height - 3,
        boxstyle="round,pad=0.1",
        edgecolor=header_color,
        facecolor=table_color,
        linewidth=2
    )
    ax.add_patch(body)

    # Fields
    field_height = (height - 3) / max(len(fields), 1)
    for i, field in enumerate(fields):
        y_pos = y + height - 3 - (i + 0.5) * field_height

        # Determine color
        if field in pks:
            color = pk_color
            marker = '🔑 '
        elif field in fks:
            color = fk_color
            marker = '🔗 '
        else:
            color = text_color
            marker = '   '

        ax.text(x + 0.5, y_pos, f"{marker}{field}",
                ha='left', va='center',
                fontsize=7, color=color,
                family='monospace')

def draw_relationship(ax, x1, y1, x2, y2, label='', style='1:N'):
    """Draw a relationship line between tables."""
    # Calculate arrow direction
    dx = x2 - x1
    dy = y2 - y1

    # Create curved arrow
    arrow = FancyArrowPatch(
        (x1, y1), (x2, y2),
        arrowstyle='-|>',
        connectionstyle=f"arc3,rad=0.2",
        color='#7F8C8D',
        linewidth=1.5,
        alpha=0.7
    )
    ax.add_patch(arrow)

    # Add label
    if label:
        mid_x = (x1 + x2) / 2
        mid_y = (y1 + y2) / 2
        ax.text(mid_x, mid_y, label,
                ha='center', va='bottom',
                fontsize=7, color='#7F8C8D',
                bbox=dict(boxstyle='round,pad=0.3', facecolor='white', edgecolor='none', alpha=0.8))

# Define table positions and details
tables = {
    'SURVEY': {
        'pos': (5, 75, 18, 15),
        'fields': ['survey_id', 'name', 'description', 'context', 'sample_size', 'created_at'],
        'pks': ['survey_id'],
        'fks': []
    },
    'QUESTION': {
        'pos': (28, 75, 18, 15),
        'fields': ['question_id', 'survey_id', 'question_code', 'question_text', 'question_type', 'num_options'],
        'pks': ['question_id'],
        'fks': ['survey_id']
    },
    'SCALE_POINT': {
        'pos': (51, 75, 18, 12),
        'fields': ['scale_point_id', 'question_id', 'scale_value', 'reference_statement'],
        'pks': ['scale_point_id'],
        'fks': ['question_id']
    },
    'EXPERIMENT': {
        'pos': (5, 50, 18, 20),
        'fields': ['experiment_id', 'survey_id', 'experiment_code', 'run_timestamp', 'n_respondents', 'llm_model', 'embedding_model', 'temperature', 'status'],
        'pks': ['experiment_id'],
        'fks': ['survey_id']
    },
    'RESPONDENT': {
        'pos': (28, 50, 18, 12),
        'fields': ['respondent_id', 'experiment_id', 'respondent_code', 'persona_description'],
        'pks': ['respondent_id'],
        'fks': ['experiment_id']
    },
    'GROUND_TRUTH_RESPONSE': {
        'pos': (5, 25, 20, 12),
        'fields': ['gt_response_id', 'experiment_id', 'respondent_id', 'question_id', 'rating_value'],
        'pks': ['gt_response_id'],
        'fks': ['experiment_id', 'respondent_id', 'question_id']
    },
    'LLM_RESPONSE': {
        'pos': (30, 25, 20, 15),
        'fields': ['llm_response_id', 'experiment_id', 'respondent_id', 'question_id', 'response_text', 'predicted_mode', 'predicted_expected_value'],
        'pks': ['llm_response_id'],
        'fks': ['experiment_id', 'respondent_id', 'question_id']
    },
    'PROBABILITY_DISTRIBUTION': {
        'pos': (55, 35, 20, 12),
        'fields': ['distribution_id', 'llm_response_id', 'scale_value', 'probability', 'sort_order'],
        'pks': ['distribution_id'],
        'fks': ['llm_response_id']
    },
    'SIMILARITY_SCORE': {
        'pos': (55, 20, 20, 12),
        'fields': ['similarity_id', 'llm_response_id', 'scale_value', 'cosine_similarity', 'normalized_similarity'],
        'pks': ['similarity_id'],
        'fks': ['llm_response_id']
    },
    'QUESTION_METRIC': {
        'pos': (5, 5, 22, 15),
        'fields': ['metric_id', 'experiment_id', 'question_id', 'metric_source', 'mode_accuracy', 'mae_expected', 'rmse_expected', 'kl_divergence'],
        'pks': ['metric_id'],
        'fks': ['experiment_id', 'question_id']
    },
    'EXPERIMENT_FILE': {
        'pos': (32, 5, 18, 12),
        'fields': ['file_id', 'experiment_id', 'file_type', 'file_path', 'file_size'],
        'pks': ['file_id'],
        'fks': ['experiment_id']
    }
}

# Draw all tables
for name, details in tables.items():
    x, y, w, h = details['pos']
    draw_table(ax, x, y, w, h, name, details['fields'], details['pks'], details['fks'])

# Define relationships
relationships = [
    # (from_table, to_table, label)
    ((14, 75), (37, 85), 'contains'),  # SURVEY -> QUESTION
    ((46, 82), (51, 82), 'has'),  # QUESTION -> SCALE_POINT
    ((14, 70), (14, 62), 'uses'),  # SURVEY -> EXPERIMENT
    ((23, 60), (28, 56), 'generates'),  # EXPERIMENT -> RESPONDENT
    ((14, 50), (15, 37), ''),  # EXPERIMENT -> GROUND_TRUTH_RESPONSE
    ((23, 50), (40, 37), ''),  # EXPERIMENT -> LLM_RESPONSE
    ((37, 50), (35, 37), ''),  # RESPONDENT -> GROUND_TRUTH_RESPONSE
    ((37, 50), (42, 40), ''),  # RESPONDENT -> LLM_RESPONSE
    ((46, 80), (20, 37), 'answered in'),  # QUESTION -> GROUND_TRUTH_RESPONSE
    ((46, 75), (45, 40), 'answered in'),  # QUESTION -> LLM_RESPONSE
    ((50, 32), (55, 41), 'generates'),  # LLM_RESPONSE -> PROBABILITY_DISTRIBUTION
    ((50, 25), (55, 26), 'produces'),  # LLM_RESPONSE -> SIMILARITY_SCORE
    ((14, 50), (16, 20), 'produces'),  # EXPERIMENT -> QUESTION_METRIC
    ((37, 75), (18, 20), 'measured by'),  # QUESTION -> QUESTION_METRIC
    ((14, 50), (32, 17), 'stores'),  # EXPERIMENT -> EXPERIMENT_FILE
]

# Draw relationships
for (x1, y1), (x2, y2), label in relationships:
    draw_relationship(ax, x1, y1, x2, y2, label)

# Add title
ax.text(50, 95, 'SSR Pipeline Database Schema - Entity Relationship Diagram',
        ha='center', va='center',
        fontsize=16, fontweight='bold',
        color=header_color)

# Add legend
legend_x = 77
legend_y = 85
ax.text(legend_x, legend_y + 5, 'Legend:',
        ha='left', va='top',
        fontsize=10, fontweight='bold',
        color=header_color)

legend_items = [
    ('🔑 Primary Key', pk_color),
    ('🔗 Foreign Key', fk_color),
    ('   Regular Field', text_color)
]

for i, (label, color) in enumerate(legend_items):
    y_pos = legend_y + 2 - i * 1.5
    ax.text(legend_x, y_pos, label,
            ha='left', va='center',
            fontsize=8, color=color,
            family='monospace')

# Add relationship info
ax.text(legend_x, legend_y - 6, 'Relationships:',
        ha='left', va='top',
        fontsize=10, fontweight='bold',
        color=header_color)

ax.text(legend_x, legend_y - 8, '—|> One-to-Many',
        ha='left', va='center',
        fontsize=8, color='#7F8C8D')

# Add notes
notes = """
Key Features:
• Fully normalized (3NF)
• 11 core tables
• Captures surveys, experiments,
  responses, and metrics
• Supports historical analysis
• Ready for PostgreSQL
"""

ax.text(legend_x, legend_y - 18, notes,
        ha='left', va='top',
        fontsize=7, color=text_color,
        bbox=dict(boxstyle='round,pad=0.5', facecolor=table_color, edgecolor=header_color, linewidth=1))

plt.tight_layout()
plt.savefig('docs/ssr_database_erd.png', dpi=300, bbox_inches='tight', facecolor='white')
print("ERD diagram saved to: docs/ssr_database_erd.png")
plt.close()
