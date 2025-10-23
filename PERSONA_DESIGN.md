# Persona System Design

## Current Limitations

Currently, personas are limited to three hardcoded attributes:
- `age_group`
- `income_bracket`
- `environmental_consciousness`

This is inflexible for different research domains.

## Proposed Solution: Flexible Persona System

### Architecture

```
Experiment
  ├── persona_config.json  (saved with experiment)
  │   ├── structured_fields (for ground truth)
  │   └── persona_descriptions (for LLM context)
  │
  └── ground_truth.csv
      └── includes persona_config reference
```

### Implementation Plan

#### 1. Enhanced Persona Configuration Structure

```python
{
    "experiment_id": "run_20251023_153000",
    "persona_system": "hybrid",  # "structured", "freeform", or "hybrid"

    # Structured fields (used for ground truth generation)
    "structured_fields": {
        "age_group": ["18-25", "26-35", "36-45", "46-55", "56-65", "65+"],
        "income_bracket": ["<$30k", "$30k-$50k", "$50k-$75k", "$75k-$100k",
                          "$100k-$150k", ">$150k"],
        "env_consciousness": ["Not concerned", "Slightly concerned",
                             "Moderately concerned", "Very concerned",
                             "Extremely concerned"],

        # Custom fields added by user
        "custom_fields": {
            "education": ["High School", "Bachelor's", "Master's", "PhD"],
            "tech_savviness": ["Novice", "Intermediate", "Advanced", "Expert"],
            "risk_tolerance": ["Risk Averse", "Moderate", "Risk Seeking"]
        }
    },

    # Ground truth mapping rules
    "ground_truth_rules": {
        "high_rating_triggers": ["Very concerned", "Extremely concerned",
                                "Expert", "Risk Seeking"],
        "low_rating_triggers": ["Not concerned", "Novice", "Risk Averse"],
        "neutral_triggers": ["Moderately concerned", "Moderate", "Intermediate"]
    },

    # Free-form persona descriptions (used for LLM response generation)
    "persona_templates": [
        {
            "id": "persona_001",
            "description": "A 35-year-old tech entrepreneur in San Francisco. " +
                          "Values innovation and efficiency. Early adopter of new " +
                          "technology. High income, environmentally conscious.",
            "structured_mapping": {
                "age_group": "36-45",
                "income_bracket": ">$150k",
                "env_consciousness": "Very concerned",
                "education": "Master's",
                "tech_savviness": "Expert",
                "risk_tolerance": "Risk Seeking"
            }
        },
        {
            "id": "persona_002",
            "description": "A 68-year-old retired teacher living in rural Iowa. " +
                          "Fixed income, cautious about change. Prefers traditional " +
                          "methods. Not very tech-savvy.",
            "structured_mapping": {
                "age_group": "65+",
                "income_bracket": "<$30k",
                "env_consciousness": "Not concerned",
                "education": "Bachelor's",
                "tech_savviness": "Novice",
                "risk_tolerance": "Risk Averse"
            }
        }
    ]
}
```

#### 2. Updated UI Components

**Settings Page - Persona Configuration Tab:**

```
[Persona System Type]
○ Structured Only (current behavior)
○ Free-form Descriptions Only
● Hybrid (structured + descriptions) [RECOMMENDED]

[Structured Fields]
┌─────────────────────────────────────┐
│ Field Name: education               │
│ Categories (one per line):          │
│ High School                         │
│ Bachelor's Degree                   │
│ Master's Degree                     │
│ PhD                                 │
│                                     │
│ [+ Add Field] [- Remove Field]      │
└─────────────────────────────────────┘

[Persona Templates]
┌─────────────────────────────────────┐
│ Template 1                          │
│ Description:                        │
│ [Large text area for description]   │
│                                     │
│ Structured Mapping:                 │
│ age_group: 36-45                    │
│ income_bracket: >$150k              │
│ env_consciousness: Very concerned   │
│ education: Master's                 │
│                                     │
│ [+ Add Template] [- Remove]         │
└─────────────────────────────────────┘
```

**Run Experiment Page:**

```
[Persona Configuration]

Persona Mode: [Dropdown: Auto-generate | Use Templates | Mixed]

If Auto-generate:
  - Number of personas: [50]
  - Distribution: [Uniform ▼]
  - Uses structured fields only

If Use Templates:
  - Shows list of saved templates
  - Select which templates to use
  - Number of respondents per template

If Mixed:
  - 50% auto-generated from structured fields
  - 50% from templates
```

#### 3. Modified Pipeline Files

**src/llm_client.py - Enhanced RespondentProfile:**

```python
@dataclass
class RespondentProfile:
    """Enhanced profile with flexible attributes."""

    # Core attributes (always present)
    respondent_id: str

    # Structured attributes (for ground truth)
    structured_attributes: Dict[str, str]  # e.g., {"age_group": "26-35", ...}

    # Free-form description (for LLM context)
    description: Optional[str] = None

    # Full attribute set (structured + custom)
    all_attributes: Dict = None

    def to_dict(self) -> Dict:
        return {
            'respondent_id': self.respondent_id,
            'structured_attributes': self.structured_attributes,
            'description': self.description,
            'all_attributes': self.all_attributes
        }

    def get_llm_context(self) -> str:
        """Get context string for LLM."""
        if self.description:
            return self.description
        else:
            # Generate from structured attributes
            attrs = [f"{k}: {v}" for k, v in self.structured_attributes.items()]
            return "Respondent profile: " + ", ".join(attrs)


def generate_flexible_profiles(
    persona_config: Dict,
    n_profiles: int = 100,
    mode: str = "auto"
) -> List[RespondentProfile]:
    """
    Generate profiles using flexible configuration.

    Args:
        persona_config: Configuration dict with structured_fields and persona_templates
        n_profiles: Number of profiles to generate
        mode: "auto" (from structured), "templates" (from descriptions), or "mixed"

    Returns:
        List of RespondentProfile objects
    """
    profiles = []

    if mode == "auto":
        # Generate from structured fields
        structured_fields = persona_config['structured_fields']

        for i in range(n_profiles):
            attrs = {}
            for field_name, categories in structured_fields.items():
                if field_name != "custom_fields":
                    attrs[field_name] = random.choice(categories)

            # Add custom fields
            if 'custom_fields' in structured_fields:
                for field_name, categories in structured_fields['custom_fields'].items():
                    attrs[field_name] = random.choice(categories)

            profile = RespondentProfile(
                respondent_id=f"R{i+1:03d}",
                structured_attributes=attrs,
                all_attributes=attrs
            )
            profiles.append(profile)

    elif mode == "templates":
        # Use persona templates
        templates = persona_config.get('persona_templates', [])

        for i in range(n_profiles):
            template = random.choice(templates)

            profile = RespondentProfile(
                respondent_id=f"R{i+1:03d}",
                structured_attributes=template['structured_mapping'],
                description=template['description'],
                all_attributes=template['structured_mapping']
            )
            profiles.append(profile)

    elif mode == "mixed":
        # 50/50 split
        half = n_profiles // 2
        auto_profiles = generate_flexible_profiles(persona_config, half, "auto")
        template_profiles = generate_flexible_profiles(persona_config, n_profiles - half, "templates")
        profiles = auto_profiles + template_profiles

    return profiles
```

**ground_truth_pipeline.py - Modified:**

```python
def generate_ground_truth_ratings_flexible(
    survey: Survey,
    profiles: List[RespondentProfile],
    persona_config: Dict,
    seed: int = 100
) -> pd.DataFrame:
    """
    Generate ground truth using flexible persona attributes.
    """
    np.random.seed(seed)

    # Get rules from config
    rules = persona_config.get('ground_truth_rules', {})
    high_triggers = rules.get('high_rating_triggers', [])
    low_triggers = rules.get('low_rating_triggers', [])
    neutral_triggers = rules.get('neutral_triggers', [])

    records = []

    for profile in profiles:
        # Determine tendency based on ALL attributes
        tendency_score = 0

        for attr_name, attr_value in profile.structured_attributes.items():
            if attr_value in high_triggers:
                tendency_score += 1
            elif attr_value in low_triggers:
                tendency_score -= 1
            # neutral has no effect

        # Map score to tendency
        if tendency_score > 0:
            tendency = "positive"
        elif tendency_score < 0:
            tendency = "negative"
        else:
            tendency = "neutral"

        # Generate ratings based on tendency (same as before)
        for question in survey.questions:
            # ... existing logic ...
            pass

    return pd.DataFrame(records)
```

#### 4. Experiment Metadata Storage

**Save persona config with each experiment:**

```python
# In ground_truth_pipeline.py main()

# Save persona configuration
persona_config_path = experiment_dir / 'persona_config.json'
with open(persona_config_path, 'w') as f:
    json.dump(st.session_state.persona_config, f, indent=2)

print(f"    ✓ Saved persona configuration to {persona_config_path}")
```

**Load persona config when viewing results:**

```python
# In View Results page

persona_config_path = selected_exp_path / 'persona_config.json'
if persona_config_path.exists():
    with open(persona_config_path, 'r') as f:
        persona_config = json.load(f)

    # Display persona configuration
    with st.expander("Persona Configuration Used"):
        st.json(persona_config)
```

### Benefits of This Approach

1. **Backward Compatible**: Existing experiments still work
2. **Flexible**: Can adapt to any research domain
3. **Traceable**: Each experiment knows what personas were used
4. **Rich Context**: LLM gets full persona descriptions
5. **Deterministic Ground Truth**: Structured fields still control ground truth
6. **Reusable**: Save persona templates for future experiments

### Migration Path

**Phase 1** (Quick win):
- Add custom fields support to existing structured system
- Save persona_config.json with experiments

**Phase 2**:
- Add persona template editor in UI
- Allow free-form descriptions

**Phase 3**:
- Advanced: Import personas from CSV
- Persona library/sharing

### Example Use Cases

**Market Research:**
```python
custom_fields = {
    "brand_loyalty": ["Switcher", "Somewhat Loyal", "Very Loyal"],
    "shopping_frequency": ["Rarely", "Monthly", "Weekly", "Daily"],
    "price_sensitivity": ["Low", "Medium", "High"]
}
```

**Healthcare:**
```python
custom_fields = {
    "health_literacy": ["Low", "Medium", "High"],
    "insurance_type": ["None", "Public", "Private"],
    "chronic_conditions": ["0", "1-2", "3+"]
}
```

**Education:**
```python
custom_fields = {
    "learning_style": ["Visual", "Auditory", "Kinesthetic"],
    "prior_knowledge": ["Novice", "Intermediate", "Advanced"],
    "motivation": ["Low", "Medium", "High"]
}
```

### UI Mockup for Settings

```
╔════════════════════════════════════════════════════════════╗
║ Persona Configuration                                      ║
╠════════════════════════════════════════════════════════════╣
║                                                            ║
║ Persona System Mode:                                       ║
║ ● Hybrid (Recommended)                                     ║
║ ○ Structured Fields Only                                   ║
║ ○ Free-form Templates Only                                 ║
║                                                            ║
║ ──────────────────────────────────────────────────────────║
║                                                            ║
║ Core Structured Fields:                                    ║
║ ┌────────────────────────────────────────────────────┐    ║
║ │ Age Groups:                                        │    ║
║ │ 18-25, 26-35, 36-45, 46-55, 56-65, 65+           │    ║
║ │                                                    │    ║
║ │ Income Brackets:                                   │    ║
║ │ <$30k, $30k-$50k, $50k-$75k, ...                  │    ║
║ │                                                    │    ║
║ │ Environmental Consciousness:                       │    ║
║ │ Not concerned, Slightly concerned, ...             │    ║
║ └────────────────────────────────────────────────────┘    ║
║                                                            ║
║ ──────────────────────────────────────────────────────────║
║                                                            ║
║ Custom Fields:                                             ║
║ ┌────────────────────────────────────────────────────┐    ║
║ │ Field: education                              [X]  │    ║
║ │ Categories: High School, Bachelor's, Master's, PhD │    ║
║ │                                                    │    ║
║ │ Field: tech_savviness                        [X]  │    ║
║ │ Categories: Novice, Intermediate, Advanced, Expert │    ║
║ │                                                    │    ║
║ │ [+ Add Custom Field]                               │    ║
║ └────────────────────────────────────────────────────┘    ║
║                                                            ║
║ ──────────────────────────────────────────────────────────║
║                                                            ║
║ Persona Templates:                                         ║
║ ┌────────────────────────────────────────────────────┐    ║
║ │ Template 1: Tech Entrepreneur              [Edit]  │    ║
║ │ "A 35-year-old tech entrepreneur..."               │    ║
║ │ Mappings: age=36-45, income=>$150k, ...            │    ║
║ │                                                    │    ║
║ │ Template 2: Retired Teacher                [Edit]  │    ║
║ │ "A 68-year-old retired teacher..."                 │    ║
║ │ Mappings: age=65+, income=<$30k, ...               │    ║
║ │                                                    │    ║
║ │ [+ Add Template]                                   │    ║
║ └────────────────────────────────────────────────────┘    ║
║                                                            ║
║ [Save Configuration]                                       ║
║                                                            ║
╚════════════════════════════════════════════════════════════╝
```

## Recommendation

**Start with Phase 1**: Add custom fields support. This gives you 80% of the flexibility with minimal changes.

Would you like me to implement this? I can:
1. Add custom fields to the existing system
2. Save persona_config.json with each experiment
3. Update the UI to allow adding/editing custom fields
4. Keep it backward compatible with existing experiments
