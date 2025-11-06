"""Survey design and management module."""

from typing import Dict, List, Optional
from dataclasses import dataclass, field
import yaml
from pathlib import Path
from src.demographics import Gender, validate_age_group, validate_occupation, format_demographic_profile


@dataclass
class LikertScale:
    """Represents a Likert scale with labeled points."""
    scale_type: str  # e.g., "likert_5", "likert_7"
    labels: Dict[int, str]  # e.g., {1: "Strongly disagree", 5: "Strongly agree"}

    @property
    def min_value(self) -> int:
        return min(self.labels.keys())

    @property
    def max_value(self) -> int:
        return max(self.labels.keys())

    @property
    def num_points(self) -> int:
        return len(self.labels)


@dataclass
class Respondent:
    """Represents a survey respondent with demographic information."""
    respondent_id: str
    gender: str = "Unknown"
    age_group: str = "Unknown"
    persona_group: str = "General"
    occupation: str = "Unknown"

    def __post_init__(self):
        """Validate demographic fields."""
        self.age_group = validate_age_group(self.age_group)
        self.occupation = validate_occupation(self.occupation)

    def to_dict(self) -> Dict:
        """Convert to dictionary format."""
        return {
            'respondent_id': self.respondent_id,
            'gender': self.gender,
            'age_group': self.age_group,
            'persona_group': self.persona_group,
            'occupation': self.occupation
        }

    def format_profile(self) -> str:
        """Format demographic information for LLM prompts."""
        return format_demographic_profile(self.gender, self.age_group, self.occupation)


@dataclass
class PersonaGroup:
    """Represents a persona group with target demographics."""
    name: str
    description: str
    personas: List[str]
    target_demographics: Dict[str, List] = field(default_factory=dict)
    weight: float = 1.0  # Relative weight for sampling

    def sample_demographics(self) -> Dict[str, str]:
        """
        Sample demographics based on target distribution.

        Returns:
            Dictionary with 'gender', 'age_group', 'occupation'
        """
        import random

        demographics = {}

        # Sample gender
        if 'gender' in self.target_demographics:
            demographics['gender'] = random.choice(self.target_demographics['gender'])
        else:
            demographics['gender'] = "Unknown"

        # Sample age_group
        if 'age_group' in self.target_demographics:
            demographics['age_group'] = random.choice(self.target_demographics['age_group'])
        else:
            demographics['age_group'] = "Unknown"

        # Sample occupation
        if 'occupation' in self.target_demographics:
            demographics['occupation'] = random.choice(self.target_demographics['occupation'])
        else:
            demographics['occupation'] = "Unknown"

        return demographics


@dataclass
class Category:
    """Represents a product/service category in multi-category surveys."""
    id: str
    name: str
    description: str = ""
    context: str = ""

    def format_context(self) -> str:
        """Format category context for LLM prompts."""
        if self.description and self.context:
            return f"{self.name}: {self.description}\n{self.context}"
        elif self.context:
            return f"{self.name}: {self.context}"
        elif self.description:
            return f"{self.name}: {self.description}"
        return self.name


@dataclass
class Question:
    """Represents a survey question."""
    id: str
    text: str
    type: str  # "likert_5", "likert_7", "yes_no", "multiple_choice", "preference_scale"
    scale: Optional[LikertScale] = None
    options: Optional[List[str]] = None  # For multiple choice
    category: Optional[str] = None  # Category ID this question belongs to
    categories_compared: Optional[List[str]] = None  # For preference_scale questions

    def get_reference_statements(self) -> Dict[int, str]:
        """Get reference statements for semantic similarity rating."""
        if self.type.startswith("likert") and self.scale:
            return self.scale.labels
        elif self.type == "preference_scale" and self.scale:
            return self.scale.labels
        elif self.type == "yes_no":
            return {1: "No", 2: "Yes"}
        elif self.type == "multiple_choice" and self.options:
            return {i+1: opt for i, opt in enumerate(self.options)}
        else:
            raise ValueError(f"Cannot get reference statements for question type: {self.type}")

    @property
    def num_options(self) -> int:
        """Get number of response options."""
        if self.type.startswith("likert") and self.scale:
            return self.scale.num_points
        elif self.type == "preference_scale" and self.scale:
            return self.scale.num_points
        elif self.type == "yes_no":
            return 2
        elif self.type == "multiple_choice" and self.options:
            return len(self.options)
        return 0

    def is_comparative(self) -> bool:
        """Check if this is a comparative preference question."""
        return self.type == "preference_scale" and self.categories_compared is not None

    def format_question_text(self, categories: Dict[str, 'Category']) -> str:
        """
        Format question text with category names substituted.

        For comparative questions, replaces {category_id} placeholders with actual category names.
        """
        text = self.text
        if self.categories_compared:
            for cat_id in self.categories_compared:
                if cat_id in categories:
                    text = text.replace(f"{{{cat_id}}}", categories[cat_id].name)
        return text


@dataclass
class Survey:
    """Represents a complete survey."""
    name: str
    description: str
    context: str
    questions: List[Question]
    demographics: List[str] = field(default_factory=list)
    personas: List[str] = field(default_factory=list)
    persona_groups: List[PersonaGroup] = field(default_factory=list)
    sample_size: int = 100
    categories: Optional[List[Category]] = None  # Multi-category support

    def has_categories(self) -> bool:
        """Check if this is a multi-category survey."""
        return self.categories is not None and len(self.categories) > 0

    def get_category_by_id(self, category_id: str) -> Optional[Category]:
        """Retrieve a category by its ID."""
        if not self.categories:
            return None
        for cat in self.categories:
            if cat.id == category_id:
                return cat
        return None

    def get_categories_dict(self) -> Dict[str, Category]:
        """Get categories as a dictionary keyed by ID."""
        if not self.categories:
            return {}
        return {cat.id: cat for cat in self.categories}

    def get_questions_by_category(self, category_id: str) -> List[Question]:
        """Get all questions for a specific category."""
        return [q for q in self.questions if q.category == category_id]

    def get_comparative_questions(self) -> List[Question]:
        """Get all comparative preference questions."""
        return [q for q in self.questions if q.is_comparative()]

    def get_non_comparative_questions(self) -> List[Question]:
        """Get all non-comparative questions."""
        return [q for q in self.questions if not q.is_comparative()]

    @classmethod
    def from_config(cls, config_path: str) -> 'Survey':
        """Load survey from YAML configuration file."""
        from pathlib import Path

        # Resolve to absolute path to handle relative paths from different working directories
        config_path = Path(config_path).resolve()

        with open(config_path, 'r') as f:
            config = yaml.safe_load(f)

        survey_config = config['survey']

        # Parse categories (multi-category support)
        categories = None
        if 'categories' in survey_config:
            categories = []
            for cat_config in survey_config['categories']:
                category = Category(
                    id=cat_config['id'],
                    name=cat_config['name'],
                    description=cat_config.get('description', ''),
                    context=cat_config.get('context', '')
                )
                categories.append(category)

        # Parse questions
        questions = []
        for q_config in survey_config['questions']:
            # All questions must be directly defined
            q_type = q_config['type']
            q_id = q_config['id']
            q_text = q_config['text']
            q_scale = q_config.get('scale')
            q_options = q_config.get('options')
            q_category = q_config.get('category')  # NEW: Category assignment
            q_categories_compared = q_config.get('categories_compared')  # NEW: For preference questions

            # Create Question object based on type
            if q_type.startswith("likert") or q_type == "preference_scale":
                if not q_scale:
                    raise ValueError(f"Question {q_id}: {q_type} questions must have a scale")
                scale = LikertScale(
                    scale_type=q_type,
                    labels=q_scale
                )
                question = Question(
                    id=q_id,
                    text=q_text,
                    type=q_type,
                    scale=scale,
                    category=q_category,
                    categories_compared=q_categories_compared
                )
            elif q_type == "yes_no":
                question = Question(
                    id=q_id,
                    text=q_text,
                    type=q_type,
                    category=q_category
                )
            elif q_type == "multiple_choice":
                if not q_options:
                    raise ValueError(f"Question {q_id}: multiple_choice questions must have options")
                question = Question(
                    id=q_id,
                    text=q_text,
                    type=q_type,
                    options=q_options,
                    category=q_category
                )
            else:
                raise ValueError(f"Unknown question type: {q_type}")

            questions.append(question)

        # Parse persona_groups if present
        persona_groups = []
        if 'persona_groups' in survey_config:
            for pg_config in survey_config['persona_groups']:
                persona_group = PersonaGroup(
                    name=pg_config['name'],
                    description=pg_config.get('description', ''),
                    personas=pg_config.get('personas', []),
                    target_demographics=pg_config.get('target_demographics', {}),
                    weight=pg_config.get('weight', 1.0)
                )
                persona_groups.append(persona_group)

        return cls(
            name=survey_config['name'],
            description=survey_config['description'],
            context=survey_config['context'],
            questions=questions,
            demographics=survey_config.get('demographics', []),
            personas=survey_config.get('personas', []),
            persona_groups=persona_groups,
            sample_size=survey_config.get('sample_size', 100),
            categories=categories  # NEW: Multi-category support
        )

    def get_question_by_id(self, question_id: str) -> Optional[Question]:
        """Retrieve a question by its ID."""
        for q in self.questions:
            if q.id == question_id:
                return q
        return None

    def format_prompt(self, question: Question, respondent_profile: Optional[Dict] = None) -> str:
        """
        Format a prompt for LLM response generation with category-aware context.

        For multi-category surveys:
        - Includes category-specific context for regular questions
        - Includes both category contexts for comparative questions
        - Formats question text with category names substituted
        """
        context_parts = []

        # Add global context if present (backwards compatible)
        if self.context:
            context_parts.append(self.context)

        # Add category-specific context for multi-category surveys
        if self.has_categories():
            if question.is_comparative():
                # Comparative question: include both categories being compared
                context_parts.append("\n--- Comparison Context ---")
                for cat_id in question.categories_compared:
                    category = self.get_category_by_id(cat_id)
                    if category:
                        context_parts.append(f"\n{category.format_context()}")
            elif question.category:
                # Regular question with category assignment
                category = self.get_category_by_id(question.category)
                if category:
                    context_parts.append(f"\n--- {category.name} ---")
                    context_parts.append(category.context)

        prompt = "\n".join(context_parts)

        if prompt:
            prompt += "\n\n"

        if respondent_profile:
            prompt += "Respondent Profile:\n"
            for key, value in respondent_profile.items():
                prompt += f"- {key}: {value}\n"
            prompt += "\n"

        # Format question text (substitutes category names in comparative questions)
        question_text = question.text
        if self.has_categories() and (question.is_comparative() or question.category):
            categories_dict = self.get_categories_dict()
            question_text = question.format_question_text(categories_dict)

        prompt += f"Question: {question_text}\n\n"
        prompt += "Please provide your response in a few sentences explaining your thoughts and reasoning."

        return prompt

    def to_dict(self) -> Dict:
        """Convert survey to dictionary format."""
        return {
            'name': self.name,
            'description': self.description,
            'context': self.context,
            'questions': [
                {
                    'id': q.id,
                    'text': q.text,
                    'type': q.type,
                    'scale': q.scale.labels
                }
                for q in self.questions
            ],
            'demographics': self.demographics,
            'sample_size': self.sample_size
        }
