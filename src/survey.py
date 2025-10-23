"""Survey design and management module."""

from typing import Dict, List, Optional
from dataclasses import dataclass, field
import yaml
from pathlib import Path


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
class Question:
    """Represents a survey question."""
    id: str
    text: str
    type: str  # "likert_5", "likert_7", "yes_no", "multiple_choice"
    scale: Optional[LikertScale] = None
    options: Optional[List[str]] = None  # For multiple choice

    def get_reference_statements(self) -> Dict[int, str]:
        """Get reference statements for semantic similarity rating."""
        if self.type.startswith("likert") and self.scale:
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
        elif self.type == "yes_no":
            return 2
        elif self.type == "multiple_choice" and self.options:
            return len(self.options)
        return 0


@dataclass
class Survey:
    """Represents a complete survey."""
    name: str
    description: str
    context: str
    questions: List[Question]
    demographics: List[str] = field(default_factory=list)
    personas: List[str] = field(default_factory=list)
    sample_size: int = 100

    @classmethod
    def from_config(cls, config_path: str) -> 'Survey':
        """Load survey from YAML configuration file."""
        with open(config_path, 'r') as f:
            config = yaml.safe_load(f)

        survey_config = config['survey']

        # Parse questions
        questions = []
        for q_config in survey_config['questions']:
            q_type = q_config['type']

            if q_type.startswith("likert"):
                scale = LikertScale(
                    scale_type=q_type,
                    labels=q_config['scale']
                )
                question = Question(
                    id=q_config['id'],
                    text=q_config['text'],
                    type=q_type,
                    scale=scale
                )
            elif q_type == "yes_no":
                question = Question(
                    id=q_config['id'],
                    text=q_config['text'],
                    type=q_type
                )
            elif q_type == "multiple_choice":
                question = Question(
                    id=q_config['id'],
                    text=q_config['text'],
                    type=q_type,
                    options=q_config['options']
                )
            else:
                raise ValueError(f"Unknown question type: {q_type}")

            questions.append(question)

        return cls(
            name=survey_config['name'],
            description=survey_config['description'],
            context=survey_config['context'],
            questions=questions,
            demographics=survey_config.get('demographics', []),
            personas=survey_config.get('personas', []),
            sample_size=survey_config.get('sample_size', 100)
        )

    def get_question_by_id(self, question_id: str) -> Optional[Question]:
        """Retrieve a question by its ID."""
        for q in self.questions:
            if q.id == question_id:
                return q
        return None

    def format_prompt(self, question: Question, respondent_profile: Optional[Dict] = None) -> str:
        """Format a prompt for LLM response generation."""
        prompt = f"{self.context}\n\n"

        if respondent_profile:
            prompt += "Respondent Profile:\n"
            for key, value in respondent_profile.items():
                prompt += f"- {key}: {value}\n"
            prompt += "\n"

        prompt += f"Question: {question.text}\n\n"
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
