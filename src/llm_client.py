"""LLM client for generating survey responses."""

from typing import Dict, List, Optional
import os
from dataclasses import dataclass
from tqdm import tqdm
import time

try:
    from openai import OpenAI
except ImportError:
    OpenAI = None

try:
    from anthropic import Anthropic
except ImportError:
    Anthropic = None

from .survey import Survey, Question


@dataclass
class RespondentProfile:
    """Profile for a simulated survey respondent."""
    age_group: str
    income_bracket: str
    environmental_consciousness: str
    other_attributes: Dict = None

    def to_dict(self) -> Dict:
        profile = {
            'age_group': self.age_group,
            'income_bracket': self.income_bracket,
            'environmental_consciousness': self.environmental_consciousness
        }
        if self.other_attributes:
            profile.update(self.other_attributes)
        return profile


@dataclass
class Response:
    """Represents a survey response."""
    respondent_id: str
    question_id: str
    text_response: str
    respondent_profile: Dict
    timestamp: Optional[str] = None


class LLMClient:
    """Client for generating survey responses using LLMs."""

    def __init__(
        self,
        provider: str = "openai",
        model: str = "gpt-4",
        api_key: Optional[str] = None,
        temperature: float = 0.7,
        max_tokens: int = 500
    ):
        """
        Initialize LLM client.

        Args:
            provider: LLM provider ('openai' or 'anthropic')
            model: Model name
            api_key: API key (if None, will use environment variable)
            temperature: Sampling temperature
            max_tokens: Maximum tokens in response
        """
        self.provider = provider
        self.model = model
        self.temperature = temperature
        self.max_tokens = max_tokens

        if provider == "openai":
            if OpenAI is None:
                raise ImportError("openai package not installed. Run: pip install openai")
            api_key = api_key or os.getenv("OPENAI_API_KEY")
            self.client = OpenAI(api_key=api_key)
        elif provider == "anthropic":
            if Anthropic is None:
                raise ImportError("anthropic package not installed. Run: pip install anthropic")
            api_key = api_key or os.getenv("ANTHROPIC_API_KEY")
            self.client = Anthropic(api_key=api_key)
        else:
            raise ValueError(f"Unsupported provider: {provider}")

    def generate_response(
        self,
        prompt: str,
        system_message: Optional[str] = None
    ) -> str:
        """Generate a single response from the LLM."""
        if self.provider == "openai":
            messages = []
            if system_message:
                messages.append({"role": "system", "content": system_message})
            messages.append({"role": "user", "content": prompt})

            response = self.client.chat.completions.create(
                model=self.model,
                messages=messages,
                temperature=self.temperature,
                max_tokens=self.max_tokens
            )
            return response.choices[0].message.content

        elif self.provider == "anthropic":
            response = self.client.messages.create(
                model=self.model,
                max_tokens=self.max_tokens,
                temperature=self.temperature,
                system=system_message or "",
                messages=[{"role": "user", "content": prompt}]
            )
            return response.content[0].text

    def generate_responses(
        self,
        survey: Survey,
        respondent_profiles: List[RespondentProfile],
        system_message: Optional[str] = None
    ) -> List[Response]:
        """
        Generate responses for all questions and respondents.

        Args:
            survey: Survey object
            respondent_profiles: List of respondent profiles
            system_message: Optional system message for LLM

        Returns:
            List of Response objects
        """
        if system_message is None:
            system_message = (
                "You are participating in a survey. Answer each question thoughtfully "
                "based on the provided respondent profile. Provide a natural, detailed "
                "response that explains your perspective and reasoning."
            )

        responses = []
        total = len(respondent_profiles) * len(survey.questions)

        with tqdm(total=total, desc="Generating responses") as pbar:
            for i, profile in enumerate(respondent_profiles):
                for question in survey.questions:
                    prompt = survey.format_prompt(question, profile.to_dict())

                    try:
                        text_response = self.generate_response(prompt, system_message)

                        response = Response(
                            respondent_id=f"R{i+1:03d}",
                            question_id=question.id,
                            text_response=text_response,
                            respondent_profile=profile.to_dict()
                        )
                        responses.append(response)

                        # Brief delay to avoid rate limits
                        time.sleep(0.1)

                    except Exception as e:
                        print(f"\nError generating response for {profile} on {question.id}: {e}")

                    pbar.update(1)

        return responses


def generate_diverse_profiles(
    n_profiles: int = 100,
    persona_config: Optional[Dict] = None
) -> List[RespondentProfile]:
    """
    Generate diverse respondent profiles for survey simulation.

    Args:
        n_profiles: Number of profiles to generate
        persona_config: Optional configuration dict with custom persona fields
                       Format: {
                           'age_groups': [...],
                           'income_brackets': [...],
                           'env_consciousness': [...],
                           'custom_fields': {
                               'field_name': ['value1', 'value2', ...]
                           }
                       }

    Returns:
        List of RespondentProfile objects
    """
    import random

    # Default values
    if persona_config is None:
        persona_config = {}

    age_groups = persona_config.get('age_groups',
                                     ["18-25", "26-35", "36-45", "46-55", "56-65", "65+"])
    income_brackets = persona_config.get('income_brackets',
                                         ["<$30k", "$30k-$50k", "$50k-$75k", "$75k-$100k",
                                          "$100k-$150k", ">$150k"])
    env_consciousness = persona_config.get('env_consciousness',
                                           ["Not concerned", "Slightly concerned",
                                            "Moderately concerned", "Very concerned",
                                            "Extremely concerned"])

    custom_fields = persona_config.get('custom_fields', {})

    profiles = []
    for i in range(n_profiles):
        # Generate custom attributes if any
        other_attrs = {}
        for field_name, field_values in custom_fields.items():
            if field_values:  # Only add if field has values
                other_attrs[field_name] = random.choice(field_values)

        profile = RespondentProfile(
            age_group=random.choice(age_groups),
            income_bracket=random.choice(income_brackets),
            environmental_consciousness=random.choice(env_consciousness),
            other_attributes=other_attrs if other_attrs else None
        )
        profiles.append(profile)

    return profiles
