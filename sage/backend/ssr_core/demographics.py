"""Demographics constants and utilities for SSR Pipeline."""

from enum import Enum
from typing import Dict, List


class Gender(Enum):
    """Gender categories for respondents."""
    MALE = "Male"
    FEMALE = "Female"
    NON_BINARY = "Non-binary"
    PREFER_NOT_TO_SAY = "Prefer not to say"
    UNKNOWN = "Unknown"

    @classmethod
    def from_string(cls, value: str) -> 'Gender':
        """Convert string to Gender enum, defaulting to UNKNOWN."""
        if not value or value == "Unknown":
            return cls.UNKNOWN

        value_upper = value.upper().replace(" ", "_")
        for gender in cls:
            if gender.name == value_upper:
                return gender

        # Try matching by value
        for gender in cls:
            if gender.value.upper() == value.upper():
                return gender

        return cls.UNKNOWN


# Age group definitions
AGE_GROUPS = [
    "18-24",
    "25-34",
    "35-44",
    "45-54",
    "55-64",
    "65+",
    "Unknown"
]

# Occupation categories
OCCUPATION_CATEGORIES = [
    "Student",
    "Professional",
    "Manager",
    "Technical",
    "Service",
    "Sales",
    "Administrative",
    "Skilled Trade",
    "Retired",
    "Unemployed",
    "Self-employed",
    "Other",
    "Unknown"
]


def validate_age_group(age_group: str) -> str:
    """Validate age group value, returning 'Unknown' if invalid."""
    if age_group in AGE_GROUPS:
        return age_group
    return "Unknown"


def validate_occupation(occupation: str) -> str:
    """Validate occupation value, returning 'Unknown' if invalid."""
    if occupation in OCCUPATION_CATEGORIES:
        return occupation
    return "Unknown"


def get_demographic_summary(demographics: List[Dict]) -> Dict:
    """
    Calculate demographic distribution summary.

    Args:
        demographics: List of dicts with 'gender', 'age_group', 'occupation' keys

    Returns:
        Dictionary with counts for each demographic category
    """
    summary = {
        'gender': {},
        'age_group': {},
        'occupation': {},
        'total': len(demographics)
    }

    for demo in demographics:
        # Count gender
        gender = demo.get('gender', 'Unknown')
        summary['gender'][gender] = summary['gender'].get(gender, 0) + 1

        # Count age group
        age_group = demo.get('age_group', 'Unknown')
        summary['age_group'][age_group] = summary['age_group'].get(age_group, 0) + 1

        # Count occupation
        occupation = demo.get('occupation', 'Unknown')
        summary['occupation'][occupation] = summary['occupation'].get(occupation, 0) + 1

    return summary


def format_demographic_profile(gender: str, age_group: str, occupation: str) -> str:
    """
    Format demographic information into a natural language string for LLM prompts.

    Args:
        gender: Gender category
        age_group: Age group
        occupation: Occupation category

    Returns:
        Formatted demographic profile string
    """
    parts = []

    if gender and gender != "Unknown":
        parts.append(f"{gender.lower()}")

    if age_group and age_group != "Unknown":
        parts.append(f"aged {age_group}")

    if occupation and occupation != "Unknown":
        if occupation == "Retired":
            parts.append("retired")
        elif occupation == "Student":
            parts.append("a student")
        elif occupation == "Unemployed":
            parts.append("currently unemployed")
        elif occupation == "Self-employed":
            parts.append("self-employed")
        else:
            parts.append(f"working in {occupation.lower()}")

    if not parts:
        return "respondent with unspecified demographics"

    return ", ".join(parts)
