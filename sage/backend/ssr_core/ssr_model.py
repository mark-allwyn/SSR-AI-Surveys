"""Semantic Similarity Rating (SSR) model implementation.

This implementation follows the methodology from:
'LLMs Reproduce Human Purchase Intent via Semantic Similarity Elicitation'
arXiv:2510.08338v2

Key features:
- Uses OpenAI's text-embedding-3-small model
- Implements paper's normalization: subtract min + proportional mapping
- Temperature parameter controls distribution spread
"""

from typing import Dict, List, Optional, Tuple
import numpy as np
from dataclasses import dataclass
from openai import OpenAI
from sklearn.metrics.pairwise import cosine_similarity
from tqdm import tqdm
import os

from .survey import Question, Survey
from .llm_client import Response


@dataclass
class RatingDistribution:
    """Represents a probability distribution over Likert scale points."""
    question_id: str
    respondent_id: str
    distribution: np.ndarray  # Probability distribution over scale points
    scale_labels: Dict[int, str]
    text_response: str
    similarities: Optional[np.ndarray] = None  # Raw similarity scores

    @property
    def expected_value(self) -> float:
        """Calculate expected value of the distribution."""
        scale_points = np.array(list(self.scale_labels.keys()))
        return np.sum(scale_points * self.distribution)

    @property
    def entropy(self) -> float:
        """Calculate Shannon entropy of the distribution."""
        # Avoid log(0)
        probs = self.distribution[self.distribution > 0]
        return -np.sum(probs * np.log(probs))

    @property
    def mode(self) -> int:
        """Return the most likely scale point."""
        scale_points = np.array(list(self.scale_labels.keys()))
        return scale_points[np.argmax(self.distribution)]


class SemanticSimilarityRater:
    """
    Converts textual responses to Likert scale probability distributions
    using semantic similarity.

    Implements the methodology from arXiv:2510.08338v2.
    """

    def __init__(
        self,
        model_name: str = "text-embedding-3-small",
        temperature: float = 1.0,
        normalize_method: str = "paper",
        api_key: Optional[str] = None,
        use_openai: bool = True
    ):
        """
        Initialize the SSR model.

        Args:
            model_name: Embedding model name
                       - "text-embedding-3-small" (OpenAI, paper default)
                       - "text-embedding-3-large" (OpenAI, higher quality)
                       - Any sentence-transformers model (fallback)
            temperature: Temperature parameter (higher = more spread out distribution)
            normalize_method: Method to convert similarities to probabilities
                            - 'paper': Paper's method (subtract min + proportional)
                            - 'softmax': Standard softmax normalization
                            - 'linear': Linear normalization
            api_key: OpenAI API key (if None, uses OPENAI_API_KEY env var)
            use_openai: If False, falls back to sentence-transformers
        """
        self.model_name = model_name
        self.temperature = temperature
        self.normalize_method = normalize_method
        self.use_openai = use_openai

        # Cache for reference statement embeddings (keyed by tuple of statements)
        self._reference_cache = {}

        # Initialize embedding model
        if use_openai and model_name.startswith("text-embedding"):
            self.client = OpenAI(api_key=api_key or os.getenv("OPENAI_API_KEY"))
            self.encoder = None
        else:
            # Fallback to sentence transformers
            from sentence_transformers import SentenceTransformer
            self.encoder = SentenceTransformer(model_name)
            self.client = None
            self.use_openai = False

    def get_embeddings(self, texts: List[str]) -> np.ndarray:
        """
        Get embeddings for a list of texts.

        Args:
            texts: List of text strings to embed

        Returns:
            Array of embeddings (shape: n_texts x embedding_dim)
        """
        if self.use_openai:
            # Use OpenAI embeddings (as in paper)
            response = self.client.embeddings.create(
                model=self.model_name,
                input=texts
            )
            embeddings = np.array([item.embedding for item in response.data])
            return embeddings
        else:
            # Use sentence transformers
            return self.encoder.encode(texts)

    def compute_semantic_similarities(
        self,
        response_text: str,
        reference_statements: List[str]
    ) -> np.ndarray:
        """
        Compute semantic similarities between response and reference statements.

        Args:
            response_text: The textual response to rate
            reference_statements: List of reference statements (scale labels)

        Returns:
            Array of similarity scores (cosine similarities)
        """
        # Get embeddings
        all_texts = [response_text] + reference_statements
        embeddings = self.get_embeddings(all_texts)

        response_embedding = embeddings[0:1]
        reference_embeddings = embeddings[1:]

        # Compute cosine similarities
        similarities = cosine_similarity(response_embedding, reference_embeddings)[0]

        return similarities

    def similarities_to_probabilities(
        self,
        similarities: np.ndarray
    ) -> np.ndarray:
        """
        Convert similarity scores to probability distribution.

        Implements the paper's methodology:
        1. Subtract minimum similarity from all similarities
        2. Apply temperature scaling
        3. Normalize proportionally to sum to 1

        Args:
            similarities: Array of similarity scores

        Returns:
            Probability distribution (sums to 1)
        """
        if self.normalize_method == "paper":
            # Paper's method (arXiv:2510.08338v2)
            # Subtract minimum to shift to positive range
            shifted = similarities - np.min(similarities)

            # Apply temperature scaling (controls "smeariness" of distribution)
            # Higher temperature = more uniform, lower = more peaked
            scaled = shifted / self.temperature

            # Normalize proportionally
            probabilities = scaled / np.sum(scaled) if np.sum(scaled) > 0 else np.ones_like(scaled) / len(scaled)

        elif self.normalize_method == "softmax":
            # Standard softmax with temperature
            exp_sim = np.exp(similarities / self.temperature)
            probabilities = exp_sim / np.sum(exp_sim)

        elif self.normalize_method == "linear":
            # Linear normalization (shift to positive, then normalize)
            shifted = similarities - np.min(similarities)
            probabilities = shifted / np.sum(shifted) if np.sum(shifted) > 0 else np.ones_like(shifted) / len(shifted)

        elif self.normalize_method == "rank":
            # Rank-based: convert to ranks, then normalize
            ranks = np.argsort(np.argsort(similarities)) + 1
            probabilities = ranks / np.sum(ranks)

        else:
            raise ValueError(f"Unknown normalize_method: {self.normalize_method}")

        return probabilities

    def rate_response(
        self,
        response: Response,
        question: Question
    ) -> RatingDistribution:
        """
        Rate a single response and convert to Likert distribution.

        Args:
            response: Response object
            question: Question object with scale information

        Returns:
            RatingDistribution object
        """
        # Get reference statements from scale labels
        scale_labels = question.get_reference_statements()
        reference_statements = [scale_labels[i] for i in sorted(scale_labels.keys())]

        # Compute similarities
        similarities = self.compute_semantic_similarities(
            response.text_response,
            reference_statements
        )

        # Convert to probabilities
        probabilities = self.similarities_to_probabilities(similarities)

        return RatingDistribution(
            question_id=question.id,
            respondent_id=response.respondent_id,
            distribution=probabilities,
            scale_labels=scale_labels,
            text_response=response.text_response,
            similarities=similarities
        )

    def rate_responses(
        self,
        responses: List[Response],
        survey: Survey,
        show_progress: bool = True
    ) -> List[RatingDistribution]:
        """
        Rate multiple responses using batched embeddings for performance.

        Args:
            responses: List of Response objects
            survey: Survey object
            show_progress: Whether to show progress bar

        Returns:
            List of RatingDistribution objects
        """
        distributions = []

        # Group responses by question for batch processing
        responses_by_question = {}
        for response in responses:
            if response.question_id not in responses_by_question:
                responses_by_question[response.question_id] = []
            responses_by_question[response.question_id].append(response)

        # Process each question's responses as a batch
        total_questions = len(responses_by_question)
        question_iterator = tqdm(responses_by_question.items(),
                                desc="Rating responses",
                                total=total_questions,
                                disable=not show_progress)

        for question_id, question_responses in question_iterator:
            question = survey.get_question_by_id(question_id)
            if question is None:
                print(f"Warning: Question {question_id} not found in survey")
                continue

            # Batch process all responses for this question
            batch_distributions = self._rate_responses_batch(question_responses, question)
            distributions.extend(batch_distributions)

        return distributions

    def _rate_responses_batch(
        self,
        responses: List[Response],
        question: Question
    ) -> List[RatingDistribution]:
        """
        Rate a batch of responses for the same question using batched embeddings.
        Uses caching for reference statement embeddings to avoid re-computing.

        Args:
            responses: List of Response objects for the same question
            question: Question object

        Returns:
            List of RatingDistribution objects
        """
        if not responses:
            return []

        # Get reference statements (same for all responses to this question)
        scale_labels = question.get_reference_statements()
        reference_statements = [scale_labels[i] for i in sorted(scale_labels.keys())]

        # Check cache for reference embeddings
        cache_key = tuple(reference_statements)
        if cache_key in self._reference_cache:
            reference_embeddings = self._reference_cache[cache_key]
            # Only embed response texts
            response_texts = [r.text_response for r in responses]
            response_embeddings = self.get_embeddings(response_texts)
        else:
            # First time seeing these reference statements - embed everything together
            all_texts = [r.text_response for r in responses] + reference_statements
            all_embeddings = self.get_embeddings(all_texts)

            # Split and cache reference embeddings
            n_responses = len(responses)
            response_embeddings = all_embeddings[:n_responses]
            reference_embeddings = all_embeddings[n_responses:]
            self._reference_cache[cache_key] = reference_embeddings

        # Compute similarities for all responses at once
        similarities_matrix = cosine_similarity(response_embeddings, reference_embeddings)

        # Create distributions
        distributions = []
        for i, response in enumerate(responses):
            similarities = similarities_matrix[i]
            probabilities = self.similarities_to_probabilities(similarities)

            distribution = RatingDistribution(
                question_id=question.id,
                respondent_id=response.respondent_id,
                distribution=probabilities,
                scale_labels=scale_labels,
                text_response=response.text_response,
                similarities=similarities
            )
            distributions.append(distribution)

        return distributions


class ResponseRater:
    """
    High-level interface compatible with the pymc-labs/semantic-similarity-rating package.
    Now uses paper's methodology by default.
    """

    def __init__(
        self,
        model_name: str = "text-embedding-3-small",
        temperature: float = 1.0,
        normalize_method: str = "paper",
        api_key: Optional[str] = None,
        use_openai: bool = True
    ):
        """
        Initialize ResponseRater.

        Args:
            model_name: Embedding model (default: OpenAI text-embedding-3-small)
            temperature: Temperature for distribution spread
            normalize_method: 'paper' for arXiv:2510.08338v2 method
            api_key: OpenAI API key
            use_openai: If True, uses OpenAI embeddings
        """
        self.rater = SemanticSimilarityRater(
            model_name=model_name,
            temperature=temperature,
            normalize_method=normalize_method,
            api_key=api_key,
            use_openai=use_openai
        )

    def get_response_pmfs(
        self,
        responses: List[str],
        reference_statements: List[str]
    ) -> np.ndarray:
        """
        Get probability mass functions for responses.

        Args:
            responses: List of textual responses
            reference_statements: List of reference statements (scale labels)

        Returns:
            Array of shape (n_responses, n_scale_points) with probability distributions
        """
        pmfs = []

        for response_text in responses:
            similarities = self.rater.compute_semantic_similarities(
                response_text,
                reference_statements
            )
            probabilities = self.rater.similarities_to_probabilities(similarities)
            pmfs.append(probabilities)

        return np.array(pmfs)
