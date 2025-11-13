"""
FastAPI Backend for SSR Pipeline Demo Application

This backend provides REST API endpoints for the React demo app to:
- Load and manage survey configurations
- Generate respondent profiles from persona groups
- Generate LLM text responses
- Apply SSR to convert text to probability distributions
- View response datasets
"""

from fastapi import FastAPI, HTTPException, WebSocket, WebSocketDisconnect, Request
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import StreamingResponse
from pydantic import BaseModel, Field
from typing import List, Dict, Optional, Any
import sys
import json
import asyncio
import logging
from pathlib import Path
from datetime import datetime
from dotenv import load_dotenv
import time

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Load environment variables from .env file
load_dotenv()

# Import core SSR modules from bundled ssr_core package
from ssr_core.survey import Survey, PersonaGroup, Question, Category
from ssr_core.llm_client import LLMClient, generate_diverse_profiles, RespondentProfile, Response
from ssr_core.ssr_model import SemanticSimilarityRater, RatingDistribution
from collections import defaultdict
import numpy as np

# Import ground truth metrics
from ground_truth_metrics import compare_survey_runs

# Helper function to convert numpy types to Python native types
def convert_numpy_types(obj):
    """Recursively convert numpy types to Python native types for JSON serialization"""
    import math

    if isinstance(obj, np.integer):
        return int(obj)
    elif isinstance(obj, np.floating):
        val = float(obj)
        # Handle NaN and Infinity
        if math.isnan(val):
            return None  # Convert NaN to null in JSON
        elif math.isinf(val):
            return None  # Convert Infinity to null in JSON
        return val
    elif isinstance(obj, (float, int)) and not isinstance(obj, bool):
        # Handle Python native float/int that might be NaN or Inf
        if isinstance(obj, float):
            if math.isnan(obj):
                return None
            elif math.isinf(obj):
                return None
        return obj
    elif isinstance(obj, np.ndarray):
        return obj.tolist()
    elif isinstance(obj, np.bool_):
        return bool(obj)
    elif isinstance(obj, dict):
        return {key: convert_numpy_types(value) for key, value in obj.items()}
    elif isinstance(obj, list):
        return [convert_numpy_types(item) for item in obj]
    elif isinstance(obj, tuple):
        return tuple(convert_numpy_types(item) for item in obj)
    return obj

# Initialize FastAPI app
app = FastAPI(
    title="SSR Pipeline Demo API",
    description="API for SSR (Semantic Similarity Rating) Pipeline Demo Application",
    version="1.0.0"
)

# Enable CORS for React app
app.add_middleware(
    CORSMiddleware,
    allow_origins=[
        "http://localhost:3000",  # React dev server
        "http://localhost:3001",  # Alternative port
    ],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# ===================
# Pydantic Models
# ===================

class SurveyListItem(BaseModel):
    id: str
    name: str
    description: str
    num_questions: int
    num_persona_groups: int
    has_categories: bool

class PersonaGroupSchema(BaseModel):
    name: str
    description: str
    personas: List[str]
    target_demographics: Dict[str, List[str]]
    weight: float

class CategorySchema(BaseModel):
    id: str
    name: str
    description: str
    context: str

class QuestionSchema(BaseModel):
    id: str
    text: str
    type: str
    scale: Optional[Dict[int, str]] = None
    options: Optional[List[str]] = None
    category: Optional[str] = None
    categories_compared: Optional[List[str]] = None

class SurveySchema(BaseModel):
    name: str
    description: str
    context: str
    questions: List[QuestionSchema]
    persona_groups: List[PersonaGroupSchema]
    categories: Optional[List[CategorySchema]] = None
    demographics: List[str]
    sample_size: int

class CreateSurveyRequest(BaseModel):
    yaml_content: str
    filename: str

class GenerateProfilesRequest(BaseModel):
    survey_id: str
    num_profiles: int = Field(default=100, ge=10, le=500)

class GenerateResponsesRequest(BaseModel):
    survey_id: str
    profiles: List[Dict[str, Any]]
    llm_provider: str = Field(default="openai", pattern="^(openai|anthropic)$")
    model: str = "gpt-4"
    temperature: float = Field(default=0.7, ge=0.0, le=2.0)

class ApplySSRRequest(BaseModel):
    survey_id: str
    responses: List[Dict[str, Any]]
    temperature: float = Field(default=1.0, ge=0.1, le=5.0)
    normalize_method: str = Field(default="paper", pattern="^(paper|softmax|linear)$")

class RunSurveyRequest(BaseModel):
    survey_id: str
    num_profiles: int = Field(default=100, ge=10, le=500)
    llm_provider: str = Field(default="openai", pattern="^(openai|anthropic)$")
    model: str = "gpt-4"
    llm_temperature: float = Field(default=0.7, ge=0.0, le=2.0)
    ssr_temperature: float = Field(default=1.0, ge=0.1, le=5.0)
    normalize_method: str = Field(default="paper", pattern="^(paper|softmax|linear)$")
    seed: int = Field(default=100, ge=0, le=10000)

class CreateGroundTruthFromSSRRequest(BaseModel):
    survey_id: str
    name: str
    description: str
    num_profiles: int = Field(default=500, ge=10, le=2000)
    llm_provider: str = Field(default="openai", pattern="^(openai|anthropic)$")
    model: str = "gpt-4"
    llm_temperature: float = Field(default=0.7, ge=0.0, le=2.0)
    ssr_temperature: float = Field(default=1.0, ge=0.1, le=5.0)
    normalize_method: str = Field(default="paper", pattern="^(paper|softmax|linear)$")
    seed: int = Field(default=42, ge=0, le=10000)

class UploadGroundTruthRequest(BaseModel):
    survey_id: str
    name: str
    description: str
    distributions: Dict[str, Any]

# ===================
# Helper Functions
# ===================

def get_config_dir() -> Path:
    """Get path to config directory"""
    return Path(__file__).parent.parent / "config"

def get_results_dir() -> Path:
    """Get path to results directory"""
    results_dir = Path(__file__).parent.parent / "results"
    results_dir.mkdir(exist_ok=True)
    return results_dir

def get_ground_truths_dir() -> Path:
    """Get path to ground truths directory"""
    gt_dir = Path(__file__).parent.parent / "ground_truths"
    gt_dir.mkdir(exist_ok=True)
    return gt_dir

def get_experiments_dir() -> Path:
    """Get path to experiments directory"""
    exp_dir = Path(__file__).parent.parent / "experiments"
    exp_dir.mkdir(exist_ok=True)
    return exp_dir

def get_survey_path(survey_id: str) -> Path:
    """Get path to survey YAML file"""
    config_dir = get_config_dir()
    yaml_path = config_dir / f"{survey_id}.yaml"
    if not yaml_path.exists():
        raise HTTPException(status_code=404, detail=f"Survey '{survey_id}' not found")
    return yaml_path

def save_survey_run(run_result: dict) -> None:
    """Save survey run results to disk"""
    results_dir = get_results_dir()
    run_path = results_dir / f"{run_result['run_id']}.json"
    run_path.write_text(json.dumps(run_result, indent=2))

    # Update index
    index_path = results_dir / "runs_index.json"
    if index_path.exists():
        index = json.loads(index_path.read_text())
    else:
        index = []

    # Add metadata to index
    index.append({
        "run_id": run_result["run_id"],
        "survey_id": run_result["survey_id"],
        "timestamp": run_result.get("timestamp", datetime.now().isoformat()),
        "num_profiles": run_result["num_profiles"],
        "num_responses": run_result["num_responses"],
        "config": run_result["config"]
    })

    index_path.write_text(json.dumps(index, indent=2))

def aggregate_distributions_by_question(distributions: List[RatingDistribution], responses: List[Response]) -> Dict:
    """Aggregate individual respondent distributions into per-question averages"""
    aggregated = {}

    # Group by category and question
    by_question = defaultdict(list)
    for dist in distributions:
        # Find category from responses
        category = "general"
        for r in responses:
            if r.respondent_id == dist.respondent_id and r.question_id == dist.question_id:
                category = r.category or "general"
                break

        key = (category, dist.question_id)
        by_question[key].append(dist)

    # Calculate averages
    for (category, question_id), dists in by_question.items():
        if category not in aggregated:
            aggregated[category] = {}

        # Stack all probability distributions
        prob_arrays = [d.distribution for d in dists]

        # Average across respondents
        mean_probs = np.mean(prob_arrays, axis=0)
        std_probs = np.std(prob_arrays, axis=0)

        aggregated[category][question_id] = {
            "mean_probabilities": mean_probs.tolist(),
            "std_probabilities": std_probs.tolist(),
            "sample_size": len(dists),
            "mean_mode": int(np.mean([d.mode for d in dists])),
            "mean_expected_value": float(np.mean([d.expected_value for d in dists])),
            "mean_entropy": float(np.mean([d.entropy for d in dists]))
        }

    return aggregated

def save_ground_truth(ground_truth: dict) -> None:
    """Save ground truth to disk"""
    gt_dir = get_ground_truths_dir()
    gt_path = gt_dir / f"{ground_truth['id']}.json"
    gt_path.write_text(json.dumps(ground_truth, indent=2))

    # Update index
    index_path = gt_dir / "ground_truths_index.json"
    if index_path.exists():
        index = json.loads(index_path.read_text())
    else:
        index = []

    # Add metadata to index
    index.append({
        "id": ground_truth["id"],
        "name": ground_truth["name"],
        "survey_id": ground_truth["survey_id"],
        "source": ground_truth["source"],
        "created_at": ground_truth["created_at"],
        "generation_config": ground_truth.get("generation_config")
    })

    index_path.write_text(json.dumps(index, indent=2))

def load_survey(survey_id: str) -> Survey:
    """Load survey from YAML file"""
    survey_path = get_survey_path(survey_id)
    try:
        return Survey.from_config(str(survey_path))
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error loading survey: {str(e)}")

def survey_to_schema(survey: Survey) -> SurveySchema:
    """Convert Survey object to Pydantic schema"""
    questions_schema = []
    for q in survey.questions:
        q_dict = {
            "id": q.id,
            "text": q.text,
            "type": q.type,
            "category": q.category,
            "categories_compared": q.categories_compared
        }
        if q.scale:
            q_dict["scale"] = q.scale.labels
        if q.options:
            q_dict["options"] = q.options
        questions_schema.append(QuestionSchema(**q_dict))

    persona_groups_schema = []
    for pg in survey.persona_groups:
        persona_groups_schema.append(PersonaGroupSchema(
            name=pg.name,
            description=pg.description,
            personas=pg.personas,
            target_demographics=pg.target_demographics,
            weight=pg.weight
        ))

    categories_schema = None
    if survey.categories:
        categories_schema = [
            CategorySchema(
                id=c.id,
                name=c.name,
                description=c.description,
                context=c.context
            ) for c in survey.categories
        ]

    return SurveySchema(
        name=survey.name,
        description=survey.description,
        context=survey.context,
        questions=questions_schema,
        persona_groups=persona_groups_schema,
        categories=categories_schema,
        demographics=survey.demographics,
        sample_size=survey.sample_size
    )

# ===================
# API Endpoints
# ===================

@app.get("/")
async def root():
    """Root endpoint"""
    return {
        "message": "SSR Pipeline Demo API",
        "version": "1.0.0",
        "docs": "/docs"
    }

@app.get("/api/surveys", response_model=List[SurveyListItem])
async def get_surveys():
    """List all available surveys"""
    config_dir = get_config_dir()
    if not config_dir.exists():
        return []

    surveys = []
    for yaml_file in config_dir.glob("*.yaml"):
        try:
            survey = Survey.from_config(str(yaml_file))
            surveys.append(SurveyListItem(
                id=yaml_file.stem,
                name=survey.name,
                description=survey.description,
                num_questions=len(survey.questions),
                num_persona_groups=len(survey.persona_groups),
                has_categories=survey.has_categories()
            ))
        except Exception as e:
            # Skip invalid survey files
            continue

    return surveys

@app.get("/api/surveys/{survey_id}", response_model=SurveySchema)
async def get_survey(survey_id: str):
    """Get survey configuration"""
    survey = load_survey(survey_id)
    return survey_to_schema(survey)

@app.post("/api/surveys")
async def create_survey(request: CreateSurveyRequest):
    """Create new survey from YAML content"""
    config_dir = get_config_dir()
    config_dir.mkdir(exist_ok=True)

    # Validate filename
    if not request.filename.endswith('.yaml'):
        request.filename += '.yaml'

    survey_path = config_dir / request.filename
    if survey_path.exists():
        raise HTTPException(status_code=400, detail="Survey with this name already exists")

    # Validate YAML by trying to load it
    try:
        import yaml
        survey_config = yaml.safe_load(request.yaml_content)
        if 'survey' not in survey_config:
            raise ValueError("YAML must contain 'survey' key")
    except Exception as e:
        raise HTTPException(status_code=400, detail=f"Invalid YAML: {str(e)}")

    # Save survey
    survey_path.write_text(request.yaml_content)

    return {
        "survey_id": survey_path.stem,
        "status": "created",
        "path": str(survey_path)
    }

@app.put("/api/surveys/{survey_id}")
async def update_survey(survey_id: str, request: CreateSurveyRequest):
    """Update existing survey with new YAML content"""
    survey_path = get_survey_path(survey_id)

    # Validate YAML by trying to load it
    try:
        import yaml
        survey_config = yaml.safe_load(request.yaml_content)
        if 'survey' not in survey_config:
            raise ValueError("YAML must contain 'survey' key")
    except Exception as e:
        raise HTTPException(status_code=400, detail=f"Invalid YAML: {str(e)}")

    # Save updated survey
    survey_path.write_text(request.yaml_content)

    return {
        "survey_id": survey_id,
        "status": "updated",
        "path": str(survey_path)
    }

@app.delete("/api/surveys/{survey_id}")
async def delete_survey(survey_id: str):
    """Delete a survey"""
    survey_path = get_survey_path(survey_id)

    try:
        survey_path.unlink()
        return {
            "survey_id": survey_id,
            "status": "deleted"
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error deleting survey: {str(e)}")

@app.post("/api/generate-profiles")
async def generate_profiles(request: GenerateProfilesRequest):
    """Generate respondent profiles from persona groups"""
    survey = load_survey(request.survey_id)

    try:
        profiles = generate_diverse_profiles(
            n_profiles=request.num_profiles,
            persona_groups=survey.persona_groups
        )

        return {
            "survey_id": request.survey_id,
            "num_profiles": len(profiles),
            "profiles": [p.to_dict() for p in profiles]
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error generating profiles: {str(e)}")

@app.post("/api/generate-responses")
async def generate_responses(request: GenerateResponsesRequest):
    """Generate LLM text responses"""
    survey = load_survey(request.survey_id)

    # Convert dict profiles back to RespondentProfile objects
    profiles = []
    for p_dict in request.profiles:
        profile = RespondentProfile(
            description=p_dict.get('description', ''),
            respondent_id=p_dict.get('respondent_id'),
            gender=p_dict.get('gender', 'Unknown'),
            age_group=p_dict.get('age_group', 'Unknown'),
            persona_group=p_dict.get('persona_group', 'General'),
            occupation=p_dict.get('occupation', 'Unknown')
        )
        profiles.append(profile)

    try:
        # Initialize LLM client
        llm_client = LLMClient(
            provider=request.llm_provider,
            model=request.model,
            temperature=request.llm_temperature
        )

        # Generate responses (using concurrent version for better performance)
        responses = llm_client.generate_responses_concurrent(survey, profiles, max_concurrent=20)

        return {
            "survey_id": request.survey_id,
            "num_responses": len(responses),
            "responses": [
                {
                    "respondent_id": r.respondent_id,
                    "question_id": r.question_id,
                    "text_response": r.text_response,
                    "respondent_profile": r.respondent_profile,
                    "category": r.category
                } for r in responses
            ]
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error generating responses: {str(e)}")

@app.post("/api/apply-ssr")
async def apply_ssr(request: ApplySSRRequest):
    """Apply SSR to convert text responses to probability distributions"""
    survey = load_survey(request.survey_id)

    # Convert dict responses back to Response objects
    responses = []
    for r_dict in request.responses:
        response = Response(
            respondent_id=r_dict['respondent_id'],
            question_id=r_dict['question_id'],
            text_response=r_dict['text_response'],
            respondent_profile=r_dict['respondent_profile'],
            category=r_dict.get('category')
        )
        responses.append(response)

    try:
        # Initialize SSR rater
        rater = SemanticSimilarityRater(
            temperature=request.temperature,
            normalize_method=request.normalize_method
        )

        # Apply SSR
        distributions = rater.rate_responses(responses, survey, show_progress=False)

        # Organize by category and question
        organized_distributions = {}
        for dist in distributions:
            # Get category
            category = "general"
            for r in responses:
                if r.respondent_id == dist.respondent_id and r.question_id == dist.question_id:
                    category = r.category or "general"
                    break

            # Get respondent profile
            profile = {}
            for r in responses:
                if r.respondent_id == dist.respondent_id and r.question_id == dist.question_id:
                    profile = r.respondent_profile
                    break

            if category not in organized_distributions:
                organized_distributions[category] = {}

            if dist.question_id not in organized_distributions[category]:
                organized_distributions[category][dist.question_id] = {}

            organized_distributions[category][dist.question_id][dist.respondent_id] = {
                "probabilities": dist.distribution.tolist(),
                "mode": int(dist.mode),
                "expected_value": float(dist.expected_value),
                "entropy": float(dist.entropy),
                "text_response": dist.text_response,
                "gender": profile.get('gender', 'Unknown'),
                "age_group": profile.get('age_group', 'Unknown'),
                "persona_group": profile.get('persona_group', 'General'),
                "occupation": profile.get('occupation', 'Unknown')
            }

        return {
            "survey_id": request.survey_id,
            "num_distributions": len(distributions),
            "distributions": organized_distributions
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error applying SSR: {str(e)}")

@app.post("/api/run-survey-stream")
async def run_survey_stream(request: RunSurveyRequest):
    """Run complete survey pipeline with streaming progress updates"""

    async def generate():
        try:
            survey = load_survey(request.survey_id)

            # Send initial status
            msg_data = {'status': 'starting', 'message': f'Starting survey: {survey.name}', 'progress': 0}
            yield f"data: {json.dumps(msg_data)}\n\n"
            await asyncio.sleep(0.1)

            # Step 1: Generate profiles
            msg_data = {'status': 'running', 'message': f'Step 1/3: Generating {request.num_profiles} profiles...', 'progress': 10}
            yield f"data: {json.dumps(msg_data)}\n\n"
            await asyncio.sleep(0.1)

            profiles = generate_diverse_profiles(
                n_profiles=request.num_profiles,
                persona_groups=survey.persona_groups
            )

            msg_data = {'status': 'running', 'message': f'Generated {len(profiles)} profiles', 'progress': 25}
            yield f"data: {json.dumps(msg_data)}\n\n"
            await asyncio.sleep(0.1)

            # Step 2: Generate LLM responses
            num_api_calls = len(profiles) * len(survey.questions)
            num_batches = (num_api_calls + 49) // 50  # Calculate number of batches
            concurrent_limit = 20
            num_batches = (num_api_calls + concurrent_limit - 1) // concurrent_limit
            msg_data = {'status': 'running', 'message': f'Step 2/3: Generating LLM responses ({num_api_calls} API calls in ~{num_batches} batches of {concurrent_limit})...', 'progress': 30}
            yield f"data: {json.dumps(msg_data)}\n\n"
            await asyncio.sleep(0.1)

            print(f"\n=== Starting LLM Response Generation ===")
            print(f"Total API calls: {num_api_calls}")
            print(f"Concurrent limit: {concurrent_limit}")
            print(f"Expected batches: ~{num_batches}")
            print(f"Estimated time: ~{num_batches * 3} seconds (assuming 3s per batch)\n")

            llm_client = LLMClient(
                provider=request.llm_provider,
                model=request.model,
                temperature=request.llm_temperature
            )

            import time
            start_time = time.time()
            responses = llm_client.generate_responses_concurrent(survey, profiles, max_concurrent=concurrent_limit)
            elapsed = time.time() - start_time

            print(f"\n=== LLM Response Generation Complete ===")
            print(f"Total responses: {len(responses)}")
            print(f"Time taken: {elapsed:.2f} seconds")
            print(f"Average per call: {elapsed/len(responses):.2f} seconds\n")

            msg_data = {'status': 'running', 'message': f'Generated {len(responses)} responses in {elapsed:.1f}s', 'progress': 60}
            yield f"data: {json.dumps(msg_data)}\n\n"
            await asyncio.sleep(0.1)

            # Step 3: Apply SSR
            msg_data = {'status': 'running', 'message': f'Step 3/3: Applying SSR to {len(responses)} responses...', 'progress': 65}
            yield f"data: {json.dumps(msg_data)}\n\n"
            await asyncio.sleep(0.1)

            print(f"\n=== Starting SSR Application ===")
            print(f"Number of responses to process: {len(responses)}")
            print(f"Temperature: {request.ssr_temperature}")
            print(f"Normalize method: {request.normalize_method}\n")

            ssr_start_time = time.time()
            rater = SemanticSimilarityRater(
                temperature=request.ssr_temperature,
                normalize_method=request.normalize_method
            )
            distributions = rater.rate_responses(responses, survey, show_progress=False)
            ssr_elapsed = time.time() - ssr_start_time

            print(f"\n=== SSR Application Complete ===")
            print(f"Distributions generated: {len(distributions)}")
            print(f"Time taken: {ssr_elapsed:.2f} seconds")
            print(f"Average per response: {ssr_elapsed/len(responses):.2f} seconds\n")

            msg_data = {'status': 'running', 'message': f'Generated {len(distributions)} distributions in {ssr_elapsed:.1f}s', 'progress': 90}
            yield f"data: {json.dumps(msg_data)}\n\n"
            await asyncio.sleep(0.1)

            # Organize results
            organized_distributions = {}
            for dist in distributions:
                category = "general"
                profile = {}
                for r in responses:
                    if r.respondent_id == dist.respondent_id and r.question_id == dist.question_id:
                        category = r.category or "general"
                        profile = r.respondent_profile
                        break

                if category not in organized_distributions:
                    organized_distributions[category] = {}

                if dist.question_id not in organized_distributions[category]:
                    organized_distributions[category][dist.question_id] = {}

                # json.dumps() will handle all escaping automatically
                organized_distributions[category][dist.question_id][dist.respondent_id] = {
                    "probabilities": dist.distribution.tolist(),
                    "mode": int(dist.mode),
                    "expected_value": float(dist.expected_value),
                    "entropy": float(dist.entropy),
                    "text_response": dist.text_response or "",
                    "gender": profile.get('gender', 'Unknown'),
                    "age_group": profile.get('age_group', 'Unknown'),
                    "persona_group": profile.get('persona_group', 'General'),
                    "occupation": profile.get('occupation', 'Unknown')
                }

            run_id = f"run_{datetime.now().strftime('%Y%m%d_%H%M%S')}"

            result = {
                "run_id": run_id,
                "survey_id": request.survey_id,
                "survey_name": survey.name,
                "timestamp": datetime.now().isoformat(),
                "num_profiles": len(profiles),
                "num_responses": len(responses),
                "num_distributions": len(distributions),
                "distributions": organized_distributions,
                "config": {
                    "llm_provider": request.llm_provider,
                    "model": request.model,
                    "llm_temperature": request.llm_temperature,
                    "ssr_temperature": request.ssr_temperature,
                    "normalize_method": request.normalize_method,
                    "seed": request.seed
                }
            }

            # Save run results to disk
            save_survey_run(result)

            # Send completion with results
            print(f"\n=== Preparing Completion Message ===")
            print(f"Result size - Profiles: {len(profiles)}, Responses: {len(responses)}, Distributions: {len(distributions)}")

            completion_data = {
                'status': 'complete',
                'message': 'Survey complete!',
                'progress': 100,
                'result': result
            }

            print("Serializing completion data to JSON...")
            try:
                json_str = json.dumps(completion_data)
                print(f"JSON serialization successful! Length: {len(json_str)} characters")
                yield f"data: {json_str}\n\n"
                print("Completion message sent successfully")
            except Exception as json_error:
                print(f"ERROR during JSON serialization: {type(json_error).__name__}: {str(json_error)}")
                raise

        except Exception as e:
            error_data = {
                'status': 'error',
                'message': f'Error: {str(e)}',
                'progress': 0
            }
            yield f"data: {json.dumps(error_data)}\n\n"

    return StreamingResponse(generate(), media_type="text/event-stream")

@app.post("/api/run-survey")
async def run_survey(request: RunSurveyRequest):
    """Run complete survey pipeline (profiles → responses → SSR)"""
    survey = load_survey(request.survey_id)

    try:
        # Step 1: Generate profiles
        profiles = generate_diverse_profiles(
            n_profiles=request.num_profiles,
            persona_groups=survey.persona_groups
        )

        # Step 2: Generate LLM responses
        llm_client = LLMClient(
            provider=request.llm_provider,
            model=request.model,
            temperature=request.llm_temperature
        )
        responses = llm_client.generate_responses_concurrent(survey, profiles, max_concurrent=20)

        # Step 3: Apply SSR
        rater = SemanticSimilarityRater(
            temperature=request.ssr_temperature,
            normalize_method=request.normalize_method
        )
        distributions = rater.rate_responses(responses, survey, show_progress=False)

        # Organize results
        organized_distributions = {}
        for dist in distributions:
            # Get category and profile
            category = "general"
            profile = {}
            for r in responses:
                if r.respondent_id == dist.respondent_id and r.question_id == dist.question_id:
                    category = r.category or "general"
                    profile = r.respondent_profile
                    break

            if category not in organized_distributions:
                organized_distributions[category] = {}

            if dist.question_id not in organized_distributions[category]:
                organized_distributions[category][dist.question_id] = {}

            organized_distributions[category][dist.question_id][dist.respondent_id] = {
                "probabilities": dist.distribution.tolist(),
                "mode": int(dist.mode),
                "expected_value": float(dist.expected_value),
                "entropy": float(dist.entropy),
                "text_response": dist.text_response,
                "gender": profile.get('gender', 'Unknown'),
                "age_group": profile.get('age_group', 'Unknown'),
                "persona_group": profile.get('persona_group', 'General'),
                "occupation": profile.get('occupation', 'Unknown')
            }

        # Generate run ID
        run_id = f"run_{datetime.now().strftime('%Y%m%d_%H%M%S')}"

        logger.info(f"Survey run complete! Run ID: {run_id}")
        logger.info(f"  - Profiles: {len(profiles)}")
        logger.info(f"  - Responses: {len(responses)}")
        logger.info(f"  - Distributions: {len(distributions)}")

        response_data = {
            "run_id": run_id,
            "survey_id": request.survey_id,
            "num_profiles": len(profiles),
            "num_responses": len(responses),
            "num_distributions": len(distributions),
            "distributions": organized_distributions,
            "config": {
                "llm_provider": request.llm_provider,
                "model": request.model,
                "llm_temperature": request.llm_temperature,
                "ssr_temperature": request.ssr_temperature,
                "normalize_method": request.normalize_method,
                "seed": request.seed
            }
        }

        # Save survey run to file
        results_dir = get_results_dir()
        results_dir.mkdir(exist_ok=True)
        run_path = results_dir / f"{run_id}.json"
        run_path.write_text(json.dumps(response_data, indent=2))

        # Update runs index
        index_path = results_dir / "runs_index.json"
        index = []
        if index_path.exists():
            index = json.loads(index_path.read_text())

        # Add new run to index
        index.append({
            "run_id": run_id,
            "survey_id": request.survey_id,
            "timestamp": datetime.now().isoformat(),
            "num_profiles": len(profiles),
            "num_responses": len(responses),
            "config": response_data["config"]
        })
        index_path.write_text(json.dumps(index, indent=2))

        logger.info(f"Survey run saved: {run_path}")

        # Convert numpy types to Python native types for JSON serialization
        return convert_numpy_types(response_data)
    except Exception as e:
        import traceback
        logger.error(f"Error running survey: {str(e)}")
        logger.error(traceback.format_exc())
        raise HTTPException(status_code=500, detail=f"Error running survey: {str(e)}")

# ===================
# Survey History Endpoints
# ===================

@app.get("/api/survey-runs")
async def get_survey_runs(survey_id: Optional[str] = None):
    """List all survey runs with optional filtering by survey_id"""
    results_dir = get_results_dir()
    index_path = results_dir / "runs_index.json"

    if not index_path.exists():
        return []

    index = json.loads(index_path.read_text())

    # Filter by survey_id if provided
    if survey_id:
        index = [run for run in index if run["survey_id"] == survey_id]

    # Sort by timestamp descending (newest first)
    index.sort(key=lambda x: x.get("timestamp", ""), reverse=True)

    return index

@app.get("/api/survey-runs/{run_id}")
async def get_survey_run(run_id: str):
    """Get specific survey run details"""
    results_dir = get_results_dir()
    run_path = results_dir / f"{run_id}.json"

    if not run_path.exists():
        raise HTTPException(status_code=404, detail=f"Run '{run_id}' not found")

    return json.loads(run_path.read_text())

@app.delete("/api/survey-runs/{run_id}")
async def delete_survey_run(run_id: str):
    """Delete a survey run"""
    results_dir = get_results_dir()
    run_path = results_dir / f"{run_id}.json"

    if not run_path.exists():
        raise HTTPException(status_code=404, detail=f"Run '{run_id}' not found")

    # Remove from index
    index_path = results_dir / "runs_index.json"
    if index_path.exists():
        index = json.loads(index_path.read_text())
        index = [run for run in index if run["run_id"] != run_id]
        index_path.write_text(json.dumps(index, indent=2))

    # Delete file
    run_path.unlink()

    return {"run_id": run_id, "status": "deleted"}

# ===================
# Ground Truth Endpoints
# ===================

@app.post("/api/ground-truths/from-ssr")
async def create_ground_truth_from_ssr(request: CreateGroundTruthFromSSRRequest):
    """Generate ground truth by running full SSR pipeline with personas"""
    survey = load_survey(request.survey_id)

    try:
        # Step 1: Generate profiles from persona groups
        profiles = generate_diverse_profiles(
            n_profiles=request.num_profiles,
            persona_groups=survey.persona_groups
        )

        # Step 2: Generate LLM responses
        llm_client = LLMClient(
            provider=request.llm_provider,
            model=request.model,
            temperature=request.llm_temperature
        )
        responses = llm_client.generate_responses_concurrent(
            survey=survey,
            respondent_profiles=profiles,
            max_concurrent=20
        )

        # Step 3: Apply SSR
        rater = SemanticSimilarityRater(
            temperature=request.ssr_temperature,
            normalize_method=request.normalize_method
        )
        distributions = rater.rate_responses(responses, survey, show_progress=False)

        # Step 4: Aggregate distributions per question
        aggregated_distributions = aggregate_distributions_by_question(distributions, responses)

        # Step 5: Organize raw distributions
        organized_distributions = {}
        for dist in distributions:
            category = "general"
            profile = {}
            for r in responses:
                if r.respondent_id == dist.respondent_id and r.question_id == dist.question_id:
                    category = r.category or "general"
                    profile = r.respondent_profile
                    break

            if category not in organized_distributions:
                organized_distributions[category] = {}
            if dist.question_id not in organized_distributions[category]:
                organized_distributions[category][dist.question_id] = {}

            organized_distributions[category][dist.question_id][dist.respondent_id] = {
                "probabilities": dist.distribution.tolist(),
                "mode": int(dist.mode),
                "expected_value": float(dist.expected_value),
                "entropy": float(dist.entropy),
                "text_response": dist.text_response or "",
                "gender": profile.get('gender', 'Unknown'),
                "age_group": profile.get('age_group', 'Unknown'),
                "persona_group": profile.get('persona_group', 'General'),
                "occupation": profile.get('occupation', 'Unknown')
            }

        # Calculate persona distribution
        persona_distribution = {}
        for profile in profiles:
            pg = profile.persona_group
            persona_distribution[pg] = persona_distribution.get(pg, 0) + 1

        # Step 6: Save as ground truth
        ground_truth_id = f"gt_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
        ground_truth = {
            "id": ground_truth_id,
            "name": request.name,
            "description": request.description,
            "survey_id": request.survey_id,
            "survey_name": survey.name,
            "source": "ssr_generated",
            "created_at": datetime.now().isoformat(),
            "generation_config": {
                "num_profiles": request.num_profiles,
                "llm_provider": request.llm_provider,
                "model": request.model,
                "llm_temperature": request.llm_temperature,
                "ssr_temperature": request.ssr_temperature,
                "normalize_method": request.normalize_method,
                "seed": request.seed,
                "persona_groups": [pg.name for pg in survey.persona_groups],
                "persona_distribution": persona_distribution
            },
            "aggregated_distributions": aggregated_distributions,
            "raw_distributions": organized_distributions
        }

        save_ground_truth(ground_truth)

        return {
            "id": ground_truth_id,
            "status": "created",
            "name": request.name,
            "num_profiles": request.num_profiles,
            "num_responses": len(responses)
        }

    except Exception as e:
        import traceback
        error_detail = f"Error creating ground truth: {str(e)}\n{traceback.format_exc()}"
        print(error_detail)  # Log to console
        raise HTTPException(status_code=500, detail=f"Error creating ground truth: {str(e)}")

@app.post("/api/ground-truths/from-upload")
async def create_ground_truth_from_upload(request: UploadGroundTruthRequest):
    """Create ground truth from uploaded data"""
    survey = load_survey(request.survey_id)

    ground_truth_id = f"gt_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
    ground_truth = {
        "id": ground_truth_id,
        "name": request.name,
        "description": request.description,
        "survey_id": request.survey_id,
        "survey_name": survey.name,
        "source": "uploaded",
        "created_at": datetime.now().isoformat(),
        "aggregated_distributions": request.distributions
    }

    save_ground_truth(ground_truth)

    return {
        "id": ground_truth_id,
        "status": "created",
        "name": request.name
    }

@app.get("/api/ground-truths")
async def get_ground_truths(survey_id: Optional[str] = None):
    """List all ground truths with optional filtering by survey_id"""
    gt_dir = get_ground_truths_dir()
    index_path = gt_dir / "ground_truths_index.json"

    if not index_path.exists():
        return []

    index = json.loads(index_path.read_text())

    # Filter by survey_id if provided
    if survey_id:
        index = [gt for gt in index if gt["survey_id"] == survey_id]

    # Sort by created_at descending (newest first)
    index.sort(key=lambda x: x.get("created_at", ""), reverse=True)

    return index

@app.get("/api/ground-truths/{gt_id}")
async def get_ground_truth(gt_id: str):
    """Get specific ground truth details"""
    gt_dir = get_ground_truths_dir()
    gt_path = gt_dir / f"{gt_id}.json"

    if not gt_path.exists():
        raise HTTPException(status_code=404, detail=f"Ground truth '{gt_id}' not found")

    return json.loads(gt_path.read_text())

@app.delete("/api/ground-truths/{gt_id}")
async def delete_ground_truth(gt_id: str):
    """Delete a ground truth"""
    gt_dir = get_ground_truths_dir()
    gt_path = gt_dir / f"{gt_id}.json"

    if not gt_path.exists():
        raise HTTPException(status_code=404, detail=f"Ground truth '{gt_id}' not found")

    # Remove from index
    index_path = gt_dir / "ground_truths_index.json"
    if index_path.exists():
        index = json.loads(index_path.read_text())
        index = [gt for gt in index if gt["id"] != gt_id]
        index_path.write_text(json.dumps(index, indent=2))

    # Delete file
    gt_path.unlink()

    return {"id": gt_id, "status": "deleted"}

@app.post("/api/ground-truths/compare")
async def compare_run_to_ground_truth(run_id: str, ground_truth_id: str):
    """Compare a survey run against ground truth"""
    # Load ground truth
    gt_dir = get_ground_truths_dir()
    gt_path = gt_dir / f"{ground_truth_id}.json"
    if not gt_path.exists():
        raise HTTPException(status_code=404, detail=f"Ground truth '{ground_truth_id}' not found")
    ground_truth = json.loads(gt_path.read_text())

    # Load survey run
    results_dir = get_results_dir()
    run_path = results_dir / f"{run_id}.json"
    if not run_path.exists():
        raise HTTPException(status_code=404, detail=f"Run '{run_id}' not found")
    test_run = json.loads(run_path.read_text())

    # Verify they're for the same survey
    if ground_truth["survey_id"] != test_run["survey_id"]:
        raise HTTPException(
            status_code=400,
            detail="Ground truth and run are for different surveys"
        )

    try:
        # Perform comparison
        comparison_results = compare_survey_runs(ground_truth, test_run)

        response_data = {
            "run_id": run_id,
            "ground_truth_id": ground_truth_id,
            "survey_id": test_run["survey_id"],
            "comparison": comparison_results,
            # Include distributions for visualization
            "test_run_distributions": test_run.get("distributions", {}),
            "ground_truth_distributions": ground_truth.get("aggregated_distributions", {})
        }

        # Convert numpy types to Python native types for JSON serialization
        return convert_numpy_types(response_data)
    except Exception as e:
        import traceback
        logger.error(f"Error comparing: {str(e)}")
        logger.error(traceback.format_exc())
        raise HTTPException(status_code=500, detail=f"Error comparing: {str(e)}")

# WebSocket endpoint for real-time progress (future enhancement)
@app.websocket("/ws/run-survey")
async def websocket_run_survey(websocket: WebSocket):
    """WebSocket endpoint for real-time survey execution progress"""
    await websocket.accept()
    try:
        # Receive request
        data = await websocket.receive_text()
        request_data = json.loads(data)

        # Send progress updates as survey runs
        await websocket.send_json({"status": "starting", "progress": 0})

        # TODO: Implement progress tracking
        # For now, just acknowledge
        await websocket.send_json({"status": "complete", "progress": 100})

    except WebSocketDisconnect:
        pass
    except Exception as e:
        await websocket.send_json({"status": "error", "message": str(e)})

# Run with: uvicorn main:app --reload --port 8000
if __name__ == "__main__":
    import uvicorn
    uvicorn.run("main:app", host="0.0.0.0", port=8000, reload=True)
