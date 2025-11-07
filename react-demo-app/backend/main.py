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
from pathlib import Path
from datetime import datetime
from dotenv import load_dotenv
import time

# Load environment variables from .env file
load_dotenv()

# Add parent directory to path to access src module
parent_path = Path(__file__).parent.parent.parent
sys.path.insert(0, str(parent_path))

from src.survey import Survey, PersonaGroup, Question, Category
from src.llm_client import LLMClient, generate_diverse_profiles, RespondentProfile, Response
from src.ssr_model import SemanticSimilarityRater, RatingDistribution

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

# ===================
# Helper Functions
# ===================

def get_config_dir() -> Path:
    """Get path to config directory"""
    return Path(__file__).parent.parent.parent / "config"

def get_survey_path(survey_id: str) -> Path:
    """Get path to survey YAML file"""
    config_dir = get_config_dir()
    yaml_path = config_dir / f"{survey_id}.yaml"
    if not yaml_path.exists():
        raise HTTPException(status_code=404, detail=f"Survey '{survey_id}' not found")
    return yaml_path

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
        responses = llm_client.generate_responses_concurrent(survey, profiles, max_concurrent=10)

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
            yield f"data: {json.dumps({'status': 'starting', 'message': f'Starting survey: {survey.name}', 'progress': 0})}\n\n"
            await asyncio.sleep(0.1)

            # Step 1: Generate profiles
            yield f"data: {json.dumps({'status': 'running', 'message': f'Step 1/3: Generating {request.num_profiles} profiles...', 'progress': 10})}\n\n"
            await asyncio.sleep(0.1)

            profiles = generate_diverse_profiles(
                n_profiles=request.num_profiles,
                persona_groups=survey.persona_groups
            )

            yield f"data: {json.dumps({'status': 'running', 'message': f'✓ Generated {len(profiles)} profiles', 'progress': 25})}\n\n"
            await asyncio.sleep(0.1)

            # Step 2: Generate LLM responses
            num_api_calls = len(profiles) * len(survey.questions)
            yield f"data: {json.dumps({'status': 'running', 'message': f'Step 2/3: Generating LLM responses ({num_api_calls} API calls)...', 'progress': 30})}\n\n"
            await asyncio.sleep(0.1)

            llm_client = LLMClient(
                provider=request.llm_provider,
                model=request.model,
                temperature=request.llm_temperature
            )
            responses = llm_client.generate_responses_concurrent(survey, profiles, max_concurrent=10)

            yield f"data: {json.dumps({'status': 'running', 'message': f'✓ Generated {len(responses)} responses', 'progress': 60})}\n\n"
            await asyncio.sleep(0.1)

            # Step 3: Apply SSR
            yield f"data: {json.dumps({'status': 'running', 'message': 'Step 3/3: Applying SSR to responses...', 'progress': 65})}\n\n"
            await asyncio.sleep(0.1)

            rater = SemanticSimilarityRater(
                temperature=request.ssr_temperature,
                normalize_method=request.normalize_method
            )
            distributions = rater.rate_responses(responses, survey, show_progress=False)

            yield f"data: {json.dumps({'status': 'running', 'message': f'✓ Generated {len(distributions)} distributions', 'progress': 90})}\n\n"
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

            run_id = f"run_{datetime.now().strftime('%Y%m%d_%H%M%S')}"

            result = {
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

            # Send completion with results
            yield f"data: {json.dumps({'status': 'complete', 'message': '✓ Survey complete!', 'progress': 100, 'result': result})}\n\n"

        except Exception as e:
            yield f"data: {json.dumps({'status': 'error', 'message': f'Error: {str(e)}', 'progress': 0})}\n\n"

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
        responses = llm_client.generate_responses_concurrent(survey, profiles, max_concurrent=10)

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

        logger.info(f"✓ Survey run complete! Run ID: {run_id}")
        logger.info(f"  - Profiles: {len(profiles)}")
        logger.info(f"  - Responses: {len(responses)}")
        logger.info(f"  - Distributions: {len(distributions)}")

        return {
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
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error running survey: {str(e)}")

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
