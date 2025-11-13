# SSR Pipeline Demo Backend

FastAPI backend server for the SSR Pipeline Demo Application.

## Installation

### Using uv (Recommended)

[uv](https://github.com/astral-sh/uv) is a fast Python package installer and resolver.

1. Install uv if you haven't already:
```bash
curl -LsSf https://astral.sh/uv/install.sh | sh
# or on macOS:
brew install uv
```

2. Create virtual environment and install dependencies:
```bash
cd backend
uv venv
source .venv/bin/activate  # On Windows: .venv\Scripts\activate
uv pip install -e .
```

Or use uv sync for development:
```bash
uv sync
source .venv/bin/activate
```

### Using pip (Traditional)

```bash
cd backend
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate
pip install -r requirements.txt
```

## Environment Variables

Create a `.env` file in the backend directory:

```bash
OPENAI_API_KEY=your-openai-api-key
ANTHROPIC_API_KEY=your-anthropic-api-key
```

Or export them:
```bash
export OPENAI_API_KEY="your-api-key"
export ANTHROPIC_API_KEY="your-api-key"
```

## Running the Server

```bash
python main.py
```

Or with uvicorn directly:
```bash
uvicorn main:app --reload --port 8000
```

The API will be available at:
- API: http://localhost:8000
- Interactive docs: http://localhost:8000/docs
- OpenAPI spec: http://localhost:8000/openapi.json

## API Endpoints

### Survey Management
- `GET /api/surveys` - List all surveys
- `GET /api/surveys/{survey_id}` - Get survey details
- `POST /api/surveys` - Create new survey from YAML

### Pipeline Operations
- `POST /api/generate-profiles` - Generate respondent profiles from persona groups
- `POST /api/generate-responses` - Generate LLM text responses
- `POST /api/apply-ssr` - Apply SSR to convert text to distributions
- `POST /api/run-survey` - Run complete pipeline (profiles → responses → SSR)

### Health Check
- `GET /` - API status and version

## Development

### Install with dev dependencies:
```bash
uv pip install -e ".[dev]"
```

### Run tests:
```bash
pytest
```

## Project Structure

```
backend/
├── main.py              # FastAPI application
├── pyproject.toml       # Project configuration (uv)
├── requirements.txt     # Pip fallback
├── .python-version      # Python version for uv
└── README.md
```

## Dependencies

- FastAPI - Web framework
- Uvicorn - ASGI server
- Pydantic - Data validation
- OpenAI - GPT models
- Anthropic - Claude models
- PyYAML - YAML parsing
- Pandas/Numpy - Data processing

The backend reuses existing SSR pipeline modules from `src/` directory.
