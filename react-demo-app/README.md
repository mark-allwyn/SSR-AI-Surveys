# SSR Pipeline Demo Application

A modern React-based demonstration application for the Semantic Similarity Rating (SSR) Pipeline. This application provides an interactive interface to create surveys, generate LLM responses, and apply SSR to convert text into probability distributions.

## Features

- **Survey Builder**: Create and configure surveys with questions, persona groups, and categories
- **Survey Preview**: View and explore existing survey configurations
- **Survey Runner**: Execute the complete SSR pipeline (profile generation → LLM responses → SSR distributions)
- **Response Viewer**: View and export LLM response datasets with probability distributions

## Architecture

### Backend (FastAPI)
- REST API for survey management and pipeline execution
- Reuses existing SSR pipeline Python modules
- CORS-enabled for local development
- Located in `backend/`

### Frontend (React + TypeScript)
- Modern React 18 with TypeScript
- Material-UI for components
- React Query for server state management
- React Router for navigation
- Recharts for data visualization
- Located in `frontend/`

## Prerequisites

- Python 3.8+
- Node.js 16+
- npm or yarn
- OpenAI API key and/or Anthropic API key

## Installation

### Backend Setup

#### Using uv (Recommended - Fast!)

1. Install uv if you haven't already:
```bash
curl -LsSf https://astral.sh/uv/install.sh | sh
# or on macOS: brew install uv
```

2. Navigate to backend directory:
```bash
cd backend
```

3. Create virtual environment and install dependencies:
```bash
uv venv
source .venv/bin/activate  # On Windows: .venv\Scripts\activate
uv pip install -e .
```

Or use uv sync:
```bash
uv sync
source .venv/bin/activate
```

4. Set environment variables:
```bash
export OPENAI_API_KEY="your-api-key"
export ANTHROPIC_API_KEY="your-api-key"
```

5. Start the backend server:
```bash
python main.py
```

#### Using pip (Traditional)

1. Navigate to backend directory:
```bash
cd backend
```

2. Create and activate virtual environment:
```bash
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate
```

3. Install dependencies:
```bash
pip install -r requirements.txt
```

4. Set environment variables:
```bash
export OPENAI_API_KEY="your-api-key"
export ANTHROPIC_API_KEY="your-api-key"
```

5. Start the backend server:
```bash
python main.py
```

The API will be available at `http://localhost:8000`
API documentation at `http://localhost:8000/docs`

### Frontend Setup

1. Navigate to frontend directory:
```bash
cd frontend
```

2. Install dependencies:
```bash
npm install
```

3. Create `.env` file:
```bash
cp .env.example .env
```

4. Start the development server:
```bash
npm start
```

The app will open at `http://localhost:3000`

## Usage

### Creating a Survey

1. Navigate to **Survey Builder** from the home page
2. Fill in basic survey information (name, description, context)
3. Add categories (optional) for multi-product comparisons
4. Add questions with different question types:
   - Yes/No
   - Likert scales (5-point, 7-point)
   - Multiple choice
   - Preference scales (comparative)
5. Create persona groups with:
   - Persona descriptions
   - Target demographics (gender, age, occupation)
   - Sampling weights
6. Switch to YAML Preview tab to view generated configuration
7. Enter a filename and click "Save Survey"

### Previewing Surveys

1. Navigate to **Survey Preview**
2. Select a survey from the dropdown
3. View survey details, questions, categories, and persona groups

### Running a Survey

1. Navigate to **Run Survey**
2. Select a survey from the dropdown
3. Configure run parameters:
   - Number of profiles to generate
   - LLM provider (OpenAI or Anthropic)
   - Model selection
   - LLM temperature (response creativity)
   - SSR temperature (distribution sharpness)
   - Normalization method
   - Random seed (for reproducibility)
4. Click "Run Survey"
5. Wait for processing (may take several minutes)
6. View results:
   - Response dataset table with demographics
   - Probability distributions
   - Sample visualizations
7. Export results to CSV

## Project Structure

```
react-demo-app/
├── backend/
│   ├── main.py              # FastAPI application
│   ├── requirements.txt     # Python dependencies
│   └── __init__.py
├── frontend/
│   ├── public/
│   │   └── index.html
│   ├── src/
│   │   ├── components/      # React components
│   │   │   ├── Layout/
│   │   │   ├── SurveyBuilder/
│   │   │   ├── SurveyPreview/
│   │   │   └── SurveyRunner/
│   │   ├── pages/           # Page components
│   │   │   ├── HomePage.tsx
│   │   │   ├── SurveyBuilderPage.tsx
│   │   │   ├── SurveyPreviewPage.tsx
│   │   │   └── SurveyRunnerPage.tsx
│   │   ├── services/        # API and data management
│   │   │   ├── api.ts       # API client
│   │   │   ├── hooks.ts     # React Query hooks
│   │   │   └── types.ts     # TypeScript types
│   │   ├── App.tsx          # Main app component
│   │   ├── index.tsx        # Entry point
│   │   └── index.css
│   ├── package.json
│   ├── tsconfig.json
│   └── .env.example
└── README.md
```

## API Endpoints

### Surveys
- `GET /api/surveys` - List all surveys
- `GET /api/surveys/{survey_id}` - Get survey details
- `POST /api/surveys` - Create new survey

### Pipeline Operations
- `POST /api/generate-profiles` - Generate respondent profiles
- `POST /api/generate-responses` - Generate LLM responses
- `POST /api/apply-ssr` - Apply SSR to responses
- `POST /api/run-survey` - Run complete pipeline

## Development

### Running Tests
```bash
# Backend
cd backend
pytest

# Frontend
cd frontend
npm test
```

### Building for Production
```bash
# Frontend
cd frontend
npm run build
```

## Technologies

### Backend
- FastAPI - Web framework
- Pydantic - Data validation
- Uvicorn - ASGI server
- Existing SSR pipeline modules

### Frontend
- React 18 - UI library
- TypeScript - Type safety
- Material-UI - Component library
- React Query - Server state management
- React Router - Routing
- Recharts - Data visualization
- Axios - HTTP client

## Notes

- The backend must be running for the frontend to work
- Survey configurations are saved to the `config/` directory
- LLM API keys must be set as environment variables
- The application is designed for local development and demonstration

## Support

For issues or questions, please refer to the main SSR Pipeline documentation.
