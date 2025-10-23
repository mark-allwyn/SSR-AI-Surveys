# Streamlit UI for SSR Pipeline

A user-friendly web interface for the Semantic Similarity Rating (SSR) Pipeline.

## Features

- 🏠 **Dashboard**: Overview of experiments and quick access
- ▶️ **Run Experiment**: Configure and execute SSR pipeline with customizable personas
- 📊 **View Results**: Explore experiment results with visualizations
- 🎮 **Live Demo**: Test SSR on individual responses in real-time
- ⚙️ **Settings**: API configuration, persona defaults, and experiment management

## Quick Start

### 1. Ensure Dependencies are Installed

```bash
pip install streamlit
# or
uv pip install streamlit
```

### 2. Launch the UI

From the project root directory:

```bash
streamlit run ui/app.py
```

The UI will open in your default web browser at `http://localhost:8501`

### 3. Configure API Key

1. Go to **Settings** page
2. Enter your OpenAI API key (starts with `sk-`)
3. Choose to save to session or `.env` file
4. Test the connection

## Navigation

The UI uses Streamlit's multi-page app structure:

```
ui/
├── app.py                          # Main entry point (Dashboard)
├── pages/
│   ├── 2_▶️_Run_Experiment.py     # Configure and run experiments
│   ├── 3_📊_View_Results.py       # View experiment results
│   ├── 4_🎮_Live_Demo.py          # Interactive SSR testing
│   └── 5_⚙️_Settings.py           # Configuration and management
├── components/
│   └── metrics_cards.py            # Reusable UI components
└── utils/
    └── data_loader.py              # Data loading utilities
```

## Usage Guide

### Running an Experiment

1. **Go to Run Experiment page**
2. **Select a survey** from the dropdown
3. **Configure experiment settings:**
   - Number of respondents (10-200)
   - Random seed for reproducibility
   - Response styles (human/LLM)
4. **Customize persona parameters** (optional):
   - Age groups
   - Income brackets
   - Environmental consciousness levels
5. **Adjust SSR settings:**
   - Temperature (default: 1.0)
6. **Click "Run Experiment"**
7. **View progress** as pipeline executes
8. **Review results** when complete

### Viewing Results

1. **Go to View Results page**
2. **Select an experiment** from dropdown
3. **Explore:**
   - Overall metrics (Human vs LLM accuracy)
   - Visual report (confusion matrices, charts)
   - Question-by-question analysis
   - Ground truth data with filters
   - Downloadable reports (PNG, TXT, MD)

### Using Live Demo

1. **Go to Live Demo page**
2. **Configure question:**
   - Select question type (Yes/No, Likert-5, etc.)
   - Enter question text
   - Define scale labels
3. **Enter response text** or use examples
4. **Adjust temperature** (optional)
5. **Click "Process Response"**
6. **View results:**
   - Predicted rating
   - Probability distribution
   - Confidence metrics
   - Interpretation

### Managing Settings

#### API Configuration
- Enter OpenAI API key
- Save to session or `.env` file
- Test connection
- Clear API key if needed

#### Persona Defaults
- Configure age groups
- Configure income brackets
- Configure environmental consciousness levels
- Reset to defaults

#### Experiment Management
- View all experiments in table
- Delete individual experiments
- Bulk delete (keep N most recent)

## Customizing Persona Parameters

The UI allows you to customize persona parameters both **globally** (in Settings) and **per-experiment** (in Run Experiment).

**Default Persona Categories:**

```python
age_groups = ["18-25", "26-35", "36-45", "46-55", "56-65", "65+"]
income_brackets = ["<$30k", "$30k-$50k", "$50k-$75k", "$75k-$100k", "$100k-$150k", ">$150k"]
env_consciousness = ["Not concerned", "Slightly concerned", "Moderately concerned",
                     "Very concerned", "Extremely concerned"]
```

**To customize:**

1. Go to Settings → Persona Defaults tab
2. Edit the text areas (one category per line)
3. Click "Save Persona Configuration"

OR

1. In Run Experiment page
2. Expand the persona configuration expanders
3. Edit for this experiment only

## Tips & Best Practices

### Experiment Configuration

- **Start small**: Begin with 50 respondents for quick testing
- **Use seeds**: Keep the same random seed to reproduce results
- **Temperature**: Stick with 1.0 (paper default) unless experimenting

### Persona Configuration

- **More categories = more diversity**: But processing time increases
- **Environmental consciousness is key**: Directly influences ground truth generation
- **Balanced distribution**: Profiles are randomly sampled from categories

### Performance

**Expected processing times:**
- 50 respondents, 6 questions: ~2-3 minutes
- 100 respondents, 6 questions: ~4-5 minutes
- 200 respondents, 6 questions: ~8-10 minutes

**Optimization tips:**
- Close unnecessary browser tabs
- Avoid running multiple experiments simultaneously
- Monitor API usage/costs

## Troubleshooting

### UI won't start

```bash
# Check if streamlit is installed
pip list | grep streamlit

# Reinstall if needed
pip install streamlit --upgrade
```

### API key errors

- Verify key starts with `sk-`
- Check key has embedding permissions
- Test connection in Settings page
- Check OpenAI account has credits

### Slow performance

- Reduce number of respondents
- Check internet connection
- Monitor OpenAI API status
- Clear browser cache

### Experiments not showing

- Check `experiments/` folder exists
- Verify experiment folders have required files:
  - `ground_truth.csv`
  - `report.png`
  - `report.txt`
  - `report.md`

## Deployment Options

### Local Only (Current Setup)

```bash
streamlit run ui/app.py
```

Access at `http://localhost:8501`

### Streamlit Cloud (Public/Private)

1. Push code to GitHub
2. Go to [share.streamlit.io](https://share.streamlit.io)
3. Connect repository
4. Deploy app
5. Add secrets (API keys) in dashboard

### Docker (Advanced)

Create `Dockerfile`:

```dockerfile
FROM python:3.10-slim

WORKDIR /app

COPY requirements.txt .
RUN pip install -r requirements.txt

COPY . .

EXPOSE 8501

CMD ["streamlit", "run", "ui/app.py"]
```

Build and run:

```bash
docker build -t ssr-pipeline-ui .
docker run -p 8501:8501 ssr-pipeline-ui
```

## Environment Variables

The UI looks for these environment variables:

```bash
OPENAI_API_KEY=sk-...       # OpenAI API key (required)
```

You can set these in:
- `.env` file (root directory)
- System environment variables
- Streamlit secrets (for cloud deployment)

## File Structure

```
ui/
├── README.md                   # This file
├── app.py                      # Main app (Dashboard)
│
├── pages/                      # Multi-page app pages
│   ├── 2_▶️_Run_Experiment.py
│   ├── 3_📊_View_Results.py
│   ├── 4_🎮_Live_Demo.py
│   └── 5_⚙️_Settings.py
│
├── components/                 # Reusable components
│   ├── __init__.py
│   └── metrics_cards.py       # Metric display components
│
└── utils/                      # Utility functions
    ├── __init__.py
    └── data_loader.py         # Experiment data loading
```

## Customization

### Adding New Pages

1. Create new file in `ui/pages/`
2. Name with format: `N_Emoji_Page_Name.py`
3. Add page content
4. It will automatically appear in sidebar

### Styling

Edit CSS in `app.py`:

```python
st.markdown("""
<style>
    .your-custom-class {
        /* your styles */
    }
</style>
""", unsafe_allow_html=True)
```

### Adding Components

Create new component in `ui/components/`:

```python
# ui/components/new_component.py
import streamlit as st

def my_component(data):
    st.markdown("Custom component")
    # ...
```

Use in pages:

```python
from ui.components.new_component import my_component

my_component(data)
```

## Support

- **Documentation**: See main [README.md](../README.md)
- **Issues**: Report bugs on GitHub
- **Paper**: [arXiv:2510.08338v2](https://arxiv.org/abs/2510.08338v2)

## Version

- **UI Version**: 1.0.0 (MVP)
- **Streamlit Version**: >=1.30.0
- **Python Version**: >=3.10

---

Made with ❤️ using Streamlit 🎈
