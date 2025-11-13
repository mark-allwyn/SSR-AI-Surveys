/**
 * Home Page
 * Landing page with overview and quick actions
 */

import React from 'react';
import { useNavigate } from 'react-router-dom';
import {
  Box,
  Typography,
  Grid,
  Card,
  CardContent,
  CardActions,
  Button,
  Paper,
  Alert,
} from '@mui/material';
import {
  Create as CreateIcon,
  Visibility as VisibilityIcon,
  PlayArrow as PlayArrowIcon,
  Science as ScienceIcon,
  History as HistoryIcon,
  CompareArrows as CompareArrowsIcon,
} from '@mui/icons-material';
import { useHealthCheck } from '../services/hooks';

const HomePage: React.FC = () => {
  const navigate = useNavigate();
  const { data: health, isLoading, isError } = useHealthCheck();

  return (
    <Box>
      {/* Header */}
      <Box sx={{ mb: 4, textAlign: 'center' }}>
        <Box sx={{ display: 'flex', justifyContent: 'center', alignItems: 'center', gap: 2, mb: 2 }}>
          <ScienceIcon sx={{ fontSize: 48, color: 'primary.main' }} />
          <Typography variant="h2" component="h1">
            S.A.G.E
          </Typography>
        </Box>
        <Typography variant="h5" color="text.secondary" gutterBottom>
          Synthetic Audience Generation Engine
        </Typography>
        <Typography variant="body1" color="text.secondary" sx={{ mt: 2, maxWidth: 800, mx: 'auto' }}>
          Generate synthetic survey audiences and transform qualitative responses into quantitative
          probability distributions using advanced semantic similarity techniques powered by large language models.
        </Typography>
      </Box>

      {/* API Status */}
      <Box sx={{ mb: 4 }}>
        {isLoading && (
          <Alert severity="info">Checking API connection...</Alert>
        )}
        {isError && (
          <Alert severity="error">
            Cannot connect to API backend. Please ensure the FastAPI server is running on http://localhost:8000
          </Alert>
        )}
        {health && (
          <Alert severity="success">
            API Connected - Version {health.version}
          </Alert>
        )}
      </Box>

      {/* Quick Actions */}
      <Typography variant="h4" gutterBottom sx={{ mb: 3 }}>
        Quick Actions
      </Typography>
      <Grid container spacing={3} sx={{ mb: 6 }}>
        <Grid item xs={12} md={4}>
          <Card sx={{ height: '100%', display: 'flex', flexDirection: 'column' }}>
            <CardContent sx={{ flexGrow: 1 }}>
              <CreateIcon sx={{ fontSize: 40, color: 'primary.main', mb: 2 }} />
              <Typography variant="h5" component="h2" gutterBottom>
                Survey Builder
              </Typography>
              <Typography variant="body2" color="text.secondary">
                Create and configure surveys with questions, persona groups, and categories.
                Define demographic targets and rating scales.
              </Typography>
            </CardContent>
            <CardActions>
              <Button
                size="large"
                variant="contained"
                fullWidth
                onClick={() => navigate('/builder')}
              >
                Build Survey
              </Button>
            </CardActions>
          </Card>
        </Grid>

        <Grid item xs={12} md={4}>
          <Card sx={{ height: '100%', display: 'flex', flexDirection: 'column' }}>
            <CardContent sx={{ flexGrow: 1 }}>
              <VisibilityIcon sx={{ fontSize: 40, color: 'primary.main', mb: 2 }} />
              <Typography variant="h5" component="h2" gutterBottom>
                Survey Preview
              </Typography>
              <Typography variant="body2" color="text.secondary">
                View and explore existing survey configurations. See questions, categories,
                and persona group definitions.
              </Typography>
            </CardContent>
            <CardActions>
              <Button
                size="large"
                variant="contained"
                fullWidth
                onClick={() => navigate('/preview')}
              >
                Preview Surveys
              </Button>
            </CardActions>
          </Card>
        </Grid>

        <Grid item xs={12} md={4}>
          <Card sx={{ height: '100%', display: 'flex', flexDirection: 'column' }}>
            <CardContent sx={{ flexGrow: 1 }}>
              <PlayArrowIcon sx={{ fontSize: 40, color: 'primary.main', mb: 2 }} />
              <Typography variant="h5" component="h2" gutterBottom>
                Run Survey
              </Typography>
              <Typography variant="body2" color="text.secondary">
                Execute the complete pipeline: generate profiles, collect LLM responses,
                and apply SSR to produce probability distributions.
              </Typography>
            </CardContent>
            <CardActions>
              <Button
                size="large"
                variant="contained"
                fullWidth
                onClick={() => navigate('/runner')}
              >
                Run Survey
              </Button>
            </CardActions>
          </Card>
        </Grid>

        <Grid item xs={12} md={6}>
          <Card sx={{ height: '100%', display: 'flex', flexDirection: 'column' }}>
            <CardContent sx={{ flexGrow: 1 }}>
              <HistoryIcon sx={{ fontSize: 40, color: 'secondary.main', mb: 2 }} />
              <Typography variant="h5" component="h2" gutterBottom>
                Survey History
              </Typography>
              <Typography variant="body2" color="text.secondary">
                View all previously run surveys. Filter, search, and explore past results.
                Compare runs and export data for analysis.
              </Typography>
            </CardContent>
            <CardActions>
              <Button
                size="large"
                variant="outlined"
                fullWidth
                onClick={() => navigate('/history')}
              >
                View History
              </Button>
            </CardActions>
          </Card>
        </Grid>

        <Grid item xs={12} md={6}>
          <Card sx={{ height: '100%', display: 'flex', flexDirection: 'column' }}>
            <CardContent sx={{ flexGrow: 1 }}>
              <CompareArrowsIcon sx={{ fontSize: 40, color: 'secondary.main', mb: 2 }} />
              <Typography variant="h5" component="h2" gutterBottom>
                Ground Truth Testing
              </Typography>
              <Typography variant="body2" color="text.secondary">
                Create ground truths and run experiments to validate synthetic data quality.
                Compare distributions and calculate statistical metrics.
              </Typography>
            </CardContent>
            <CardActions>
              <Button
                size="large"
                variant="outlined"
                fullWidth
                onClick={() => navigate('/ground-truth')}
              >
                Run Experiments
              </Button>
            </CardActions>
          </Card>
        </Grid>
      </Grid>

      {/* How It Works */}
      <Paper sx={{ p: 4, mb: 4 }}>
        <Typography variant="h4" gutterBottom>
          How It Works
        </Typography>
        <Grid container spacing={3} sx={{ mt: 2 }}>
          <Grid item xs={12} md={3}>
            <Typography variant="h6" color="primary" gutterBottom>
              1. Define Survey
            </Typography>
            <Typography variant="body2" color="text.secondary">
              Create questions, categories, and persona groups that represent your target demographics.
            </Typography>
          </Grid>
          <Grid item xs={12} md={3}>
            <Typography variant="h6" color="primary" gutterBottom>
              2. Generate Profiles
            </Typography>
            <Typography variant="body2" color="text.secondary">
              Automatically create diverse respondent profiles based on persona groups and demographic targets.
            </Typography>
          </Grid>
          <Grid item xs={12} md={3}>
            <Typography variant="h6" color="primary" gutterBottom>
              3. Collect Responses
            </Typography>
            <Typography variant="body2" color="text.secondary">
              Use LLMs (GPT-4, Claude) to generate natural language responses from each profile.
            </Typography>
          </Grid>
          <Grid item xs={12} md={3}>
            <Typography variant="h6" color="primary" gutterBottom>
              4. Apply SSR
            </Typography>
            <Typography variant="body2" color="text.secondary">
              Convert text responses to probability distributions using semantic similarity rating.
            </Typography>
          </Grid>
        </Grid>
      </Paper>

      {/* Features */}
      <Typography variant="h4" gutterBottom>
        Key Features
      </Typography>
      <Grid container spacing={2} sx={{ mt: 2 }}>
        <Grid item xs={12} md={6}>
          <Box sx={{ mb: 2 }}>
            <Typography variant="h6" gutterBottom>
              Multi-Category Support
            </Typography>
            <Typography variant="body2" color="text.secondary">
              Compare multiple products, services, or concepts within a single survey.
            </Typography>
          </Box>
        </Grid>
        <Grid item xs={12} md={6}>
          <Box sx={{ mb: 2 }}>
            <Typography variant="h6" gutterBottom>
              Flexible Question Types
            </Typography>
            <Typography variant="body2" color="text.secondary">
              Yes/No, Likert scales, multiple choice, and preference comparisons.
            </Typography>
          </Box>
        </Grid>
        <Grid item xs={12} md={6}>
          <Box sx={{ mb: 2 }}>
            <Typography variant="h6" gutterBottom>
              Persona-Based Sampling
            </Typography>
            <Typography variant="body2" color="text.secondary">
              Target specific demographics with weighted persona groups.
            </Typography>
          </Box>
        </Grid>
        <Grid item xs={12} md={6}>
          <Box sx={{ mb: 2 }}>
            <Typography variant="h6" gutterBottom>
              Multiple LLM Providers
            </Typography>
            <Typography variant="body2" color="text.secondary">
              Choose between OpenAI GPT-4 and Anthropic Claude models.
            </Typography>
          </Box>
        </Grid>
      </Grid>
    </Box>
  );
};

export default HomePage;
