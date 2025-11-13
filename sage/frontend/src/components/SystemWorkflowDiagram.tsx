/**
 * System Workflow Diagram
 * Visual representation of how the SSR Pipeline system works
 */

import React from 'react';
import {
  Box,
  Typography,
  Paper,
  Card,
  CardContent,
  Grid,
  Chip,
  Stack,
} from '@mui/material';
import {
  ArrowForward as ArrowForwardIcon,
  Description as DescriptionIcon,
  People as PeopleIcon,
  Psychology as PsychologyIcon,
  AutoAwesome as AutoAwesomeIcon,
  BarChart as BarChartIcon,
  CompareArrows as CompareArrowsIcon,
} from '@mui/icons-material';

const SystemWorkflowDiagram: React.FC = () => {
  const steps = [
    {
      number: 1,
      title: 'Survey Definition',
      icon: <DescriptionIcon sx={{ fontSize: 40 }} />,
      description: 'Define survey questions, persona groups, and categories',
      color: '#4A90E2',
      items: ['Questions & Scales', 'Persona Groups', 'Target Demographics'],
    },
    {
      number: 2,
      title: 'Profile Generation',
      icon: <PeopleIcon sx={{ fontSize: 40 }} />,
      description: 'Generate synthetic respondent profiles based on persona groups',
      color: '#7B68EE',
      items: ['Age, Gender, Occupation', 'Persona Assignment', 'Demographic Distribution'],
    },
    {
      number: 3,
      title: 'LLM Response Generation',
      icon: <PsychologyIcon sx={{ fontSize: 40 }} />,
      description: 'LLMs generate natural language responses as synthetic respondents',
      color: '#FF6B9D',
      items: ['OpenAI / Anthropic', 'Context-Aware', 'Persona-Based Responses'],
    },
    {
      number: 4,
      title: 'SSR Application',
      icon: <AutoAwesomeIcon sx={{ fontSize: 40 }} />,
      description: 'Convert text responses to probability distributions using SSR',
      color: '#FF6E3A',
      items: ['Text → Probabilities', 'Confidence Scores', 'Distribution Metrics'],
    },
    {
      number: 5,
      title: 'Ground Truth Creation',
      icon: <BarChartIcon sx={{ fontSize: 40 }} />,
      description: 'Aggregate distributions to create ground truth baseline',
      color: '#50C878',
      items: ['Large Sample Size', 'Statistical Aggregation', 'Baseline Distribution'],
    },
    {
      number: 6,
      title: 'Experimentation & Comparison',
      icon: <CompareArrowsIcon sx={{ fontSize: 40 }} />,
      description: 'Run experiments with different parameters and compare to ground truth',
      color: '#367588',
      items: ['Different LLM Models', 'Varying Sample Sizes', 'Statistical Comparison'],
    },
  ];

  return (
    <Paper sx={{ p: 4, mb: 4, bgcolor: '#f8f9fa' }}>
      <Typography variant="h5" gutterBottom sx={{ fontWeight: 'bold', mb: 1 }}>
        How the SSR Pipeline System Works
      </Typography>
      <Typography variant="body2" color="text.secondary" sx={{ mb: 4 }}>
        The SSR (Semantic Similarity Rating) Pipeline converts LLM text responses into probability distributions,
        enabling statistical analysis and comparison of synthetic survey data.
      </Typography>

      {/* Workflow Steps */}
      <Grid container spacing={3}>
        {steps.map((step, index) => (
          <React.Fragment key={step.number}>
            <Grid item xs={12} md={4}>
              <Card
                sx={{
                  height: '100%',
                  borderTop: 4,
                  borderColor: step.color,
                  transition: 'transform 0.2s, box-shadow 0.2s',
                  '&:hover': {
                    transform: 'translateY(-4px)',
                    boxShadow: 4,
                  },
                }}
              >
                <CardContent>
                  <Stack spacing={2}>
                    {/* Step Number and Icon */}
                    <Box sx={{ display: 'flex', alignItems: 'center', gap: 2 }}>
                      <Chip
                        label={step.number}
                        sx={{
                          bgcolor: step.color,
                          color: 'white',
                          fontWeight: 'bold',
                          fontSize: '1.1rem',
                        }}
                      />
                      <Box sx={{ color: step.color }}>{step.icon}</Box>
                    </Box>

                    {/* Title */}
                    <Typography variant="h6" sx={{ fontWeight: 'bold' }}>
                      {step.title}
                    </Typography>

                    {/* Description */}
                    <Typography variant="body2" color="text.secondary">
                      {step.description}
                    </Typography>

                    {/* Key Items */}
                    <Box>
                      {step.items.map((item, idx) => (
                        <Typography
                          key={idx}
                          variant="body2"
                          sx={{
                            display: 'flex',
                            alignItems: 'center',
                            gap: 1,
                            mb: 0.5,
                          }}
                        >
                          <Box
                            component="span"
                            sx={{
                              width: 6,
                              height: 6,
                              borderRadius: '50%',
                              bgcolor: step.color,
                              display: 'inline-block',
                            }}
                          />
                          {item}
                        </Typography>
                      ))}
                    </Box>
                  </Stack>
                </CardContent>
              </Card>
            </Grid>

            {/* Arrow between steps (except after last step) */}
            {index < steps.length - 1 && (index + 1) % 3 === 0 && (
              <Grid item xs={12} sx={{ display: { xs: 'none', md: 'flex' }, justifyContent: 'center', my: -2 }}>
                <ArrowForwardIcon
                  sx={{
                    fontSize: 40,
                    color: 'text.secondary',
                    transform: 'rotate(90deg)',
                  }}
                />
              </Grid>
            )}
          </React.Fragment>
        ))}
      </Grid>

      {/* Key Concepts */}
      <Box sx={{ mt: 4, p: 3, bgcolor: 'white', borderRadius: 2 }}>
        <Typography variant="h6" gutterBottom sx={{ fontWeight: 'bold' }}>
          Key Concepts
        </Typography>
        <Grid container spacing={2}>
          <Grid item xs={12} md={4}>
            <Typography variant="subtitle2" sx={{ fontWeight: 'bold', color: '#4A90E2', mb: 1 }}>
              SSR (Semantic Similarity Rating)
            </Typography>
            <Typography variant="body2" color="text.secondary">
              A technique that converts natural language responses into numerical probability distributions
              by measuring semantic similarity between the response and each rating option.
            </Typography>
          </Grid>
          <Grid item xs={12} md={4}>
            <Typography variant="subtitle2" sx={{ fontWeight: 'bold', color: '#FF6E3A', mb: 1 }}>
              Ground Truth
            </Typography>
            <Typography variant="body2" color="text.secondary">
              A baseline dataset created with a large sample size and aggregated distributions,
              used as the reference point for comparing experimental runs.
            </Typography>
          </Grid>
          <Grid item xs={12} md={4}>
            <Typography variant="subtitle2" sx={{ fontWeight: 'bold', color: '#50C878', mb: 1 }}>
              Experiment Validation
            </Typography>
            <Typography variant="body2" color="text.secondary">
              Testing different configurations (models, sample sizes, temperatures) and comparing
              their distributions to ground truth using statistical metrics (KL divergence, etc.).
            </Typography>
          </Grid>
        </Grid>
      </Box>
    </Paper>
  );
};

export default SystemWorkflowDiagram;
