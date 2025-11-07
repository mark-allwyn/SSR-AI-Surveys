/**
 * Survey Runner Page
 * Execute surveys and view LLM responses
 */

import React, { useState } from 'react';
import { useParams } from 'react-router-dom';
import {
  Box,
  Typography,
  Paper,
  FormControl,
  InputLabel,
  Select,
  MenuItem,
  CircularProgress,
  Alert,
  Grid,
  Button,
  Snackbar,
} from '@mui/material';
import { PlayArrow as PlayArrowIcon } from '@mui/icons-material';
import { useSurveys, useSurvey } from '../services/hooks';
import RunConfigPanel from '../components/SurveyRunner/RunConfigPanel';
import RunProgress from '../components/SurveyRunner/RunProgress';
import ResponseDataset from '../components/SurveyRunner/ResponseDataset';
import { RunSurveyConfig, RunSurveyResponse } from '../services/types';

const SurveyRunnerPage: React.FC = () => {
  const { surveyId: urlSurveyId } = useParams<{ surveyId?: string }>();
  const [selectedSurveyId, setSelectedSurveyId] = useState<string>(urlSurveyId || '');
  const [runConfig, setRunConfig] = useState<RunSurveyConfig>({
    survey_id: '',
    num_profiles: 50,
    llm_provider: 'openai',
    model: 'gpt-3.5-turbo',
    llm_temperature: 0.7,
    ssr_temperature: 1.0,
    normalize_method: 'paper',
    seed: 100,
  });
  const [runResult, setRunResult] = useState<RunSurveyResponse | null>(null);
  const [snackbar, setSnackbar] = useState({ open: false, message: '', severity: 'success' as 'success' | 'error' });

  const { data: surveys, isLoading: surveysLoading, error: surveysError } = useSurveys();
  const {
    data: survey,
    isLoading: surveyLoading,
    error: surveyError,
  } = useSurvey(selectedSurveyId, { enabled: !!selectedSurveyId });

  const handleSurveyChange = (event: any) => {
    const newSurveyId = event.target.value;
    setSelectedSurveyId(newSurveyId);
    setRunConfig({ ...runConfig, survey_id: newSurveyId });
    setRunResult(null);
  };

  const [progressMessages, setProgressMessages] = useState<string[]>([]);
  const [currentProgress, setCurrentProgress] = useState(0);
  const [isStreaming, setIsStreaming] = useState(false);

  const handleRunSurvey = async () => {
    if (!selectedSurveyId) {
      setSnackbar({
        open: true,
        message: 'Please select a survey first',
        severity: 'error',
      });
      return;
    }

    setIsStreaming(true);
    setProgressMessages([]);
    setCurrentProgress(0);
    setRunResult(null);

    try {
      const response = await fetch('http://localhost:8000/api/run-survey-stream', {
        method: 'POST',
        headers: {
          'Content-Type': 'application/json',
        },
        body: JSON.stringify({ ...runConfig, survey_id: selectedSurveyId }),
      });

      const reader = response.body?.getReader();
      const decoder = new TextDecoder();

      if (!reader) throw new Error('No reader available');

      while (true) {
        const { done, value } = await reader.read();
        if (done) break;

        const chunk = decoder.decode(value);
        const lines = chunk.split('\n');

        for (const line of lines) {
          if (line.startsWith('data: ')) {
            const data = JSON.parse(line.slice(6));

            setProgressMessages(prev => [...prev, data.message]);
            setCurrentProgress(data.progress);

            if (data.status === 'complete' && data.result) {
              setRunResult(data.result);
              setSnackbar({
                open: true,
                message: `Survey completed! Generated ${data.result.num_responses} responses.`,
                severity: 'success',
              });
            } else if (data.status === 'error') {
              setSnackbar({
                open: true,
                message: data.message,
                severity: 'error',
              });
            }
          }
        }
      }
    } catch (error: any) {
      setSnackbar({
        open: true,
        message: `Error: ${error.message}`,
        severity: 'error',
      });
    } finally {
      setIsStreaming(false);
    }
  };

  const handleCloseSnackbar = () => {
    setSnackbar({ ...snackbar, open: false });
  };

  return (
    <Box>
      {/* Header */}
      <Box sx={{ mb: 4 }}>
        <Typography variant="h3" component="h1" gutterBottom>
          Run Survey
        </Typography>
        <Typography variant="body1" color="text.secondary">
          Execute the complete SSR pipeline: generate profiles, collect LLM responses, and apply semantic similarity rating.
        </Typography>
      </Box>

      {/* Survey Selector */}
      <Paper sx={{ p: 3, mb: 3 }}>
        <FormControl fullWidth>
          <InputLabel>Select Survey</InputLabel>
          <Select
            value={selectedSurveyId}
            label="Select Survey"
            onChange={handleSurveyChange}
            disabled={surveysLoading || isStreaming}
          >
            <MenuItem value="">
              <em>Choose a survey...</em>
            </MenuItem>
            {surveys?.map((s) => (
              <MenuItem key={s.id} value={s.id}>
                {s.name} ({s.num_questions} questions)
              </MenuItem>
            ))}
          </Select>
        </FormControl>
      </Paper>

      {/* Loading States */}
      {surveysLoading && (
        <Box sx={{ display: 'flex', justifyContent: 'center', p: 4 }}>
          <CircularProgress />
        </Box>
      )}

      {/* Error States */}
      {surveysError && (
        <Alert severity="error" sx={{ mb: 3 }}>
          Error loading surveys. Please ensure the backend API is running.
        </Alert>
      )}

      {surveyError && (
        <Alert severity="error" sx={{ mb: 3 }}>
          Error loading survey details.
        </Alert>
      )}

      {/* Survey Details */}
      {survey && !surveyLoading && (
        <Paper sx={{ p: 3, mb: 3 }}>
          <Typography variant="h5" gutterBottom>
            {survey.name}
          </Typography>

          {survey.description && (
            <Typography variant="body1" color="text.secondary" paragraph>
              {survey.description}
            </Typography>
          )}

          {survey.context && (
            <Box sx={{ mb: 3 }}>
              <Typography variant="subtitle2" color="primary" gutterBottom>
                Context
              </Typography>
              <Typography variant="body2" color="text.secondary">
                {survey.context}
              </Typography>
            </Box>
          )}

          <Grid container spacing={3}>
            {/* Questions */}
            <Grid item xs={12} md={6}>
              <Typography variant="subtitle2" color="primary" gutterBottom>
                Questions ({survey.questions.length})
              </Typography>
              <Box sx={{ maxHeight: 300, overflow: 'auto', bgcolor: 'background.default', borderRadius: 1, p: 2 }}>
                {survey.questions.map((q, index) => (
                  <Box key={q.id} sx={{ mb: 2 }}>
                    <Typography variant="body2" fontWeight="bold">
                      {index + 1}. {q.text}
                    </Typography>
                    {q.options && (
                      <Box sx={{ mt: 1, ml: 2 }}>
                        {q.options.map((option, optIndex) => (
                          <Typography key={optIndex} variant="caption" color="text.secondary" display="block">
                            • {option}
                          </Typography>
                        ))}
                      </Box>
                    )}
                    {q.scale && (
                      <Box sx={{ mt: 1, ml: 2 }}>
                        {Object.entries(q.scale).map(([value, label]) => (
                          <Typography key={value} variant="caption" color="text.secondary" display="block">
                            {value}: {label}
                          </Typography>
                        ))}
                      </Box>
                    )}
                  </Box>
                ))}
              </Box>
            </Grid>

            {/* Persona Groups */}
            <Grid item xs={12} md={6}>
              <Typography variant="subtitle2" color="primary" gutterBottom>
                Persona Groups ({survey.persona_groups.length})
              </Typography>
              <Box sx={{ maxHeight: 300, overflow: 'auto', bgcolor: 'background.default', borderRadius: 1, p: 2 }}>
                {survey.persona_groups.map((group, index) => (
                  <Box key={index} sx={{ mb: 2 }}>
                    <Typography variant="body2" fontWeight="bold">
                      {group.name} (Weight: {group.weight})
                    </Typography>
                    <Typography variant="caption" color="text.secondary" display="block" sx={{ mt: 0.5 }}>
                      {group.description}
                    </Typography>
                    <Typography variant="caption" color="text.secondary" display="block" sx={{ mt: 1 }}>
                      {group.personas.length} persona(s)
                    </Typography>
                  </Box>
                ))}
              </Box>
            </Grid>
          </Grid>
        </Paper>
      )}

      {/* Configuration Panel */}
      {survey && !surveyLoading && !runResult && (
        <Grid container spacing={3}>
          <Grid item xs={12} lg={4}>
            <RunConfigPanel
              config={runConfig}
              setConfig={setRunConfig}
              disabled={isStreaming}
            />
            <Box sx={{ mt: 2 }}>
              <Button
                fullWidth
                variant="contained"
                size="large"
                startIcon={isStreaming ? <CircularProgress size={20} /> : <PlayArrowIcon />}
                onClick={handleRunSurvey}
                disabled={isStreaming || !selectedSurveyId}
              >
                {isStreaming ? 'Running...' : 'Run Survey'}
              </Button>
            </Box>
          </Grid>

          <Grid item xs={12} lg={8}>
            {/* Progress Indicator */}
            {isStreaming && (
              <RunProgress progress={currentProgress} messages={progressMessages} />
            )}

            {/* Empty State */}
            {!runResult && !isStreaming && (
              <Paper sx={{ p: 6, textAlign: 'center' }}>
                <Typography variant="h6" color="text.secondary">
                  Configure the settings and click "Run Survey" to execute the pipeline
                </Typography>
              </Paper>
            )}
          </Grid>
        </Grid>
      )}

      {/* Results Display - Full Width */}
      {runResult && !isStreaming && survey && (
        <Box>
          <Box sx={{ mb: 2, display: 'flex', justifyContent: 'space-between', alignItems: 'center' }}>
            <Typography variant="h5">Survey Results</Typography>
            <Button
              variant="outlined"
              onClick={() => {
                setRunResult(null);
                setProgressMessages([]);
                setCurrentProgress(0);
              }}
            >
              Run Another Survey
            </Button>
          </Box>
          <ResponseDataset result={runResult} survey={survey} />
        </Box>
      )}

      {/* Snackbar for notifications */}
      <Snackbar
        open={snackbar.open}
        autoHideDuration={6000}
        onClose={handleCloseSnackbar}
        anchorOrigin={{ vertical: 'bottom', horizontal: 'center' }}
      >
        <Alert onClose={handleCloseSnackbar} severity={snackbar.severity} sx={{ width: '100%' }}>
          {snackbar.message}
        </Alert>
      </Snackbar>
    </Box>
  );
};

export default SurveyRunnerPage;
