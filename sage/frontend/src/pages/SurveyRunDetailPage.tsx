/**
 * Survey Run Detail Page
 * View detailed information about a specific survey run
 */

import React, { useState } from 'react';
import { useParams, useNavigate } from 'react-router-dom';
import {
  Box,
  Typography,
  Paper,
  Button,
  Grid,
  Chip,
  CircularProgress,
  Alert,
  Breadcrumbs,
  Link,
  Divider,
  Dialog,
  DialogTitle,
  DialogContent,
  DialogActions,
  FormControl,
  InputLabel,
  Select,
  MenuItem,
} from '@mui/material';
import {
  ArrowBack as ArrowBackIcon,
  Download as DownloadIcon,
  CompareArrows as CompareArrowsIcon,
} from '@mui/icons-material';
import { useSurveyRun, useSurvey, useGroundTruths, useCompareToGroundTruth } from '../services/hooks';
import ResponseDataset from '../components/SurveyRunner/ResponseDataset';

const SurveyRunDetailPage: React.FC = () => {
  const { runId } = useParams<{ runId: string }>();
  const navigate = useNavigate();
  const [compareDialogOpen, setCompareDialogOpen] = useState(false);
  const [selectedGroundTruth, setSelectedGroundTruth] = useState('');

  const { data: run, isLoading, error } = useSurveyRun(runId || '');
  const { data: survey } = useSurvey(run?.survey_id || '', { enabled: !!run });
  const { data: groundTruths } = useGroundTruths(run?.survey_id || '', { enabled: !!run });

  const compareMutation = useCompareToGroundTruth({
    onSuccess: (data) => {
      console.log('Comparison results:', data);
      // TODO: Navigate to comparison results page or show in modal
      alert('Comparison complete! Check console for results.');
      setCompareDialogOpen(false);
    },
  });

  const handleExportJSON = () => {
    if (!run) return;
    const dataStr = JSON.stringify(run, null, 2);
    const dataBlob = new Blob([dataStr], { type: 'application/json' });
    const url = URL.createObjectURL(dataBlob);
    const link = document.createElement('a');
    link.href = url;
    link.download = `${run.run_id}.json`;
    link.click();
    URL.revokeObjectURL(url);
  };

  const handleCompare = () => {
    if (runId && selectedGroundTruth) {
      compareMutation.mutate({ runId, groundTruthId: selectedGroundTruth });
    }
  };

  const formatDate = (timestamp: string) => {
    return new Date(timestamp).toLocaleString();
  };

  if (isLoading) {
    return (
      <Box sx={{ display: 'flex', justifyContent: 'center', p: 4 }}>
        <CircularProgress />
      </Box>
    );
  }

  if (error || !run) {
    return (
      <Box>
        <Alert severity="error" sx={{ mb: 3 }}>
          Error loading survey run details. The run may not exist.
        </Alert>
        <Button startIcon={<ArrowBackIcon />} onClick={() => navigate('/history')}>
          Back to History
        </Button>
      </Box>
    );
  }

  return (
    <Box>
      {/* Breadcrumbs */}
      <Breadcrumbs sx={{ mb: 2 }}>
        <Link
          component="button"
          variant="body1"
          onClick={() => navigate('/')}
          sx={{ textDecoration: 'none', cursor: 'pointer' }}
        >
          Home
        </Link>
        <Link
          component="button"
          variant="body1"
          onClick={() => navigate('/history')}
          sx={{ textDecoration: 'none', cursor: 'pointer' }}
        >
          Survey History
        </Link>
        <Typography color="text.primary">{run.run_id}</Typography>
      </Breadcrumbs>

      {/* Header */}
      <Box sx={{ mb: 3, display: 'flex', justifyContent: 'space-between', alignItems: 'flex-start' }}>
        <Box>
          <Typography variant="h4" component="h1" gutterBottom>
            Survey Run Details
          </Typography>
          <Typography variant="body1" color="text.secondary" fontFamily="monospace">
            {run.run_id}
          </Typography>
        </Box>
        <Box sx={{ display: 'flex', gap: 1 }}>
          <Button startIcon={<CompareArrowsIcon />} variant="outlined" onClick={() => setCompareDialogOpen(true)}>
            Compare to Ground Truth
          </Button>
          <Button startIcon={<DownloadIcon />} variant="outlined" onClick={handleExportJSON}>
            Export JSON
          </Button>
          <Button startIcon={<ArrowBackIcon />} onClick={() => navigate('/history')}>
            Back
          </Button>
        </Box>
      </Box>

      {/* Run Metadata */}
      <Paper sx={{ p: 3, mb: 3 }}>
        <Typography variant="h6" gutterBottom>
          Run Information
        </Typography>
        <Divider sx={{ mb: 2 }} />
        <Grid container spacing={3}>
          <Grid item xs={12} md={6}>
            <Typography variant="subtitle2" color="text.secondary" gutterBottom>
              Survey
            </Typography>
            <Typography variant="body1" fontWeight="medium">
              {run.survey_name || run.survey_id}
            </Typography>
          </Grid>
          <Grid item xs={12} md={6}>
            <Typography variant="subtitle2" color="text.secondary" gutterBottom>
              Timestamp
            </Typography>
            <Typography variant="body1">{formatDate(run.timestamp)}</Typography>
          </Grid>
          <Grid item xs={12} md={4}>
            <Typography variant="subtitle2" color="text.secondary" gutterBottom>
              Profiles Generated
            </Typography>
            <Chip label={run.num_profiles} color="primary" />
          </Grid>
          <Grid item xs={12} md={4}>
            <Typography variant="subtitle2" color="text.secondary" gutterBottom>
              Responses Collected
            </Typography>
            <Chip label={run.num_responses} color="secondary" />
          </Grid>
          <Grid item xs={12} md={4}>
            <Typography variant="subtitle2" color="text.secondary" gutterBottom>
              Distributions Generated
            </Typography>
            <Chip label={run.num_distributions} color="info" />
          </Grid>
        </Grid>
      </Paper>

      {/* Configuration */}
      <Paper sx={{ p: 3, mb: 3 }}>
        <Typography variant="h6" gutterBottom>
          Configuration
        </Typography>
        <Divider sx={{ mb: 2 }} />
        <Grid container spacing={3}>
          <Grid item xs={12} md={6}>
            <Typography variant="subtitle2" color="text.secondary" gutterBottom>
              LLM Provider
            </Typography>
            <Chip label={run.config.llm_provider} variant="outlined" />
          </Grid>
          <Grid item xs={12} md={6}>
            <Typography variant="subtitle2" color="text.secondary" gutterBottom>
              Model
            </Typography>
            <Chip label={run.config.model} variant="outlined" sx={{ fontFamily: 'monospace' }} />
          </Grid>
          <Grid item xs={12} md={4}>
            <Typography variant="subtitle2" color="text.secondary" gutterBottom>
              LLM Temperature
            </Typography>
            <Typography variant="body1">{run.config.llm_temperature}</Typography>
          </Grid>
          <Grid item xs={12} md={4}>
            <Typography variant="subtitle2" color="text.secondary" gutterBottom>
              SSR Temperature
            </Typography>
            <Typography variant="body1">{run.config.ssr_temperature}</Typography>
          </Grid>
          <Grid item xs={12} md={4}>
            <Typography variant="subtitle2" color="text.secondary" gutterBottom>
              Seed
            </Typography>
            <Typography variant="body1" fontFamily="monospace">
              {run.config.seed}
            </Typography>
          </Grid>
          <Grid item xs={12}>
            <Typography variant="subtitle2" color="text.secondary" gutterBottom>
              Normalization Method
            </Typography>
            <Chip label={run.config.normalize_method} variant="outlined" />
          </Grid>
        </Grid>
      </Paper>

      {/* Results */}
      {survey && (
        <Box>
          <Typography variant="h5" gutterBottom sx={{ mb: 2 }}>
            Survey Results
          </Typography>
          <ResponseDataset
            result={{
              run_id: run.run_id,
              survey_id: run.survey_id,
              num_profiles: run.num_profiles,
              num_responses: run.num_responses,
              num_distributions: run.num_distributions,
              distributions: run.distributions,
              config: run.config,
            }}
            survey={survey}
          />
        </Box>
      )}

      {/* Compare to Ground Truth Dialog */}
      <Dialog
        open={compareDialogOpen}
        onClose={() => setCompareDialogOpen(false)}
        maxWidth="sm"
        fullWidth
      >
        <DialogTitle>Compare to Ground Truth</DialogTitle>
        <DialogContent>
          <Typography variant="body2" color="text.secondary" sx={{ mb: 3 }}>
            Select a ground truth to compare this run against. The system will calculate statistical
            metrics including KL divergence, JS divergence, Wasserstein distance, and more.
          </Typography>
          <FormControl fullWidth>
            <InputLabel>Select Ground Truth</InputLabel>
            <Select
              value={selectedGroundTruth}
              label="Select Ground Truth"
              onChange={(e) => setSelectedGroundTruth(e.target.value)}
            >
              {groundTruths && groundTruths.length > 0 ? (
                groundTruths.map((gt) => (
                  <MenuItem key={gt.id} value={gt.id}>
                    {gt.name}
                    {gt.source === 'ssr_generated' && gt.generation_config && (
                      <Typography variant="caption" color="text.secondary" sx={{ ml: 1 }}>
                        ({gt.generation_config.model}, n={gt.generation_config.num_profiles})
                      </Typography>
                    )}
                  </MenuItem>
                ))
              ) : (
                <MenuItem value="" disabled>
                  No ground truths available for this survey
                </MenuItem>
              )}
            </Select>
          </FormControl>
        </DialogContent>
        <DialogActions>
          <Button onClick={() => setCompareDialogOpen(false)}>Cancel</Button>
          <Button
            onClick={handleCompare}
            variant="contained"
            disabled={!selectedGroundTruth || compareMutation.isPending}
          >
            {compareMutation.isPending ? 'Comparing...' : 'Compare'}
          </Button>
        </DialogActions>
      </Dialog>
    </Box>
  );
};

export default SurveyRunDetailPage;
