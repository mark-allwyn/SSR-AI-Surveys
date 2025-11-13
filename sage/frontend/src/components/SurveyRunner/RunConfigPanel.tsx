/**
 * Run Config Panel Component
 * Configuration options for running surveys
 */

import React from 'react';
import {
  Paper,
  Typography,
  TextField,
  FormControl,
  InputLabel,
  Select,
  MenuItem,
  Grid,
  Slider,
  Box,
} from '@mui/material';
import { RunSurveyConfig, LLM_PROVIDERS, OPENAI_MODELS, ANTHROPIC_MODELS, NORMALIZE_METHODS } from '../../services/types';

interface RunConfigPanelProps {
  config: RunSurveyConfig;
  setConfig: (config: RunSurveyConfig) => void;
  disabled?: boolean;
}

const RunConfigPanel: React.FC<RunConfigPanelProps> = ({ config, setConfig, disabled }) => {
  const handleChange = (field: keyof RunSurveyConfig, value: any) => {
    setConfig({ ...config, [field]: value });
  };

  const getModelOptions = () => {
    return config.llm_provider === 'openai' ? OPENAI_MODELS : ANTHROPIC_MODELS;
  };

  return (
    <Paper sx={{ p: 3 }}>
      <Typography variant="h5" gutterBottom>
        Run Configuration
      </Typography>

      <Grid container spacing={2}>
        {/* Sample Size */}
        <Grid item xs={12}>
          <TextField
            fullWidth
            type="number"
            label="Number of Profiles"
            value={config.num_profiles}
            onChange={(e) => handleChange('num_profiles', parseInt(e.target.value) || 50)}
            inputProps={{ min: 10, max: 500 }}
            helperText="Number of respondent profiles to generate"
            disabled={disabled}
          />
        </Grid>

        {/* LLM Provider */}
        <Grid item xs={12}>
          <FormControl fullWidth disabled={disabled}>
            <InputLabel>LLM Provider</InputLabel>
            <Select
              value={config.llm_provider}
              label="LLM Provider"
              onChange={(e) => {
                // Update provider and model atomically
                const newProvider = e.target.value as 'openai' | 'anthropic';
                const newModels = newProvider === 'openai' ? OPENAI_MODELS : ANTHROPIC_MODELS;
                setConfig({
                  ...config,
                  llm_provider: newProvider,
                  model: newModels[0].value
                });
              }}
            >
              {LLM_PROVIDERS.map((provider) => (
                <MenuItem key={provider.value} value={provider.value}>
                  {provider.label}
                </MenuItem>
              ))}
            </Select>
          </FormControl>
        </Grid>

        {/* Model */}
        <Grid item xs={12}>
          <FormControl fullWidth disabled={disabled}>
            <InputLabel>Model</InputLabel>
            <Select
              value={config.model}
              label="Model"
              onChange={(e) => handleChange('model', e.target.value)}
            >
              {getModelOptions().map((model) => (
                <MenuItem key={model.value} value={model.value}>
                  {model.label}
                </MenuItem>
              ))}
            </Select>
          </FormControl>
        </Grid>

        {/* LLM Temperature */}
        <Grid item xs={12}>
          <Typography gutterBottom>
            LLM Temperature: {config.llm_temperature}
          </Typography>
          <Slider
            value={config.llm_temperature}
            onChange={(_, value) => handleChange('llm_temperature', value)}
            min={0}
            max={2}
            step={0.1}
            marks={[
              { value: 0, label: '0' },
              { value: 1, label: '1' },
              { value: 2, label: '2' },
            ]}
            disabled={disabled}
          />
          <Typography variant="caption" color="text.secondary">
            Controls randomness in LLM responses
          </Typography>
        </Grid>

        {/* SSR Temperature */}
        <Grid item xs={12}>
          <Typography gutterBottom>
            SSR Temperature: {config.ssr_temperature}
          </Typography>
          <Slider
            value={config.ssr_temperature}
            onChange={(_, value) => handleChange('ssr_temperature', value)}
            min={0.1}
            max={5}
            step={0.1}
            marks={[
              { value: 0.1, label: '0.1' },
              { value: 1, label: '1' },
              { value: 5, label: '5' },
            ]}
            disabled={disabled}
          />
          <Typography variant="caption" color="text.secondary">
            Controls distribution sharpness in SSR
          </Typography>
        </Grid>

        {/* Normalization Method */}
        <Grid item xs={12}>
          <FormControl fullWidth disabled={disabled}>
            <InputLabel>Normalization Method</InputLabel>
            <Select
              value={config.normalize_method}
              label="Normalization Method"
              onChange={(e) => handleChange('normalize_method', e.target.value)}
            >
              {NORMALIZE_METHODS.map((method) => (
                <MenuItem key={method.value} value={method.value}>
                  <Box>
                    <Typography variant="body2">{method.label}</Typography>
                    <Typography variant="caption" color="text.secondary">
                      {method.description}
                    </Typography>
                  </Box>
                </MenuItem>
              ))}
            </Select>
          </FormControl>
        </Grid>

        {/* Seed */}
        <Grid item xs={12}>
          <TextField
            fullWidth
            type="number"
            label="Random Seed"
            value={config.seed}
            onChange={(e) => handleChange('seed', parseInt(e.target.value) || 100)}
            inputProps={{ min: 0, max: 10000 }}
            helperText="For reproducibility"
            disabled={disabled}
          />
        </Grid>
      </Grid>
    </Paper>
  );
};

export default RunConfigPanel;
