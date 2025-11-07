/**
 * Survey Form Component
 * Main form for creating/editing surveys
 */

import React from 'react';
import {
  Box,
  Paper,
  Typography,
  TextField,
  Grid,
  FormControl,
  InputLabel,
  Select,
  MenuItem,
  OutlinedInput,
  Chip,
} from '@mui/material';
import { SurveyBuilderState, DEMOGRAPHICS_OPTIONS } from '../../services/types';
import QuestionEditor from './QuestionEditor';
import PersonaGroupEditor from './PersonaGroupEditor';
import CategoryEditor from './CategoryEditor';

interface SurveyFormProps {
  surveyData: SurveyBuilderState;
  setSurveyData: (data: SurveyBuilderState) => void;
}

const SurveyForm: React.FC<SurveyFormProps> = ({ surveyData, setSurveyData }) => {
  const handleChange = (field: keyof SurveyBuilderState, value: any) => {
    setSurveyData({ ...surveyData, [field]: value });
  };

  return (
    <Box>
      {/* Basic Information */}
      <Paper sx={{ p: 3, mb: 3 }}>
        <Typography variant="h5" gutterBottom>
          Basic Information
        </Typography>
        <Grid container spacing={2}>
          <Grid item xs={12}>
            <TextField
              fullWidth
              label="Survey Name"
              value={surveyData.name}
              onChange={(e) => handleChange('name', e.target.value)}
              required
            />
          </Grid>
          <Grid item xs={12}>
            <TextField
              fullWidth
              label="Description"
              value={surveyData.description}
              onChange={(e) => handleChange('description', e.target.value)}
              multiline
              rows={2}
              required
            />
          </Grid>
          <Grid item xs={12}>
            <TextField
              fullWidth
              label="Context"
              value={surveyData.context}
              onChange={(e) => handleChange('context', e.target.value)}
              multiline
              rows={3}
              helperText="General context or background for the survey"
              required
            />
          </Grid>
          <Grid item xs={12} md={6}>
            <TextField
              fullWidth
              type="number"
              label="Sample Size"
              value={surveyData.sample_size}
              onChange={(e) => handleChange('sample_size', parseInt(e.target.value) || 100)}
              inputProps={{ min: 10, max: 1000 }}
            />
          </Grid>
          <Grid item xs={12} md={6}>
            <FormControl fullWidth>
              <InputLabel>Demographics</InputLabel>
              <Select
                multiple
                value={surveyData.demographics}
                onChange={(e) => handleChange('demographics', e.target.value)}
                input={<OutlinedInput label="Demographics" />}
                renderValue={(selected) => (
                  <Box sx={{ display: 'flex', flexWrap: 'wrap', gap: 0.5 }}>
                    {selected.map((value) => (
                      <Chip key={value} label={value} size="small" />
                    ))}
                  </Box>
                )}
              >
                {DEMOGRAPHICS_OPTIONS.map((option) => (
                  <MenuItem key={option} value={option}>
                    {option}
                  </MenuItem>
                ))}
              </Select>
            </FormControl>
          </Grid>
        </Grid>
      </Paper>

      {/* Categories */}
      <CategoryEditor
        categories={surveyData.categories || []}
        setCategories={(categories) => handleChange('categories', categories)}
      />

      {/* Questions */}
      <QuestionEditor
        questions={surveyData.questions}
        setQuestions={(questions) => handleChange('questions', questions)}
        categories={surveyData.categories || []}
      />

      {/* Persona Groups */}
      <PersonaGroupEditor
        personaGroups={surveyData.persona_groups}
        setPersonaGroups={(groups) => handleChange('persona_groups', groups)}
      />
    </Box>
  );
};

export default SurveyForm;
