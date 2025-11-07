/**
 * YAML Preview Component
 * Shows generated YAML from survey data
 */

import React from 'react';
import { Paper, Typography, Box } from '@mui/material';
import { SurveyBuilderState } from '../../services/types';

interface YAMLPreviewProps {
  surveyData: SurveyBuilderState;
}

const YAMLPreview: React.FC<YAMLPreviewProps> = ({ surveyData }) => {
  const generateYAML = (): string => {
    const yaml: string[] = [];

    yaml.push('survey:');
    yaml.push(`  name: "${surveyData.name || 'Untitled Survey'}"`);
    yaml.push(`  description: "${surveyData.description || 'No description'}"`);
    yaml.push(`  context: "${surveyData.context || 'No context provided'}"`);
    yaml.push(`  sample_size: ${surveyData.sample_size}`);

    // Demographics
    yaml.push('  demographics:');
    if (surveyData.demographics.length > 0) {
      surveyData.demographics.forEach(demo => {
        yaml.push(`    - ${demo}`);
      });
    } else {
      yaml.push(`    []`);
    }

    // Categories (if any)
    if (surveyData.categories && surveyData.categories.length > 0) {
      yaml.push('  categories:');
      surveyData.categories.forEach(cat => {
        yaml.push(`    - id: ${cat.id}`);
        yaml.push(`      name: "${cat.name}"`);
        yaml.push(`      description: "${cat.description}"`);
        yaml.push(`      context: "${cat.context}"`);
      });
    }

    // Questions
    yaml.push('  questions:');
    if (surveyData.questions.length > 0) {
      surveyData.questions.forEach(q => {
        yaml.push(`    - id: ${q.id}`);
        yaml.push(`      text: "${q.text}"`);
        yaml.push(`      type: ${q.type}`);
        if (q.category) {
          yaml.push(`      category: ${q.category}`);
        }
        if (q.categories_compared && q.categories_compared.length > 0) {
          yaml.push(`      categories_compared: [${q.categories_compared.join(', ')}]`);
        }
        if (q.options && q.options.length > 0) {
          yaml.push(`      options:`);
          q.options.forEach(opt => {
            yaml.push(`        - "${opt}"`);
          });
        }
      });
    } else {
      yaml.push(`    []`);
    }

    // Persona Groups
    yaml.push('  persona_groups:');
    if (surveyData.persona_groups.length > 0) {
      surveyData.persona_groups.forEach(pg => {
        yaml.push(`    - name: "${pg.name}"`);
        yaml.push(`      description: "${pg.description}"`);
        yaml.push(`      weight: ${pg.weight}`);
        yaml.push(`      personas:`);
        if (pg.personas.length > 0) {
          pg.personas.forEach(p => {
            yaml.push(`        - "${p}"`);
          });
        } else {
          yaml.push(`        []`);
        }
        yaml.push(`      target_demographics:`);
        yaml.push(`        gender: [${pg.target_demographics.gender.map(g => `"${g}"`).join(', ')}]`);
        yaml.push(`        age_group: [${pg.target_demographics.age_group.map(a => `"${a}"`).join(', ')}]`);
        yaml.push(`        occupation: [${pg.target_demographics.occupation.map(o => `"${o}"`).join(', ')}]`);
      });
    } else {
      yaml.push(`    []`);
    }

    return yaml.join('\n');
  };

  return (
    <Paper sx={{ p: 3 }}>
      <Typography variant="h6" gutterBottom>
        YAML Preview
      </Typography>
      <Typography variant="body2" color="text.secondary" gutterBottom>
        This is the YAML configuration that will be saved for your survey.
      </Typography>
      <Box
        component="pre"
        sx={{
          p: 2,
          mt: 2,
          backgroundColor: '#f5f5f5',
          borderRadius: 1,
          overflow: 'auto',
          fontFamily: 'monospace',
          fontSize: '0.875rem',
          maxHeight: '600px',
        }}
      >
        {generateYAML()}
      </Box>
    </Paper>
  );
};

export default YAMLPreview;
