/**
 * Question List Component
 * Display survey questions
 */

import React from 'react';
import { Paper, Typography, Box, Card, CardContent, Chip, Grid } from '@mui/material';
import { Question, Category } from '../../services/types';

interface QuestionListProps {
  questions: Question[];
  categories?: Category[];
}

const QuestionList: React.FC<QuestionListProps> = ({ questions, categories }) => {
  const getCategoryName = (categoryId: string): string => {
    const cat = categories?.find((c) => c.id === categoryId);
    return cat?.name || categoryId;
  };

  return (
    <Paper sx={{ p: 3 }}>
      <Typography variant="h5" gutterBottom>
        Questions ({questions.length})
      </Typography>

      <Grid container spacing={2} sx={{ mt: 1 }}>
        {questions.map((question, index) => (
          <Grid item xs={12} key={index}>
            <Card variant="outlined">
              <CardContent>
                <Box sx={{ mb: 2 }}>
                  <Typography variant="caption" color="text.secondary">
                    {question.id}
                  </Typography>
                  <Typography variant="h6" gutterBottom>
                    {question.text}
                  </Typography>
                </Box>

                <Box sx={{ display: 'flex', gap: 1, flexWrap: 'wrap', mb: 2 }}>
                  <Chip label={question.type} size="small" color="primary" />
                  {question.category && (
                    <Chip label={`Category: ${getCategoryName(question.category)}`} size="small" color="secondary" />
                  )}
                  {question.categories_compared && question.categories_compared.length > 0 && (
                    <Chip
                      label={`Comparing: ${question.categories_compared.map(getCategoryName).join(' vs ')}`}
                      size="small"
                      color="secondary"
                    />
                  )}
                </Box>

                {question.options && question.options.length > 0 && (
                  <Box>
                    <Typography variant="subtitle2" color="text.secondary" gutterBottom>
                      Options:
                    </Typography>
                    <Box sx={{ display: 'flex', gap: 1, flexWrap: 'wrap' }}>
                      {question.options.map((option, i) => (
                        <Chip key={i} label={option} size="small" variant="outlined" />
                      ))}
                    </Box>
                  </Box>
                )}

                {question.scale && (
                  <Box>
                    <Typography variant="subtitle2" color="text.secondary" gutterBottom>
                      Scale:
                    </Typography>
                    <Typography variant="body2">
                      {Object.entries(question.scale).map(([key, value]) => `${key}: ${value}`).join(', ')}
                    </Typography>
                  </Box>
                )}
              </CardContent>
            </Card>
          </Grid>
        ))}
      </Grid>
    </Paper>
  );
};

export default QuestionList;
