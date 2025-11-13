/**
 * Survey User View Page
 * Shows what a survey respondent would see when taking the survey
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
  Radio,
  RadioGroup,
  FormControlLabel,
  Button,
  LinearProgress,
  Divider,
  Grid,
} from '@mui/material';
import { ArrowBack, ArrowForward, CheckCircle } from '@mui/icons-material';
import { useSurveys, useSurvey } from '../services/hooks';

const SurveyUserViewPage: React.FC = () => {
  const { surveyId: urlSurveyId } = useParams<{ surveyId?: string }>();
  const [selectedSurveyId, setSelectedSurveyId] = useState<string>(urlSurveyId || '');
  const [currentQuestionIndex, setCurrentQuestionIndex] = useState(0);
  const [answers, setAnswers] = useState<Record<string, string>>({});
  const [isComplete, setIsComplete] = useState(false);

  const { data: surveys, isLoading: surveysLoading, error: surveysError } = useSurveys();
  const {
    data: survey,
    isLoading: surveyLoading,
    error: surveyError,
  } = useSurvey(selectedSurveyId, { enabled: !!selectedSurveyId });

  const handleSurveyChange = (event: any) => {
    const newSurveyId = event.target.value;
    setSelectedSurveyId(newSurveyId);
    setCurrentQuestionIndex(0);
    setAnswers({});
    setIsComplete(false);
  };

  const handleAnswerChange = (questionId: string, answer: string) => {
    setAnswers({ ...answers, [questionId]: answer });
  };

  const handleNext = () => {
    if (survey && currentQuestionIndex < survey.questions.length - 1) {
      setCurrentQuestionIndex(currentQuestionIndex + 1);
    }
  };

  const handlePrevious = () => {
    if (currentQuestionIndex > 0) {
      setCurrentQuestionIndex(currentQuestionIndex - 1);
    }
  };

  const handleSubmit = () => {
    setIsComplete(true);
  };

  const progress = survey ? ((currentQuestionIndex + 1) / survey.questions.length) * 100 : 0;
  const currentQuestion = survey?.questions[currentQuestionIndex];
  const isAnswered = currentQuestion ? !!answers[currentQuestion.id] : false;

  if (isComplete) {
    return (
      <Box sx={{ maxWidth: 800, mx: 'auto', mt: 4 }}>
        <Paper sx={{ p: 4, textAlign: 'center' }}>
          <CheckCircle sx={{ fontSize: 80, color: 'success.main', mb: 2 }} />
          <Typography variant="h4" gutterBottom>
            Survey Complete!
          </Typography>
          <Typography variant="body1" color="text.secondary" paragraph>
            Thank you for completing the survey.
          </Typography>
          <Button
            variant="outlined"
            onClick={() => {
              setIsComplete(false);
              setCurrentQuestionIndex(0);
              setAnswers({});
            }}
          >
            Start Over
          </Button>
        </Paper>
      </Box>
    );
  }

  return (
    <Box>
      {/* Header */}
      <Box sx={{ mb: 4 }}>
        <Typography variant="h3" component="h1" gutterBottom>
          Survey User View
        </Typography>
        <Typography variant="body1" color="text.secondary">
          Experience the survey as a respondent would see it
        </Typography>
      </Box>

      {/* Survey Selector */}
      {!selectedSurveyId && (
        <Paper sx={{ p: 3, mb: 3 }}>
          <FormControl fullWidth>
            <InputLabel>Select Survey to Preview</InputLabel>
            <Select
              value={selectedSurveyId}
              label="Select Survey to Preview"
              onChange={handleSurveyChange}
              disabled={surveysLoading}
            >
              <MenuItem value="">
                <em>Choose a survey...</em>
              </MenuItem>
              {surveys?.map((s) => (
                <MenuItem key={s.id} value={s.id}>
                  {s.name}
                </MenuItem>
              ))}
            </Select>
          </FormControl>
        </Paper>
      )}

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

      {/* Survey View */}
      {survey && !surveyLoading && (
        <Box sx={{ maxWidth: 800, mx: 'auto' }}>
          {/* Survey Header */}
          <Paper sx={{ p: 3, mb: 3 }}>
            <Typography variant="h4" gutterBottom>
              {survey.name}
            </Typography>
            {survey.description && (
              <Typography variant="body1" color="text.secondary" paragraph>
                {survey.description}
              </Typography>
            )}
            {survey.context && (
              <Box sx={{ mb: 2, p: 3, bgcolor: 'info.light', borderRadius: 1, border: '1px solid', borderColor: 'info.main' }}>
                <Typography variant="subtitle2" fontWeight="bold" color="info.dark" gutterBottom>
                  Survey Context
                </Typography>
                <Typography variant="body2" sx={{ whiteSpace: 'pre-line', lineHeight: 1.8 }}>
                  {survey.context}
                </Typography>
              </Box>
            )}
            <Box sx={{ mt: 2 }}>
              <LinearProgress variant="determinate" value={progress} />
              <Typography variant="caption" color="text.secondary" sx={{ mt: 1, display: 'block' }}>
                Question {currentQuestionIndex + 1} of {survey.questions.length}
              </Typography>
            </Box>
          </Paper>

          {/* Current Question */}
          {currentQuestion && (
            <Paper sx={{ p: 4, mb: 3 }}>
              {/* Show categories being compared side-by-side for comparison questions */}
              {currentQuestion.categories_compared && currentQuestion.categories_compared.length > 0 && survey.categories && (
                <Box sx={{ mb: 3 }}>
                  <Typography variant="subtitle2" fontWeight="bold" gutterBottom>
                    Comparing Services
                  </Typography>
                  <Grid container spacing={2}>
                    {currentQuestion.categories_compared.map((catId) => {
                      const category = survey.categories?.find(c => c.id === catId);
                      return category ? (
                        <Grid item xs={12} md={6} key={catId}>
                          <Box sx={{ p: 2, bgcolor: 'primary.light', borderRadius: 1, border: '1px solid', borderColor: 'primary.main', height: '100%' }}>
                            <Typography variant="subtitle2" fontWeight="bold" color="primary.dark" gutterBottom>
                              {category.name}
                            </Typography>
                            {category.description && (
                              <Typography variant="body2" color="primary.dark" paragraph sx={{ mb: category.context ? 1 : 0, fontSize: '0.875rem' }}>
                                {category.description}
                              </Typography>
                            )}
                            {category.context && (
                              <Typography variant="caption" color="primary.dark" sx={{ fontStyle: 'italic', whiteSpace: 'pre-line', lineHeight: 1.6, display: 'block' }}>
                                {category.context}
                              </Typography>
                            )}
                          </Box>
                        </Grid>
                      ) : null;
                    })}
                  </Grid>
                </Box>
              )}

              {/* Show category context if this question has a single category (not comparison) */}
              {currentQuestion.category && !currentQuestion.categories_compared && survey.categories && (
                <Box sx={{ mb: 3, p: 3, bgcolor: 'primary.light', borderRadius: 1, border: '1px solid', borderColor: 'primary.main' }}>
                  {(() => {
                    const category = survey.categories.find(c => c.id === currentQuestion.category);
                    return category ? (
                      <>
                        <Typography variant="subtitle2" fontWeight="bold" color="primary.dark" gutterBottom>
                          {category.name}
                        </Typography>
                        {category.description && (
                          <Typography variant="body2" color="primary.dark" paragraph sx={{ mb: category.context ? 2 : 0 }}>
                            {category.description}
                          </Typography>
                        )}
                        {category.context && (
                          <Typography variant="body2" color="primary.dark" sx={{ fontStyle: 'italic', whiteSpace: 'pre-line', lineHeight: 1.8 }}>
                            {category.context}
                          </Typography>
                        )}
                      </>
                    ) : (
                      <Typography variant="caption" color="text.secondary">
                        Category: {currentQuestion.category}
                      </Typography>
                    );
                  })()}
                </Box>
              )}

              <Typography variant="h5" gutterBottom>
                {(() => {
                  // Replace category placeholders in question text
                  let questionText = currentQuestion.text;
                  if (currentQuestion.categories_compared && survey.categories) {
                    currentQuestion.categories_compared.forEach((catId) => {
                      const category = survey.categories?.find(c => c.id === catId);
                      if (category) {
                        questionText = questionText.replace(`{${catId}}`, category.name);
                      }
                    });
                  }
                  return questionText;
                })()}
              </Typography>

              <Divider sx={{ my: 3 }} />

              <FormControl component="fieldset" fullWidth>
                <RadioGroup
                  value={answers[currentQuestion.id] || ''}
                  onChange={(e) => handleAnswerChange(currentQuestion.id, e.target.value)}
                >
                  {(() => {
                    // Get answer options - only from explicit data, never defaults
                    let answerOptions: string[] = [];

                    // If explicit options exist, use them
                    if (currentQuestion.options && currentQuestion.options.length > 0) {
                      answerOptions = currentQuestion.options;
                    }
                    // If scale exists, use scale labels
                    else if (currentQuestion.scale && Object.keys(currentQuestion.scale).length > 0) {
                      answerOptions = Object.values(currentQuestion.scale);
                    }

                    // If no options available, show error
                    if (answerOptions.length === 0) {
                      return (
                        <Typography variant="body2" color="error" sx={{ p: 2, bgcolor: 'error.light', borderRadius: 1 }}>
                          This question has no answer options defined. Please check the survey configuration.
                        </Typography>
                      );
                    }

                    // Display the answer options
                    return answerOptions.map((option, index) => (
                      <FormControlLabel
                        key={index}
                        value={option}
                        control={<Radio />}
                        label={
                          <Typography variant="body1" sx={{ py: 1.5 }}>
                            {option}
                          </Typography>
                        }
                        sx={{
                          border: '2px solid',
                          borderColor: answers[currentQuestion.id] === option ? 'primary.main' : 'divider',
                          borderRadius: 2,
                          mb: 1.5,
                          mx: 0,
                          px: 2,
                          transition: 'all 0.2s',
                          '&:hover': {
                            bgcolor: 'action.hover',
                            borderColor: 'primary.light',
                          },
                        }}
                      />
                    ));
                  })()}
                </RadioGroup>
              </FormControl>
            </Paper>
          )}

          {/* Navigation Buttons */}
          <Box sx={{ display: 'flex', justifyContent: 'space-between', mb: 4 }}>
            <Button
              variant="outlined"
              startIcon={<ArrowBack />}
              onClick={handlePrevious}
              disabled={currentQuestionIndex === 0}
            >
              Previous
            </Button>

            {currentQuestionIndex < survey.questions.length - 1 ? (
              <Button
                variant="contained"
                endIcon={<ArrowForward />}
                onClick={handleNext}
                disabled={!isAnswered}
              >
                Next
              </Button>
            ) : (
              <Button
                variant="contained"
                color="success"
                endIcon={<CheckCircle />}
                onClick={handleSubmit}
                disabled={Object.keys(answers).length !== survey.questions.length}
              >
                Submit Survey
              </Button>
            )}
          </Box>

          {/* Question Overview */}
          <Paper sx={{ p: 3 }}>
            <Typography variant="h6" gutterBottom>
              Question Overview
            </Typography>

            {/* Show categories if they exist */}
            {survey.categories && survey.categories.length > 0 && (
              <Box sx={{ mb: 2 }}>
                <Typography variant="caption" color="text.secondary" gutterBottom display="block">
                  Categories:
                </Typography>
                <Box sx={{ display: 'flex', flexWrap: 'wrap', gap: 1, mt: 1 }}>
                  {survey.categories.map((cat, idx) => (
                    <Box key={cat.id} sx={{ display: 'flex', alignItems: 'center', gap: 0.5 }}>
                      <Box
                        sx={{
                          width: 12,
                          height: 12,
                          borderRadius: '50%',
                          bgcolor: `hsl(${(idx * 360) / survey.categories!.length}, 70%, 60%)`
                        }}
                      />
                      <Typography variant="caption">{cat.name}</Typography>
                    </Box>
                  ))}
                </Box>
              </Box>
            )}

            <Box sx={{ display: 'flex', flexWrap: 'wrap', gap: 1, mt: 2 }}>
              {survey.questions.map((q, index) => {
                // Find category color if question has a category
                const categoryIndex = survey.categories?.findIndex(c => c.id === q.category) ?? -1;
                const categoryColor = categoryIndex >= 0
                  ? `hsl(${(categoryIndex * 360) / (survey.categories?.length || 1)}, 70%, 60%)`
                  : undefined;

                return (
                  <Box
                    key={q.id}
                    onClick={() => setCurrentQuestionIndex(index)}
                    sx={{
                      width: 40,
                      height: 40,
                      display: 'flex',
                      alignItems: 'center',
                      justifyContent: 'center',
                      border: '2px solid',
                      borderColor: answers[q.id] ? 'success.main' : (categoryColor || 'divider'),
                      borderRadius: 1,
                      bgcolor: currentQuestionIndex === index ? 'primary.main' : (categoryColor ? `${categoryColor}20` : 'transparent'),
                      color: currentQuestionIndex === index ? 'white' : answers[q.id] ? 'success.main' : 'text.secondary',
                      cursor: 'pointer',
                      fontWeight: 'bold',
                      '&:hover': {
                        bgcolor: currentQuestionIndex === index ? 'primary.dark' : 'action.hover',
                      },
                    }}
                  >
                    {index + 1}
                  </Box>
                );
              })}
            </Box>
          </Paper>
        </Box>
      )}

      {/* Empty State */}
      {!selectedSurveyId && !surveysLoading && surveys && surveys.length > 0 && (
        <Paper sx={{ p: 6, textAlign: 'center' }}>
          <Typography variant="h6" color="text.secondary">
            Select a survey from the dropdown above to preview the user experience
          </Typography>
        </Paper>
      )}
    </Box>
  );
};

export default SurveyUserViewPage;
