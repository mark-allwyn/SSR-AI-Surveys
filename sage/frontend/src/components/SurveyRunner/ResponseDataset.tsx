/**
 * Response Dataset Component
 * Display LLM response dataset and distributions
 */

import React, { useState } from 'react';
import {
  Paper,
  Typography,
  Box,
  Table,
  TableBody,
  TableCell,
  TableContainer,
  TableHead,
  TableRow,
  Chip,
  Button,
  Accordion,
  AccordionSummary,
  AccordionDetails,
  Grid,
  TablePagination,
} from '@mui/material';
import { ExpandMore as ExpandMoreIcon, Download as DownloadIcon } from '@mui/icons-material';
import { BarChart, Bar, XAxis, YAxis, CartesianGrid, Tooltip, Legend, ResponsiveContainer } from 'recharts';
import { RunSurveyResponse, Survey } from '../../services/types';

interface ResponseDatasetProps {
  result: RunSurveyResponse;
  survey: Survey;
}

const ResponseDataset: React.FC<ResponseDatasetProps> = ({ result, survey }) => {
  const [page, setPage] = useState(0);
  const [rowsPerPage, setRowsPerPage] = useState(10);

  // Flatten distributions into table rows
  const rows = React.useMemo(() => {
    const data: any[] = [];
    Object.entries(result.distributions).forEach(([category, questions]) => {
      Object.entries(questions).forEach(([questionId, respondents]) => {
        Object.entries(respondents).forEach(([respondentId, dist]) => {
          data.push({
            category,
            questionId,
            respondentId,
            ...dist,
          });
        });
      });
    });
    return data;
  }, [result.distributions]);

  const handleChangePage = (_event: unknown, newPage: number) => {
    setPage(newPage);
  };

  const handleChangeRowsPerPage = (event: React.ChangeEvent<HTMLInputElement>) => {
    setRowsPerPage(parseInt(event.target.value, 10));
    setPage(0);
  };

  const handleExportCSV = () => {
    const csv = [
      ['Category', 'Question ID', 'Respondent ID', 'Mode', 'Expected Value', 'Entropy', 'Text Response', 'Gender', 'Age Group', 'Persona Group', 'Occupation'],
      ...rows.map((row) => [
        row.category,
        row.questionId,
        row.respondentId,
        row.mode,
        row.expected_value.toFixed(2),
        row.entropy.toFixed(3),
        `"${row.text_response.replace(/"/g, '""')}"`,
        row.gender,
        row.age_group,
        row.persona_group,
        row.occupation,
      ]),
    ]
      .map((row) => row.join(','))
      .join('\n');

    const blob = new Blob([csv], { type: 'text/csv' });
    const url = window.URL.createObjectURL(blob);
    const a = document.createElement('a');
    a.href = url;
    a.download = `survey_results_${result.run_id}.csv`;
    a.click();
    window.URL.revokeObjectURL(url);
  };

  const handleExportJSON = () => {
    const json = JSON.stringify(result, null, 2);
    const blob = new Blob([json], { type: 'application/json' });
    const url = window.URL.createObjectURL(blob);
    const a = document.createElement('a');
    a.href = url;
    a.download = `survey_results_${result.run_id}.json`;
    a.click();
    window.URL.revokeObjectURL(url);
  };

  return (
    <Paper sx={{ p: 3 }}>
      {/* Summary */}
      <Box sx={{ mb: 3 }}>
        <Typography variant="h5" gutterBottom>
          Survey Results
        </Typography>
        <Grid container spacing={2} sx={{ mt: 1 }}>
          <Grid item xs={6} md={3}>
            <Typography variant="subtitle2" color="text.secondary">
              Run ID
            </Typography>
            <Typography variant="body2">{result.run_id}</Typography>
          </Grid>
          <Grid item xs={6} md={3}>
            <Typography variant="subtitle2" color="text.secondary">
              Profiles
            </Typography>
            <Typography variant="body2">{result.num_profiles}</Typography>
          </Grid>
          <Grid item xs={6} md={3}>
            <Typography variant="subtitle2" color="text.secondary">
              Responses
            </Typography>
            <Typography variant="body2">{result.num_responses}</Typography>
          </Grid>
          <Grid item xs={6} md={3}>
            <Typography variant="subtitle2" color="text.secondary">
              Distributions
            </Typography>
            <Typography variant="body2">{result.num_distributions}</Typography>
          </Grid>
        </Grid>

        <Box sx={{ mt: 2 }}>
          <Typography variant="subtitle2" color="text.secondary" gutterBottom>
            Configuration
          </Typography>
          <Box sx={{ display: 'flex', gap: 1, flexWrap: 'wrap' }}>
            <Chip label={`${result.config.llm_provider}: ${result.config.model}`} size="small" />
            <Chip label={`LLM Temp: ${result.config.llm_temperature}`} size="small" />
            <Chip label={`SSR Temp: ${result.config.ssr_temperature}`} size="small" />
            <Chip label={`Normalize: ${result.config.normalize_method}`} size="small" />
          </Box>
        </Box>

        <Box sx={{ mt: 2, display: 'flex', gap: 2 }}>
          <Button variant="outlined" startIcon={<DownloadIcon />} onClick={handleExportCSV}>
            Download CSV
          </Button>
          <Button variant="outlined" startIcon={<DownloadIcon />} onClick={handleExportJSON}>
            Download JSON
          </Button>
        </Box>
      </Box>

      {/* Response Table */}
      <Accordion defaultExpanded>
        <AccordionSummary expandIcon={<ExpandMoreIcon />}>
          <Typography variant="h6">Response Dataset</Typography>
        </AccordionSummary>
        <AccordionDetails>
          <TableContainer>
            <Table size="small">
              <TableHead>
                <TableRow>
                  <TableCell>Category</TableCell>
                  <TableCell>Question</TableCell>
                  <TableCell>Respondent</TableCell>
                  <TableCell>Mode</TableCell>
                  <TableCell>Expected Value</TableCell>
                  <TableCell>Entropy</TableCell>
                  <TableCell>Demographics</TableCell>
                  <TableCell>Text Response</TableCell>
                </TableRow>
              </TableHead>
              <TableBody>
                {rows.slice(page * rowsPerPage, page * rowsPerPage + rowsPerPage).map((row, index) => (
                  <TableRow key={index} hover>
                    <TableCell>{row.category}</TableCell>
                    <TableCell>{row.questionId}</TableCell>
                    <TableCell>
                      <Typography variant="caption">{row.respondentId}</Typography>
                    </TableCell>
                    <TableCell>{row.mode}</TableCell>
                    <TableCell>{row.expected_value.toFixed(2)}</TableCell>
                    <TableCell>{row.entropy.toFixed(3)}</TableCell>
                    <TableCell>
                      <Box sx={{ display: 'flex', flexDirection: 'column', gap: 0.5 }}>
                        <Chip label={row.gender} size="small" />
                        <Chip label={row.age_group} size="small" />
                        <Chip label={row.occupation} size="small" />
                      </Box>
                    </TableCell>
                    <TableCell>
                      <Typography variant="caption" sx={{ maxWidth: 300, display: 'block' }}>
                        {row.text_response}
                      </Typography>
                    </TableCell>
                  </TableRow>
                ))}
              </TableBody>
            </Table>
          </TableContainer>
          <TablePagination
            rowsPerPageOptions={[5, 10, 25, 50]}
            component="div"
            count={rows.length}
            rowsPerPage={rowsPerPage}
            page={page}
            onPageChange={handleChangePage}
            onRowsPerPageChange={handleChangeRowsPerPage}
          />
        </AccordionDetails>
      </Accordion>

      {/* Distribution Visualizations for ALL Questions */}
      <Accordion sx={{ mt: 2 }}>
        <AccordionSummary expandIcon={<ExpandMoreIcon />}>
          <Typography variant="h6">Response Distribution by Question</Typography>
        </AccordionSummary>
        <AccordionDetails>
          <Box sx={{ display: 'flex', flexDirection: 'column', gap: 4 }}>
            {survey.questions.map((question) => {
              // Get all responses for this question
              const questionResponses = rows.filter(row => row.questionId === question.id);

              if (questionResponses.length === 0) return null;

              // Aggregate probabilities across all respondents
              const numOptions = questionResponses[0].probabilities.length;
              const aggregatedProbs = new Array(numOptions).fill(0);

              questionResponses.forEach(response => {
                response.probabilities.forEach((prob: number, index: number) => {
                  aggregatedProbs[index] += prob;
                });
              });

              // Average the probabilities
              const avgProbs = aggregatedProbs.map(sum => sum / questionResponses.length);

              // Get labels from question scale or use generic labels
              const getLabel = (index: number) => {
                const scaleValue = index + 1;
                if (question.scale && question.scale[scaleValue]) {
                  return question.scale[scaleValue];
                }
                if (question.options && question.options[index]) {
                  return question.options[index];
                }
                return `Option ${scaleValue}`;
              };

              const chartData = avgProbs.map((prob, index) => ({
                label: getLabel(index),
                probability: prob,
                value: index + 1,
              }));

              return (
                <Box key={question.id}>
                  <Typography variant="subtitle1" fontWeight="bold" gutterBottom>
                    {question.text}
                  </Typography>
                  <Typography variant="caption" color="text.secondary" gutterBottom display="block">
                    Averaged across {questionResponses.length} responses
                  </Typography>
                  <ResponsiveContainer width="100%" height={300}>
                    <BarChart data={chartData}>
                      <CartesianGrid strokeDasharray="3 3" />
                      <XAxis
                        dataKey="label"
                        angle={-45}
                        textAnchor="end"
                        height={120}
                        interval={0}
                        tick={{ fontSize: 12 }}
                      />
                      <YAxis
                        label={{ value: 'Average Probability', angle: -90, position: 'insideLeft' }}
                        domain={[0, 1]}
                      />
                      <Tooltip
                        formatter={(value: number) => value.toFixed(3)}
                        labelFormatter={(label) => `${label}`}
                      />
                      <Bar dataKey="probability" fill="#1976d2" />
                    </BarChart>
                  </ResponsiveContainer>
                </Box>
              );
            })}
          </Box>
        </AccordionDetails>
      </Accordion>
    </Paper>
  );
};

export default ResponseDataset;
