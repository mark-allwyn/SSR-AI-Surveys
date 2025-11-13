/**
 * Confusion Matrix Component
 * Shows alignment between experiment and ground truth rating distributions
 */

import React from 'react';
import {
  Box,
  Typography,
  Paper,
  Grid,
  Tooltip,
} from '@mui/material';
import type { DistributionData, AggregatedDistribution } from '../services/types';

interface ConfusionMatrixProps {
  testRunDistributions?: DistributionData;
  groundTruthDistributions?: {
    [category: string]: {
      [question_id: string]: AggregatedDistribution;
    };
  };
}

interface MatrixData {
  matrix: number[][];
  labels: string[];
  maxValue: number;
}

const ConfusionMatrix: React.FC<ConfusionMatrixProps> = ({
  testRunDistributions,
  groundTruthDistributions,
}) => {
  if (!testRunDistributions || !groundTruthDistributions) {
    return null;
  }

  // Helper to aggregate test run distributions
  const aggregateTestDistributions = (
    category: string,
    questionId: string
  ): number[] | null => {
    const categoryData = testRunDistributions[category];
    if (!categoryData) return null;

    const questionData = categoryData[questionId];
    if (!questionData) return null;

    const respondentIds = Object.keys(questionData);
    if (respondentIds.length === 0) return null;

    const firstDist = questionData[respondentIds[0]];
    const numRatings = firstDist.probabilities.length;

    const sumProbs = new Array(numRatings).fill(0);
    respondentIds.forEach((rid) => {
      const dist = questionData[rid];
      dist.probabilities.forEach((prob, idx) => {
        sumProbs[idx] += prob;
      });
    });

    return sumProbs.map((sum) => sum / respondentIds.length);
  };

  // Build aggregated confusion matrix across all questions
  const buildAggregatedMatrix = (): MatrixData | null => {
    const categories = Object.keys(groundTruthDistributions);
    if (categories.length === 0) return null;

    // Determine matrix size from first question
    const firstCategory = categories[0];
    const firstQuestionId = Object.keys(groundTruthDistributions[firstCategory])[0];
    const firstGtDist = groundTruthDistributions[firstCategory][firstQuestionId];
    const numRatings = firstGtDist.mean_probabilities.length;

    // Initialize matrix
    const matrix: number[][] = Array(numRatings)
      .fill(0)
      .map(() => Array(numRatings).fill(0));

    let questionCount = 0;

    // Aggregate across all questions
    categories.forEach((category) => {
      const gtCategoryData = groundTruthDistributions[category];
      const questionIds = Object.keys(gtCategoryData);

      questionIds.forEach((questionId) => {
        const gtDist = gtCategoryData[questionId];
        const testDist = aggregateTestDistributions(category, questionId);

        if (!testDist) return;

        // For each actual rating (ground truth)
        gtDist.mean_probabilities.forEach((gtProb, actualIdx) => {
          // For each predicted rating (experiment)
          testDist.forEach((testProb, predIdx) => {
            // Add the joint probability
            matrix[actualIdx][predIdx] += gtProb * testProb;
          });
        });

        questionCount++;
      });
    });

    // Normalize by number of questions
    if (questionCount > 0) {
      for (let i = 0; i < numRatings; i++) {
        for (let j = 0; j < numRatings; j++) {
          matrix[i][j] /= questionCount;
        }
      }
    }

    const maxValue = Math.max(...matrix.flat());
    const labels = Array.from({ length: numRatings }, (_, i) => `${i + 1}`);

    return { matrix, labels, maxValue };
  };

  const matrixData = buildAggregatedMatrix();
  if (!matrixData) return null;

  const { matrix, labels, maxValue } = matrixData;

  // Color interpolation based on value
  const getColor = (value: number): string => {
    const intensity = maxValue > 0 ? value / maxValue : 0;
    // Gradient from light blue to dark blue
    const r = Math.round(54 + (255 - 54) * (1 - intensity));
    const g = Math.round(117 + (255 - 117) * (1 - intensity));
    const b = Math.round(136 + (255 - 136) * (1 - intensity));
    return `rgb(${r}, ${g}, ${b})`;
  };

  const cellSize = 60;
  const fontSize = 12;

  return (
    <Box sx={{ mt: 4, mb: 3 }}>
      <Typography variant="h5" gutterBottom>
        Confusion Matrix: Experiment vs Ground Truth
      </Typography>
      <Typography variant="body2" color="text.secondary" sx={{ mb: 3 }}>
        Shows how predicted ratings (experiment) align with actual ratings (ground truth).
        Darker colors indicate higher probability density.
      </Typography>

      <Paper sx={{ p: 3, display: 'inline-block' }}>
        <Box sx={{ position: 'relative' }}>
          {/* Y-axis label */}
          <Box
            sx={{
              position: 'absolute',
              left: -40,
              top: '50%',
              transform: 'translateY(-50%) rotate(-90deg)',
              transformOrigin: 'center',
            }}
          >
            <Typography variant="body2" fontWeight="bold">
              Ground Truth Rating
            </Typography>
          </Box>

          {/* Matrix with labels */}
          <Box sx={{ ml: 8 }}>
            {/* Column headers (Predicted) */}
            <Box sx={{ display: 'flex', mb: 1, ml: `${cellSize}px` }}>
              {labels.map((label) => (
                <Box
                  key={`col-${label}`}
                  sx={{
                    width: cellSize,
                    textAlign: 'center',
                  }}
                >
                  <Typography variant="body2" fontWeight="bold">
                    {label}
                  </Typography>
                </Box>
              ))}
            </Box>

            {/* Matrix rows */}
            {matrix.map((row, rowIdx) => (
              <Box key={`row-${rowIdx}`} sx={{ display: 'flex', mb: 0.5 }}>
                {/* Row label (Actual) */}
                <Box
                  sx={{
                    width: cellSize,
                    height: cellSize,
                    display: 'flex',
                    alignItems: 'center',
                    justifyContent: 'center',
                  }}
                >
                  <Typography variant="body2" fontWeight="bold">
                    {labels[rowIdx]}
                  </Typography>
                </Box>

                {/* Matrix cells */}
                {row.map((value, colIdx) => (
                  <Tooltip
                    key={`cell-${rowIdx}-${colIdx}`}
                    title={`Ground Truth: ${labels[rowIdx]}, Experiment: ${labels[colIdx]}, Value: ${value.toFixed(4)}`}
                    arrow
                  >
                    <Box
                      sx={{
                        width: cellSize,
                        height: cellSize,
                        bgcolor: getColor(value),
                        display: 'flex',
                        alignItems: 'center',
                        justifyContent: 'center',
                        border: '1px solid #ddd',
                        cursor: 'pointer',
                        transition: 'transform 0.1s',
                        '&:hover': {
                          transform: 'scale(1.05)',
                          zIndex: 1,
                          border: '2px solid #333',
                        },
                      }}
                    >
                      <Typography
                        variant="caption"
                        sx={{
                          fontSize: fontSize,
                          fontWeight: 'bold',
                          color: value / maxValue > 0.5 ? 'white' : 'black',
                        }}
                      >
                        {value.toFixed(3)}
                      </Typography>
                    </Box>
                  </Tooltip>
                ))}
              </Box>
            ))}

            {/* X-axis label */}
            <Box sx={{ textAlign: 'center', mt: 2, ml: `${cellSize}px` }}>
              <Typography variant="body2" fontWeight="bold">
                Experiment Rating
              </Typography>
            </Box>
          </Box>

          {/* Legend */}
          <Box sx={{ mt: 3 }}>
            <Typography variant="caption" color="text.secondary" gutterBottom display="block">
              Color Scale (Probability):
            </Typography>
            <Box sx={{ display: 'flex', alignItems: 'center', gap: 1 }}>
              <Typography variant="caption">0</Typography>
              <Box
                sx={{
                  width: 200,
                  height: 20,
                  background: 'linear-gradient(to right, rgb(255, 255, 255), rgb(54, 117, 136))',
                  border: '1px solid #ddd',
                }}
              />
              <Typography variant="caption">{maxValue.toFixed(3)}</Typography>
            </Box>
          </Box>

          {/* Interpretation guide */}
          <Box sx={{ mt: 3, p: 2, bgcolor: '#f5f5f5', borderRadius: 1 }}>
            <Typography variant="caption" fontWeight="bold" display="block" gutterBottom>
              How to Read This Matrix:
            </Typography>
            <Typography variant="caption" display="block" sx={{ mb: 0.5 }}>
              • <strong>Diagonal cells</strong> (top-left to bottom-right): Perfect agreement between experiment and ground truth
            </Typography>
            <Typography variant="caption" display="block" sx={{ mb: 0.5 }}>
              • <strong>Off-diagonal cells</strong>: Misalignment - experiment predicted different ratings than ground truth
            </Typography>
            <Typography variant="caption" display="block">
              • <strong>Darker colors</strong>: Higher probability density at that rating combination
            </Typography>
          </Box>
        </Box>
      </Paper>
    </Box>
  );
};

export default ConfusionMatrix;
