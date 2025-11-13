/**
 * Distribution Comparison Charts
 * Shows experiment vs ground truth probability distributions
 */

import React from 'react';
import {
  Box,
  Typography,
  Paper,
  Grid,
  Card,
  CardContent,
} from '@mui/material';
import {
  LineChart,
  Line,
  BarChart,
  Bar,
  XAxis,
  YAxis,
  CartesianGrid,
  Tooltip,
  Legend,
  ResponsiveContainer,
  ComposedChart,
  Area,
} from 'recharts';
import type { RatingDistribution, AggregatedDistribution, DistributionData } from '../services/types';

interface DistributionChartsProps {
  testRunDistributions?: DistributionData;
  groundTruthDistributions?: {
    [category: string]: {
      [question_id: string]: AggregatedDistribution;
    };
  };
}

const DistributionCharts: React.FC<DistributionChartsProps> = ({
  testRunDistributions,
  groundTruthDistributions,
}) => {
  if (!testRunDistributions || !groundTruthDistributions) {
    return null;
  }

  // Helper to aggregate test run distributions for a question
  const aggregateTestDistributions = (
    category: string,
    questionId: string
  ): number[] | null => {
    const categoryData = testRunDistributions[category];
    if (!categoryData) return null;

    const questionData = categoryData[questionId];
    if (!questionData) return null;

    // Aggregate probabilities across all respondents
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

  // Get all categories and questions
  const categories = Object.keys(groundTruthDistributions);

  return (
    <Box>
      <Typography variant="h5" gutterBottom sx={{ mt: 4, mb: 3 }}>
        Distribution Comparisons: Experiment vs Ground Truth
      </Typography>

      {categories.map((category) => {
        const gtCategoryData = groundTruthDistributions[category];
        const questionIds = Object.keys(gtCategoryData);

        return (
          <Box key={category} sx={{ mb: 5 }}>
            {categories.length > 1 && (
              <Typography variant="h6" gutterBottom sx={{ mb: 2 }}>
                Category: {category}
              </Typography>
            )}

            <Grid container spacing={3}>
              {questionIds.map((questionId) => {
                const gtDist = gtCategoryData[questionId];
                const testDist = aggregateTestDistributions(category, questionId);

                if (!testDist) return null;

                // Prepare chart data
                const numRatings = gtDist.mean_probabilities.length;
                const chartData = Array.from({ length: numRatings }, (_, i) => ({
                  rating: `${i + 1}`,
                  groundTruth: gtDist.mean_probabilities[i],
                  experiment: testDist[i],
                }));

                return (
                  <Grid item xs={12} md={6} key={`${category}-${questionId}`}>
                    <Card>
                      <CardContent>
                        <Typography variant="subtitle1" gutterBottom fontWeight="bold">
                          {questionId}
                        </Typography>
                        <ResponsiveContainer width="100%" height={300}>
                          <ComposedChart data={chartData}>
                            <CartesianGrid strokeDasharray="3 3" />
                            <XAxis
                              dataKey="rating"
                              label={{ value: 'Rating', position: 'insideBottom', offset: -5 }}
                            />
                            <YAxis
                              label={{ value: 'Probability', angle: -90, position: 'insideLeft' }}
                            />
                            <Tooltip
                              formatter={(value: number) => value.toFixed(3)}
                              contentStyle={{ backgroundColor: 'rgba(255, 255, 255, 0.95)' }}
                            />
                            <Legend />

                            {/* Ground truth as bars */}
                            <Bar
                              dataKey="groundTruth"
                              fill="#FF6E3A"
                              name="Ground Truth"
                              opacity={0.6}
                            />

                            {/* Experiment as line with area */}
                            <Area
                              type="monotone"
                              dataKey="experiment"
                              fill="#367588"
                              stroke="#367588"
                              fillOpacity={0.3}
                              name="Experiment (Area)"
                            />
                            <Line
                              type="monotone"
                              dataKey="experiment"
                              stroke="#367588"
                              strokeWidth={3}
                              dot={{ fill: '#367588', r: 5 }}
                              name="Experiment"
                            />
                          </ComposedChart>
                        </ResponsiveContainer>

                        {/* Show sample size */}
                        <Typography variant="caption" color="text.secondary" sx={{ mt: 1 }}>
                          Ground Truth: {gtDist.sample_size} samples | Experiment:{' '}
                          {Object.keys(testRunDistributions[category]?.[questionId] || {}).length}{' '}
                          respondents
                        </Typography>
                      </CardContent>
                    </Card>
                  </Grid>
                );
              })}
            </Grid>
          </Box>
        );
      })}
    </Box>
  );
};

export default DistributionCharts;
