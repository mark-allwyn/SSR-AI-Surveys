/**
 * Category List Component
 * Display survey categories
 */

import React from 'react';
import { Paper, Typography, Grid, Card, CardContent, Box } from '@mui/material';
import { Category } from '../../services/types';

interface CategoryListProps {
  categories: Category[];
}

const CategoryList: React.FC<CategoryListProps> = ({ categories }) => {
  return (
    <Paper sx={{ p: 3 }}>
      <Typography variant="h5" gutterBottom>
        Categories ({categories.length})
      </Typography>

      <Grid container spacing={2} sx={{ mt: 1 }}>
        {categories.map((category, index) => (
          <Grid item xs={12} md={6} key={index}>
            <Card variant="outlined">
              <CardContent>
                <Typography variant="caption" color="text.secondary">
                  {category.id}
                </Typography>
                <Typography variant="h6" gutterBottom>
                  {category.name}
                </Typography>
                <Typography variant="body2" color="text.secondary" paragraph>
                  {category.description}
                </Typography>
                {category.context && (
                  <Box sx={{ p: 2, backgroundColor: '#f5f5f5', borderRadius: 1 }}>
                    <Typography variant="caption" color="text.secondary">
                      Context:
                    </Typography>
                    <Typography variant="body2">
                      {category.context}
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

export default CategoryList;
