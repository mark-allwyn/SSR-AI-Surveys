/**
 * Category Editor Component
 * Add, edit, and remove survey categories
 */

import React, { useState } from 'react';
import {
  Box,
  Paper,
  Typography,
  Button,
  TextField,
  Grid,
  IconButton,
  Card,
  CardContent,
  CardActions,
} from '@mui/material';
import { Add as AddIcon, Delete as DeleteIcon, Edit as EditIcon } from '@mui/icons-material';
import { Category } from '../../services/types';

interface CategoryEditorProps {
  categories: Category[];
  setCategories: (categories: Category[]) => void;
}

const CategoryEditor: React.FC<CategoryEditorProps> = ({ categories, setCategories }) => {
  const [editingCategory, setEditingCategory] = useState<Partial<Category> | null>(null);
  const [editingIndex, setEditingIndex] = useState<number | null>(null);

  const handleAdd = () => {
    setEditingCategory({
      id: `cat${categories.length + 1}`,
      name: '',
      description: '',
      context: '',
    });
    setEditingIndex(null);
  };

  const handleEdit = (index: number) => {
    setEditingCategory({ ...categories[index] });
    setEditingIndex(index);
  };

  const handleSave = () => {
    if (!editingCategory || !editingCategory.id || !editingCategory.name) return;

    const newCategory: Category = {
      id: editingCategory.id,
      name: editingCategory.name,
      description: editingCategory.description || '',
      context: editingCategory.context || '',
    };

    if (editingIndex !== null) {
      // Update existing category
      const updated = [...categories];
      updated[editingIndex] = newCategory;
      setCategories(updated);
    } else {
      // Add new category
      setCategories([...categories, newCategory]);
    }

    setEditingCategory(null);
    setEditingIndex(null);
  };

  const handleDelete = (index: number) => {
    const updated = categories.filter((_, i) => i !== index);
    setCategories(updated);
  };

  const handleCancel = () => {
    setEditingCategory(null);
    setEditingIndex(null);
  };

  return (
    <Paper sx={{ p: 3, mb: 3 }}>
      <Box sx={{ display: 'flex', justifyContent: 'space-between', alignItems: 'center', mb: 2 }}>
        <Typography variant="h5">Categories</Typography>
        <Button startIcon={<AddIcon />} variant="outlined" onClick={handleAdd}>
          Add Category
        </Button>
      </Box>

      <Typography variant="body2" color="text.secondary" gutterBottom>
        Optional: Add categories to compare multiple products or services in your survey.
      </Typography>

      {/* Existing Categories */}
      <Grid container spacing={2} sx={{ mb: 2, mt: 1 }}>
        {categories.map((cat, index) => (
          <Grid item xs={12} md={6} key={index}>
            <Card variant="outlined">
              <CardContent>
                <Box sx={{ display: 'flex', justifyContent: 'space-between', alignItems: 'flex-start' }}>
                  <Box sx={{ flexGrow: 1 }}>
                    <Typography variant="subtitle2" color="text.secondary">
                      {cat.id}
                    </Typography>
                    <Typography variant="h6" gutterBottom>
                      {cat.name}
                    </Typography>
                    <Typography variant="body2" color="text.secondary">
                      {cat.description}
                    </Typography>
                    {cat.context && (
                      <Typography variant="caption" color="text.secondary" sx={{ mt: 1, display: 'block', fontStyle: 'italic' }}>
                        Context: {cat.context.substring(0, 50)}{cat.context.length > 50 ? '...' : ''}
                      </Typography>
                    )}
                  </Box>
                  <Box sx={{ display: 'flex', gap: 0.5 }}>
                    <IconButton onClick={() => handleEdit(index)} color="primary" size="small">
                      <EditIcon />
                    </IconButton>
                    <IconButton onClick={() => handleDelete(index)} color="error" size="small">
                      <DeleteIcon />
                    </IconButton>
                  </Box>
                </Box>
              </CardContent>
            </Card>
          </Grid>
        ))}
      </Grid>

      {/* Add/Edit Form */}
      {editingCategory && (
        <Card sx={{ mt: 2, backgroundColor: '#f9f9f9' }}>
          <CardContent>
            <Typography variant="h6" gutterBottom>
              {editingIndex !== null ? 'Edit Category' : 'New Category'}
            </Typography>
            <Grid container spacing={2}>
              <Grid item xs={12} md={6}>
                <TextField
                  fullWidth
                  label="Category ID"
                  value={editingCategory.id || ''}
                  onChange={(e) => setEditingCategory({ ...editingCategory, id: e.target.value })}
                  required
                />
              </Grid>
              <Grid item xs={12} md={6}>
                <TextField
                  fullWidth
                  label="Category Name"
                  value={editingCategory.name || ''}
                  onChange={(e) => setEditingCategory({ ...editingCategory, name: e.target.value })}
                  required
                />
              </Grid>
              <Grid item xs={12}>
                <TextField
                  fullWidth
                  label="Description"
                  value={editingCategory.description || ''}
                  onChange={(e) => setEditingCategory({ ...editingCategory, description: e.target.value })}
                  multiline
                  rows={2}
                />
              </Grid>
              <Grid item xs={12}>
                <TextField
                  fullWidth
                  label="Context"
                  value={editingCategory.context || ''}
                  onChange={(e) => setEditingCategory({ ...editingCategory, context: e.target.value })}
                  multiline
                  rows={2}
                  helperText="Additional context for this category"
                />
              </Grid>
            </Grid>
          </CardContent>
          <CardActions>
            <Button onClick={handleCancel}>Cancel</Button>
            <Button variant="contained" onClick={handleSave} disabled={!editingCategory.id || !editingCategory.name}>
              {editingIndex !== null ? 'Update Category' : 'Add Category'}
            </Button>
          </CardActions>
        </Card>
      )}
    </Paper>
  );
};

export default CategoryEditor;
