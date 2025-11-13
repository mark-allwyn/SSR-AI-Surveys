/**
 * Question Editor Component
 * Add, edit, and remove survey questions
 */

import React, { useState } from 'react';
import {
  Box,
  Paper,
  Typography,
  Button,
  TextField,
  Grid,
  FormControl,
  InputLabel,
  Select,
  MenuItem,
  IconButton,
  Card,
  CardContent,
  CardActions,
  Chip,
  OutlinedInput,
} from '@mui/material';
import { Add as AddIcon, Delete as DeleteIcon, Edit as EditIcon } from '@mui/icons-material';
import { Question, QuestionType, Category, QUESTION_TYPES } from '../../services/types';

interface QuestionEditorProps {
  questions: Question[];
  setQuestions: (questions: Question[]) => void;
  categories: Category[];
}

const QuestionEditor: React.FC<QuestionEditorProps> = ({ questions, setQuestions, categories }) => {
  const [editingQuestion, setEditingQuestion] = useState<Partial<Question> | null>(null);
  const [editingIndex, setEditingIndex] = useState<number | null>(null);

  const handleAdd = () => {
    setEditingQuestion({
      id: `q${questions.length + 1}`,
      text: '',
      type: 'likert_5',
    });
    setEditingIndex(null);
  };

  const handleEdit = (index: number) => {
    setEditingQuestion({ ...questions[index] });
    setEditingIndex(index);
  };

  const handleSave = () => {
    if (!editingQuestion || !editingQuestion.text) return;

    const newQuestion: Question = {
      id: editingQuestion.id || `q${questions.length + 1}`,
      text: editingQuestion.text,
      type: editingQuestion.type || 'likert_5',
      category: editingQuestion.category,
      categories_compared: editingQuestion.categories_compared,
      options: editingQuestion.options,
      scale: editingQuestion.scale,
    };

    if (editingIndex !== null) {
      // Update existing question
      const updated = [...questions];
      updated[editingIndex] = newQuestion;
      setQuestions(updated);
    } else {
      // Add new question
      setQuestions([...questions, newQuestion]);
    }

    setEditingQuestion(null);
    setEditingIndex(null);
  };

  const handleDelete = (index: number) => {
    const updated = questions.filter((_, i) => i !== index);
    setQuestions(updated);
  };

  const handleCancel = () => {
    setEditingQuestion(null);
    setEditingIndex(null);
  };

  return (
    <Paper sx={{ p: 3, mb: 3 }}>
      <Box sx={{ display: 'flex', justifyContent: 'space-between', alignItems: 'center', mb: 2 }}>
        <Typography variant="h5">Questions</Typography>
        <Button startIcon={<AddIcon />} variant="outlined" onClick={handleAdd}>
          Add Question
        </Button>
      </Box>

      {/* Existing Questions */}
      <Grid container spacing={2} sx={{ mb: 2 }}>
        {questions.map((q, index) => (
          <Grid item xs={12} key={index}>
            <Card variant="outlined">
              <CardContent>
                <Box sx={{ display: 'flex', justifyContent: 'space-between', alignItems: 'flex-start' }}>
                  <Box sx={{ flexGrow: 1 }}>
                    <Typography variant="subtitle2" color="text.secondary" gutterBottom>
                      {q.id}
                    </Typography>
                    <Typography variant="body1" gutterBottom>
                      {q.text}
                    </Typography>
                    <Box sx={{ display: 'flex', gap: 1, flexWrap: 'wrap', mt: 1 }}>
                      <Chip label={q.type} size="small" color="primary" />
                      {q.category && <Chip label={`Category: ${q.category}`} size="small" />}
                      {q.categories_compared && q.categories_compared.length > 0 && (
                        <Chip label={`Comparing: ${q.categories_compared.join(', ')}`} size="small" />
                      )}
                    </Box>
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
      {editingQuestion && (
        <Card sx={{ mt: 2, backgroundColor: '#f9f9f9' }}>
          <CardContent>
            <Typography variant="h6" gutterBottom>
              {editingIndex !== null ? 'Edit Question' : 'New Question'}
            </Typography>
            <Grid container spacing={2}>
              <Grid item xs={12}>
                <TextField
                  fullWidth
                  label="Question ID"
                  value={editingQuestion.id || ''}
                  onChange={(e) => setEditingQuestion({ ...editingQuestion, id: e.target.value })}
                  required
                />
              </Grid>
              <Grid item xs={12}>
                <TextField
                  fullWidth
                  label="Question Text"
                  value={editingQuestion.text || ''}
                  onChange={(e) => setEditingQuestion({ ...editingQuestion, text: e.target.value })}
                  multiline
                  rows={2}
                  required
                />
              </Grid>
              <Grid item xs={12} md={6}>
                <FormControl fullWidth>
                  <InputLabel>Question Type</InputLabel>
                  <Select
                    value={editingQuestion.type || 'likert_5'}
                    label="Question Type"
                    onChange={(e) => setEditingQuestion({ ...editingQuestion, type: e.target.value as QuestionType })}
                  >
                    {QUESTION_TYPES.map((qt) => (
                      <MenuItem key={qt.value} value={qt.value}>
                        {qt.label}
                      </MenuItem>
                    ))}
                  </Select>
                </FormControl>
              </Grid>
              {categories.length > 0 && (
                <>
                  <Grid item xs={12} md={6}>
                    <FormControl fullWidth>
                      <InputLabel>Category (Single)</InputLabel>
                      <Select
                        value={editingQuestion.category || ''}
                        label="Category (Single)"
                        onChange={(e) => setEditingQuestion({ ...editingQuestion, category: e.target.value, categories_compared: undefined })}
                      >
                        <MenuItem value="">
                          <em>None</em>
                        </MenuItem>
                        {categories.map((cat) => (
                          <MenuItem key={cat.id} value={cat.id}>
                            {cat.name}
                          </MenuItem>
                        ))}
                      </Select>
                    </FormControl>
                  </Grid>
                  <Grid item xs={12}>
                    <FormControl fullWidth>
                      <InputLabel>Categories Compared (Multi)</InputLabel>
                      <Select
                        multiple
                        value={editingQuestion.categories_compared || []}
                        label="Categories Compared (Multi)"
                        onChange={(e) => setEditingQuestion({ ...editingQuestion, categories_compared: e.target.value as string[], category: undefined })}
                        input={<OutlinedInput label="Categories Compared (Multi)" />}
                        renderValue={(selected) => (
                          <Box sx={{ display: 'flex', flexWrap: 'wrap', gap: 0.5 }}>
                            {selected.map((value) => (
                              <Chip key={value} label={value} size="small" />
                            ))}
                          </Box>
                        )}
                      >
                        {categories.map((cat) => (
                          <MenuItem key={cat.id} value={cat.id}>
                            {cat.name}
                          </MenuItem>
                        ))}
                      </Select>
                    </FormControl>
                  </Grid>
                </>
              )}

              {/* Answer Options - for multiple choice, yes/no, etc. */}
              {(editingQuestion.type === 'multiple_choice' || editingQuestion.type === 'yes_no') && (
                <Grid item xs={12}>
                  <Typography variant="subtitle2" gutterBottom>
                    Answer Options (one per line)
                  </Typography>
                  <TextField
                    fullWidth
                    multiline
                    rows={4}
                    value={(editingQuestion.options || []).join('\n')}
                    onChange={(e) => {
                      const options = e.target.value.split('\n').filter(opt => opt.trim() !== '');
                      setEditingQuestion({ ...editingQuestion, options });
                    }}
                    placeholder="Enter each option on a new line"
                    helperText="Each line will be a separate answer option"
                  />
                </Grid>
              )}

              {/* Scale Definition - for likert scales */}
              {editingQuestion.type?.includes('likert') && (
                <Grid item xs={12}>
                  <Typography variant="subtitle2" gutterBottom>
                    Scale Labels (format: 1: Label Text, one per line)
                  </Typography>
                  <TextField
                    fullWidth
                    multiline
                    rows={6}
                    value={
                      editingQuestion.scale
                        ? Object.entries(editingQuestion.scale)
                            .sort(([a], [b]) => Number(a) - Number(b))
                            .map(([key, value]) => `${key}: ${value}`)
                            .join('\n')
                        : ''
                    }
                    onChange={(e) => {
                      const lines = e.target.value.split('\n');
                      const scale: Record<number, string> = {};
                      lines.forEach(line => {
                        const match = line.match(/^(\d+):\s*(.+)$/);
                        if (match) {
                          scale[Number(match[1])] = match[2].trim();
                        }
                      });
                      setEditingQuestion({ ...editingQuestion, scale: Object.keys(scale).length > 0 ? scale : undefined });
                    }}
                    placeholder="1: Strongly Disagree&#10;2: Disagree&#10;3: Neutral&#10;4: Agree&#10;5: Strongly Agree"
                    helperText="Define the text label for each numeric value (e.g., '1: Strongly Disagree')"
                  />
                </Grid>
              )}

              {/* Scale Definition - for preference/comparison scales */}
              {editingQuestion.type?.includes('preference') && (
                <Grid item xs={12}>
                  <Typography variant="subtitle2" gutterBottom>
                    Comparison Scale Labels (format: 1: Label Text, one per line)
                  </Typography>
                  <TextField
                    fullWidth
                    multiline
                    rows={6}
                    value={
                      editingQuestion.scale
                        ? Object.entries(editingQuestion.scale)
                            .sort(([a], [b]) => Number(a) - Number(b))
                            .map(([key, value]) => `${key}: ${value}`)
                            .join('\n')
                        : ''
                    }
                    onChange={(e) => {
                      const lines = e.target.value.split('\n');
                      const scale: Record<number, string> = {};
                      lines.forEach(line => {
                        const match = line.match(/^(\d+):\s*(.+)$/);
                        if (match) {
                          scale[Number(match[1])] = match[2].trim();
                        }
                      });
                      setEditingQuestion({ ...editingQuestion, scale: Object.keys(scale).length > 0 ? scale : undefined });
                    }}
                    placeholder={
                      editingQuestion.categories_compared && editingQuestion.categories_compared.length >= 2
                        ? `1: Strongly prefer {${editingQuestion.categories_compared[0]}}&#10;2: Prefer {${editingQuestion.categories_compared[0]}}&#10;3: No preference&#10;4: Prefer {${editingQuestion.categories_compared[1]}}&#10;5: Strongly prefer {${editingQuestion.categories_compared[1]}}`
                        : "1: Strongly prefer first option&#10;2: Prefer first option&#10;3: No preference&#10;4: Prefer second option&#10;5: Strongly prefer second option"
                    }
                    helperText={
                      editingQuestion.categories_compared && editingQuestion.categories_compared.length >= 2
                        ? `Use {${editingQuestion.categories_compared[0]}} and {${editingQuestion.categories_compared[1]}} as placeholders for the category names`
                        : "Use {category_id} as placeholders for category names (select categories first)"
                    }
                  />
                </Grid>
              )}
            </Grid>
          </CardContent>
          <CardActions>
            <Button onClick={handleCancel}>Cancel</Button>
            <Button variant="contained" onClick={handleSave} disabled={!editingQuestion.text || !editingQuestion.id}>
              {editingIndex !== null ? 'Update Question' : 'Add Question'}
            </Button>
          </CardActions>
        </Card>
      )}

      {questions.length === 0 && !editingQuestion && (
        <Typography color="text.secondary" sx={{ textAlign: 'center', py: 4 }}>
          No questions added yet. Click "Add Question" to get started.
        </Typography>
      )}
    </Paper>
  );
};

export default QuestionEditor;
