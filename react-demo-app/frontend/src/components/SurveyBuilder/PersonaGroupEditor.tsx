/**
 * Persona Group Editor Component
 * Add, edit, and remove persona groups
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
  FormControl,
  InputLabel,
  Select,
  MenuItem,
  OutlinedInput,
  Chip,
} from '@mui/material';
import { Add as AddIcon, Delete as DeleteIcon } from '@mui/icons-material';
import { PersonaGroup, AGE_GROUPS, GENDERS, OCCUPATIONS } from '../../services/types';

interface PersonaGroupEditorProps {
  personaGroups: PersonaGroup[];
  setPersonaGroups: (groups: PersonaGroup[]) => void;
}

const PersonaGroupEditor: React.FC<PersonaGroupEditorProps> = ({ personaGroups, setPersonaGroups }) => {
  const [editingGroup, setEditingGroup] = useState<Partial<PersonaGroup> | null>(null);
  const [personaInput, setPersonaInput] = useState('');

  const handleAdd = () => {
    setEditingGroup({
      name: '',
      description: '',
      personas: [],
      target_demographics: {
        gender: [],
        age_group: [],
        occupation: [],
      },
      weight: 1.0,
    });
  };

  const handleSave = () => {
    if (!editingGroup || !editingGroup.name || !editingGroup.personas || editingGroup.personas.length === 0) return;

    const newGroup: PersonaGroup = {
      name: editingGroup.name,
      description: editingGroup.description || '',
      personas: editingGroup.personas,
      target_demographics: editingGroup.target_demographics || {
        gender: [],
        age_group: [],
        occupation: [],
      },
      weight: editingGroup.weight || 1.0,
    };

    setPersonaGroups([...personaGroups, newGroup]);
    setEditingGroup(null);
    setPersonaInput('');
  };

  const handleDelete = (index: number) => {
    const updated = personaGroups.filter((_, i) => i !== index);
    setPersonaGroups(updated);
  };

  const handleCancel = () => {
    setEditingGroup(null);
    setPersonaInput('');
  };

  const handleAddPersona = () => {
    if (!editingGroup || !personaInput.trim()) return;
    const personas = [...(editingGroup.personas || []), personaInput.trim()];
    setEditingGroup({ ...editingGroup, personas });
    setPersonaInput('');
  };

  const handleRemovePersona = (index: number) => {
    if (!editingGroup) return;
    const personas = editingGroup.personas?.filter((_, i) => i !== index) || [];
    setEditingGroup({ ...editingGroup, personas });
  };

  return (
    <Paper sx={{ p: 3, mb: 3 }}>
      <Box sx={{ display: 'flex', justifyContent: 'space-between', alignItems: 'center', mb: 2 }}>
        <Typography variant="h5">Persona Groups</Typography>
        <Button startIcon={<AddIcon />} variant="outlined" onClick={handleAdd}>
          Add Persona Group
        </Button>
      </Box>

      {/* Existing Persona Groups */}
      <Grid container spacing={2} sx={{ mb: 2 }}>
        {personaGroups.map((group, index) => (
          <Grid item xs={12} key={index}>
            <Card variant="outlined">
              <CardContent>
                <Box sx={{ display: 'flex', justifyContent: 'space-between', alignItems: 'flex-start' }}>
                  <Box sx={{ flexGrow: 1 }}>
                    <Typography variant="h6" gutterBottom>
                      {group.name} (Weight: {group.weight})
                    </Typography>
                    <Typography variant="body2" color="text.secondary" gutterBottom>
                      {group.description}
                    </Typography>
                    <Typography variant="subtitle2" sx={{ mt: 2, mb: 1 }}>
                      Personas:
                    </Typography>
                    <Box sx={{ display: 'flex', gap: 1, flexWrap: 'wrap', mb: 2 }}>
                      {group.personas.map((persona, i) => (
                        <Chip key={i} label={persona} size="small" />
                      ))}
                    </Box>
                    <Typography variant="subtitle2" gutterBottom>
                      Target Demographics:
                    </Typography>
                    <Grid container spacing={1}>
                      <Grid item xs={12} md={4}>
                        <Typography variant="caption" color="text.secondary">
                          Genders: {group.target_demographics.gender.join(', ') || 'All'}
                        </Typography>
                      </Grid>
                      <Grid item xs={12} md={4}>
                        <Typography variant="caption" color="text.secondary">
                          Ages: {group.target_demographics.age_group.join(', ') || 'All'}
                        </Typography>
                      </Grid>
                      <Grid item xs={12} md={4}>
                        <Typography variant="caption" color="text.secondary">
                          Occupations: {group.target_demographics.occupation.join(', ') || 'All'}
                        </Typography>
                      </Grid>
                    </Grid>
                  </Box>
                  <IconButton onClick={() => handleDelete(index)} color="error" size="small">
                    <DeleteIcon />
                  </IconButton>
                </Box>
              </CardContent>
            </Card>
          </Grid>
        ))}
      </Grid>

      {/* Add/Edit Form */}
      {editingGroup && (
        <Card sx={{ mt: 2, backgroundColor: '#f9f9f9' }}>
          <CardContent>
            <Typography variant="h6" gutterBottom>
              New Persona Group
            </Typography>
            <Grid container spacing={2}>
              <Grid item xs={12} md={8}>
                <TextField
                  fullWidth
                  label="Group Name"
                  value={editingGroup.name || ''}
                  onChange={(e) => setEditingGroup({ ...editingGroup, name: e.target.value })}
                  required
                />
              </Grid>
              <Grid item xs={12} md={4}>
                <TextField
                  fullWidth
                  type="number"
                  label="Weight"
                  value={editingGroup.weight || 1.0}
                  onChange={(e) => setEditingGroup({ ...editingGroup, weight: parseFloat(e.target.value) || 1.0 })}
                  inputProps={{ min: 0, step: 0.1 }}
                />
              </Grid>
              <Grid item xs={12}>
                <TextField
                  fullWidth
                  label="Description"
                  value={editingGroup.description || ''}
                  onChange={(e) => setEditingGroup({ ...editingGroup, description: e.target.value })}
                  multiline
                  rows={2}
                />
              </Grid>

              {/* Personas */}
              <Grid item xs={12}>
                <Typography variant="subtitle2" gutterBottom>
                  Personas
                </Typography>
                <Box sx={{ display: 'flex', gap: 1, mb: 1 }}>
                  <TextField
                    fullWidth
                    size="small"
                    label="Add persona description"
                    value={personaInput}
                    onChange={(e) => setPersonaInput(e.target.value)}
                    onKeyPress={(e) => e.key === 'Enter' && handleAddPersona()}
                  />
                  <Button onClick={handleAddPersona} variant="outlined">
                    Add
                  </Button>
                </Box>
                <Box sx={{ display: 'flex', gap: 1, flexWrap: 'wrap' }}>
                  {editingGroup.personas?.map((persona, i) => (
                    <Chip
                      key={i}
                      label={persona}
                      onDelete={() => handleRemovePersona(i)}
                      size="small"
                    />
                  ))}
                </Box>
              </Grid>

              {/* Target Demographics */}
              <Grid item xs={12}>
                <Typography variant="subtitle2" gutterBottom>
                  Target Demographics
                </Typography>
              </Grid>
              <Grid item xs={12} md={4}>
                <FormControl fullWidth size="small">
                  <InputLabel>Genders</InputLabel>
                  <Select
                    multiple
                    value={editingGroup.target_demographics?.gender || []}
                    label="Genders"
                    onChange={(e) => setEditingGroup({
                      ...editingGroup,
                      target_demographics: {
                        ...(editingGroup.target_demographics || { age_group: [], occupation: [] }),
                        gender: e.target.value as string[],
                      },
                    })}
                    input={<OutlinedInput label="Genders" />}
                    renderValue={(selected) => (
                      <Box sx={{ display: 'flex', flexWrap: 'wrap', gap: 0.5 }}>
                        {selected.map((value) => (
                          <Chip key={value} label={value} size="small" />
                        ))}
                      </Box>
                    )}
                  >
                    {GENDERS.map((gender) => (
                      <MenuItem key={gender} value={gender}>
                        {gender}
                      </MenuItem>
                    ))}
                  </Select>
                </FormControl>
              </Grid>
              <Grid item xs={12} md={4}>
                <FormControl fullWidth size="small">
                  <InputLabel>Age Groups</InputLabel>
                  <Select
                    multiple
                    value={editingGroup.target_demographics?.age_group || []}
                    label="Age Groups"
                    onChange={(e) => setEditingGroup({
                      ...editingGroup,
                      target_demographics: {
                        ...(editingGroup.target_demographics || { gender: [], occupation: [] }),
                        age_group: e.target.value as string[],
                      },
                    })}
                    input={<OutlinedInput label="Age Groups" />}
                    renderValue={(selected) => (
                      <Box sx={{ display: 'flex', flexWrap: 'wrap', gap: 0.5 }}>
                        {selected.map((value) => (
                          <Chip key={value} label={value} size="small" />
                        ))}
                      </Box>
                    )}
                  >
                    {AGE_GROUPS.map((age) => (
                      <MenuItem key={age} value={age}>
                        {age}
                      </MenuItem>
                    ))}
                  </Select>
                </FormControl>
              </Grid>
              <Grid item xs={12} md={4}>
                <FormControl fullWidth size="small">
                  <InputLabel>Occupations</InputLabel>
                  <Select
                    multiple
                    value={editingGroup.target_demographics?.occupation || []}
                    label="Occupations"
                    onChange={(e) => setEditingGroup({
                      ...editingGroup,
                      target_demographics: {
                        ...(editingGroup.target_demographics || { gender: [], age_group: [] }),
                        occupation: e.target.value as string[],
                      },
                    })}
                    input={<OutlinedInput label="Occupations" />}
                    renderValue={(selected) => (
                      <Box sx={{ display: 'flex', flexWrap: 'wrap', gap: 0.5 }}>
                        {selected.map((value) => (
                          <Chip key={value} label={value} size="small" />
                        ))}
                      </Box>
                    )}
                  >
                    {OCCUPATIONS.map((occ) => (
                      <MenuItem key={occ} value={occ}>
                        {occ}
                      </MenuItem>
                    ))}
                  </Select>
                </FormControl>
              </Grid>
            </Grid>
          </CardContent>
          <CardActions>
            <Button onClick={handleCancel}>Cancel</Button>
            <Button
              variant="contained"
              onClick={handleSave}
              disabled={!editingGroup.name || !editingGroup.personas || editingGroup.personas.length === 0}
            >
              Add Persona Group
            </Button>
          </CardActions>
        </Card>
      )}

      {personaGroups.length === 0 && !editingGroup && (
        <Typography color="text.secondary" sx={{ textAlign: 'center', py: 4 }}>
          No persona groups added yet. Click "Add Persona Group" to get started.
        </Typography>
      )}
    </Paper>
  );
};

export default PersonaGroupEditor;
