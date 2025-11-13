/**
 * Persona Group List Component
 * Display persona groups
 */

import React from 'react';
import { Paper, Typography, Grid, Card, CardContent, Chip, Box } from '@mui/material';
import { PersonaGroup } from '../../services/types';

interface PersonaGroupListProps {
  personaGroups: PersonaGroup[];
}

const PersonaGroupList: React.FC<PersonaGroupListProps> = ({ personaGroups }) => {
  return (
    <Paper sx={{ p: 3 }}>
      <Typography variant="h5" gutterBottom>
        Persona Groups ({personaGroups.length})
      </Typography>

      <Grid container spacing={2} sx={{ mt: 1 }}>
        {personaGroups.map((group, index) => (
          <Grid item xs={12} md={6} key={index}>
            <Card variant="outlined">
              <CardContent>
                <Box sx={{ mb: 2 }}>
                  <Typography variant="h6" gutterBottom>
                    {group.name}
                    <Chip label={`Weight: ${group.weight}`} size="small" sx={{ ml: 1 }} />
                  </Typography>
                  <Typography variant="body2" color="text.secondary">
                    {group.description}
                  </Typography>
                </Box>

                <Typography variant="subtitle2" gutterBottom>
                  Personas:
                </Typography>
                <Box sx={{ display: 'flex', gap: 1, flexWrap: 'wrap', mb: 2 }}>
                  {group.personas.map((persona, i) => (
                    <Chip key={i} label={persona} size="small" variant="outlined" />
                  ))}
                </Box>

                <Typography variant="subtitle2" gutterBottom>
                  Target Demographics:
                </Typography>
                <Grid container spacing={1}>
                  {group.target_demographics.gender.length > 0 && (
                    <Grid item xs={12}>
                      <Typography variant="caption" color="text.secondary">
                        Genders:
                      </Typography>
                      <Box sx={{ display: 'flex', gap: 0.5, flexWrap: 'wrap', mt: 0.5 }}>
                        {group.target_demographics.gender.map((g, i) => (
                          <Chip key={i} label={g} size="small" />
                        ))}
                      </Box>
                    </Grid>
                  )}
                  {group.target_demographics.age_group.length > 0 && (
                    <Grid item xs={12}>
                      <Typography variant="caption" color="text.secondary">
                        Age Groups:
                      </Typography>
                      <Box sx={{ display: 'flex', gap: 0.5, flexWrap: 'wrap', mt: 0.5 }}>
                        {group.target_demographics.age_group.map((a, i) => (
                          <Chip key={i} label={a} size="small" />
                        ))}
                      </Box>
                    </Grid>
                  )}
                  {group.target_demographics.occupation.length > 0 && (
                    <Grid item xs={12}>
                      <Typography variant="caption" color="text.secondary">
                        Occupations:
                      </Typography>
                      <Box sx={{ display: 'flex', gap: 0.5, flexWrap: 'wrap', mt: 0.5 }}>
                        {group.target_demographics.occupation.map((o, i) => (
                          <Chip key={i} label={o} size="small" />
                        ))}
                      </Box>
                    </Grid>
                  )}
                </Grid>
              </CardContent>
            </Card>
          </Grid>
        ))}
      </Grid>
    </Paper>
  );
};

export default PersonaGroupList;
