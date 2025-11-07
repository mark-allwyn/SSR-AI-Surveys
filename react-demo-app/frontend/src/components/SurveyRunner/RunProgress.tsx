/**
 * Run Progress Component
 * Shows progress during survey execution
 */

import React from 'react';
import { Paper, Typography, Box, LinearProgress, List, ListItem, ListItemText } from '@mui/material';

interface RunProgressProps {
  progress: number;
  messages: string[];
}

const RunProgress: React.FC<RunProgressProps> = ({ progress, messages }) => {
  return (
    <Paper sx={{ p: 3 }}>
      <Typography variant="h5" gutterBottom>
        Running Survey...
      </Typography>
      <Box sx={{ my: 3 }}>
        <LinearProgress variant="determinate" value={progress} />
        <Typography variant="body2" color="text.secondary" sx={{ mt: 1, textAlign: 'right' }}>
          {progress}%
        </Typography>
      </Box>
      <Typography variant="body2" color="text.secondary" paragraph>
        This may take several minutes depending on the survey size and LLM provider.
      </Typography>
      <List
        dense
        sx={{
          maxHeight: 300,
          overflow: 'auto',
          bgcolor: 'background.default',
          borderRadius: 1,
          p: 1
        }}
      >
        {messages.map((message, index) => (
          <ListItem key={index}>
            <ListItemText
              primary={message}
              primaryTypographyProps={{
                variant: 'body2',
                fontFamily: 'monospace',
              }}
            />
          </ListItem>
        ))}
      </List>
    </Paper>
  );
};

export default RunProgress;
