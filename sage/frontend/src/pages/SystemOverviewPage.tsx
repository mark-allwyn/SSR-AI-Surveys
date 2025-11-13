/**
 * System Overview Page
 * Explains how the SSR Pipeline system works
 */

import React from 'react';
import {
  Box,
  Typography,
  Container,
} from '@mui/material';
import {
  Info as InfoIcon,
} from '@mui/icons-material';
import SystemWorkflowDiagram from '../components/SystemWorkflowDiagram';

const SystemOverviewPage: React.FC = () => {
  return (
    <Container maxWidth="xl">
      <Box sx={{ py: 4 }}>
        {/* Header */}
        <Box sx={{ mb: 4, display: 'flex', alignItems: 'center', gap: 2 }}>
          <InfoIcon sx={{ fontSize: 40, color: 'primary.main' }} />
          <Box>
            <Typography variant="h3" component="h1">
              System Overview
            </Typography>
            <Typography variant="body1" color="text.secondary">
              Understanding the SSR Pipeline workflow and methodology
            </Typography>
          </Box>
        </Box>

        {/* System Workflow Diagram */}
        <SystemWorkflowDiagram />
      </Box>
    </Container>
  );
};

export default SystemOverviewPage;
