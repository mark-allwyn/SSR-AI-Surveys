/**
 * Main App Component
 * Provides routing and theme configuration for S.A.G.E
 * (Synthetic Audience Generation Engine)
 * Updated with company brand colors
 */

import React from 'react';
import { BrowserRouter as Router, Routes, Route, Navigate } from 'react-router-dom';
import { QueryClient, QueryClientProvider } from '@tanstack/react-query';
import { ReactQueryDevtools } from '@tanstack/react-query-devtools';
import { ThemeProvider, createTheme } from '@mui/material/styles';
import CssBaseline from '@mui/material/CssBaseline';
import { Box } from '@mui/material';

// Layout components
import Layout from './components/Layout/Layout';

// Page components
import HomePage from './pages/HomePage';
import SurveyBuilderPage from './pages/SurveyBuilderPage';
import SurveyPreviewPage from './pages/SurveyPreviewPage';
import SurveyRunnerPage from './pages/SurveyRunnerPage';
import SurveyUserViewPage from './pages/SurveyUserViewPage';
import SurveyHistoryPage from './pages/SurveyHistoryPage';
import SurveyRunDetailPage from './pages/SurveyRunDetailPage';
import GroundTruthTestingPage from './pages/GroundTruthTestingPage';
import SystemOverviewPage from './pages/SystemOverviewPage';

// Create React Query client
const queryClient = new QueryClient({
  defaultOptions: {
    queries: {
      refetchOnWindowFocus: false,
      retry: 1,
      staleTime: 5 * 60 * 1000, // 5 minutes
    },
  },
});

// Company brand colors
const brandColors = {
  tealBlue: '#0B617C',      // Primary - Teal Blue
  tealDark: '#022C33',      // Primary - Teal Dark
  turquoise: '#27E2CC',     // Accent - Turquoise
  electricLime: '#C4F600',  // Accent - Electric Lime
  lightGrey: '#F1F5F6',     // Primary - Light Grey
  atomicOrange: '#F69560',  // Secondary - Atomic Orange
  cornflowerBlue: '#B09DF9' // Secondary - Cornflower Blue
};

// Create Material-UI theme
const theme = createTheme({
  palette: {
    mode: 'light',
    primary: {
      main: brandColors.tealBlue,
      light: brandColors.turquoise,
      dark: brandColors.tealDark,
    },
    secondary: {
      main: brandColors.atomicOrange,
      light: brandColors.cornflowerBlue,
      dark: brandColors.tealDark,
    },
    background: {
      default: brandColors.lightGrey,
      paper: '#ffffff',
    },
    success: {
      main: brandColors.electricLime,
    },
    info: {
      main: brandColors.turquoise,
    },
  },
  typography: {
    fontFamily: [
      '-apple-system',
      'BlinkMacSystemFont',
      '"Segoe UI"',
      'Roboto',
      '"Helvetica Neue"',
      'Arial',
      'sans-serif',
    ].join(','),
    h1: {
      fontSize: '2.5rem',
      fontWeight: 600,
    },
    h2: {
      fontSize: '2rem',
      fontWeight: 600,
    },
    h3: {
      fontSize: '1.75rem',
      fontWeight: 500,
    },
    h4: {
      fontSize: '1.5rem',
      fontWeight: 500,
    },
    h5: {
      fontSize: '1.25rem',
      fontWeight: 500,
    },
    h6: {
      fontSize: '1rem',
      fontWeight: 500,
    },
  },
  components: {
    MuiButton: {
      styleOverrides: {
        root: {
          textTransform: 'none',
          borderRadius: 8,
        },
      },
    },
    MuiCard: {
      styleOverrides: {
        root: {
          borderRadius: 12,
          boxShadow: '0 2px 8px rgba(0,0,0,0.1)',
        },
      },
    },
    MuiPaper: {
      styleOverrides: {
        root: {
          borderRadius: 12,
        },
      },
    },
  },
});

const App: React.FC = () => {
  return (
    <QueryClientProvider client={queryClient}>
      <ThemeProvider theme={theme}>
        <CssBaseline />
        <Router>
          <Layout>
            <Box sx={{ minHeight: '100vh', bgcolor: 'background.default', py: 4 }}>
              <Routes>
                {/* Home page */}
                <Route path="/" element={<HomePage />} />

                {/* System Overview - understand the workflow */}
                <Route path="/overview" element={<SystemOverviewPage />} />

                {/* Survey Builder - create/edit surveys */}
                <Route path="/builder" element={<SurveyBuilderPage />} />

                {/* Survey Preview - view survey structure */}
                <Route path="/preview" element={<SurveyPreviewPage />} />
                <Route path="/preview/:surveyId" element={<SurveyPreviewPage />} />

                {/* Survey Runner - run surveys and view results */}
                <Route path="/runner" element={<SurveyRunnerPage />} />
                <Route path="/runner/:surveyId" element={<SurveyRunnerPage />} />

                {/* Survey User View - respondent experience */}
                <Route path="/user-view" element={<SurveyUserViewPage />} />
                <Route path="/user-view/:surveyId" element={<SurveyUserViewPage />} />

                {/* Survey History - view past runs */}
                <Route path="/history" element={<SurveyHistoryPage />} />
                <Route path="/history/:runId" element={<SurveyRunDetailPage />} />

                {/* Ground Truth Testing - experimentation */}
                <Route path="/ground-truth" element={<GroundTruthTestingPage />} />

                {/* Redirect unknown routes to home */}
                <Route path="*" element={<Navigate to="/" replace />} />
              </Routes>
            </Box>
          </Layout>
        </Router>
      </ThemeProvider>
      {/* React Query DevTools - only in development */}
      {process.env.NODE_ENV === 'development' && <ReactQueryDevtools initialIsOpen={false} />}
    </QueryClientProvider>
  );
};

export default App;
