/**
 * API Client for SSR Pipeline Demo Application
 * Provides functions to communicate with the FastAPI backend
 */

import axios, { AxiosInstance } from 'axios';
import {
  Survey,
  SurveyListItem,
  CreateSurveyRequest,
  GenerateProfilesRequest,
  GenerateProfilesResponse,
  GenerateResponsesRequest,
  GenerateResponsesResponse,
  ApplySSRRequest,
  ApplySSRResponse,
  RunSurveyRequest,
  RunSurveyResponse,
  SurveyRunMetadata,
  SurveyRunDetail,
  GroundTruthMetadata,
  GroundTruth,
  CreateGroundTruthFromSSRRequest,
  UploadGroundTruthRequest,
  ComparisonResults,
} from './types';

// Create axios instance with base configuration
const api: AxiosInstance = axios.create({
  baseURL: process.env.REACT_APP_API_URL || 'http://localhost:8000',
  headers: {
    'Content-Type': 'application/json',
  },
  timeout: 300000, // 5 minutes for long-running operations
});

// Add request interceptor for logging (development only)
if (process.env.NODE_ENV === 'development') {
  api.interceptors.request.use(
    (config) => {
      console.log('API Request:', config.method?.toUpperCase(), config.url);
      return config;
    },
    (error) => {
      console.error('API Request Error:', error);
      return Promise.reject(error);
    }
  );

  api.interceptors.response.use(
    (response) => {
      console.log('API Response:', response.status, response.config.url);
      return response;
    },
    (error) => {
      console.error('API Response Error:', error.response?.status, error.config?.url);
      return Promise.reject(error);
    }
  );
}

// ===================
// Survey Management
// ===================

/**
 * Get list of all available surveys
 */
export const getSurveys = async (): Promise<SurveyListItem[]> => {
  const response = await api.get<SurveyListItem[]>('/api/surveys');
  return response.data;
};

/**
 * Get detailed survey configuration
 */
export const getSurvey = async (surveyId: string): Promise<Survey> => {
  const response = await api.get<Survey>(`/api/surveys/${surveyId}`);
  return response.data;
};

/**
 * Create new survey from YAML content
 */
export const createSurvey = async (
  request: CreateSurveyRequest
): Promise<{ survey_id: string; status: string; path: string }> => {
  const response = await api.post('/api/surveys', request);
  return response.data;
};

/**
 * Update existing survey with new YAML content
 */
export const updateSurvey = async (
  surveyId: string,
  request: CreateSurveyRequest
): Promise<{ survey_id: string; status: string; path: string }> => {
  const response = await api.put(`/api/surveys/${surveyId}`, request);
  return response.data;
};

/**
 * Delete a survey
 */
export const deleteSurvey = async (
  surveyId: string
): Promise<{ survey_id: string; status: string }> => {
  const response = await api.delete(`/api/surveys/${surveyId}`);
  return response.data;
};

// ===================
// Profile Generation
// ===================

/**
 * Generate respondent profiles from persona groups
 */
export const generateProfiles = async (
  request: GenerateProfilesRequest
): Promise<GenerateProfilesResponse> => {
  const response = await api.post<GenerateProfilesResponse>('/api/generate-profiles', request);
  return response.data;
};

// ===================
// Response Generation
// ===================

/**
 * Generate LLM text responses for survey
 */
export const generateResponses = async (
  request: GenerateResponsesRequest
): Promise<GenerateResponsesResponse> => {
  const response = await api.post<GenerateResponsesResponse>('/api/generate-responses', request);
  return response.data;
};

// ===================
// SSR Application
// ===================

/**
 * Apply SSR to convert text responses to probability distributions
 */
export const applySSR = async (request: ApplySSRRequest): Promise<ApplySSRResponse> => {
  const response = await api.post<ApplySSRResponse>('/api/apply-ssr', request);
  return response.data;
};

// ===================
// Complete Pipeline
// ===================

/**
 * Run complete survey pipeline (profiles → responses → SSR)
 * This is the main endpoint for the "Run Survey" feature
 */
export const runSurvey = async (request: RunSurveyRequest): Promise<RunSurveyResponse> => {
  const response = await api.post<RunSurveyResponse>('/api/run-survey', request);
  return response.data;
};

// ===================
// Health Check
// ===================

/**
 * Check if API is responding
 */
export const healthCheck = async (): Promise<{ message: string; version: string }> => {
  const response = await api.get('/');
  return response.data;
};

// ===================
// Survey History
// ===================

/**
 * Get list of all survey runs with optional filtering by survey_id
 */
export const getSurveyRuns = async (surveyId?: string): Promise<SurveyRunMetadata[]> => {
  const url = surveyId ? `/api/survey-runs?survey_id=${surveyId}` : '/api/survey-runs';
  const response = await api.get<SurveyRunMetadata[]>(url);
  return response.data;
};

/**
 * Get detailed survey run information
 */
export const getSurveyRun = async (runId: string): Promise<SurveyRunDetail> => {
  const response = await api.get<SurveyRunDetail>(`/api/survey-runs/${runId}`);
  return response.data;
};

/**
 * Delete a survey run
 */
export const deleteSurveyRun = async (runId: string): Promise<{ run_id: string; status: string }> => {
  const response = await api.delete(`/api/survey-runs/${runId}`);
  return response.data;
};

// ===================
// Ground Truth
// ===================

/**
 * Get list of all ground truths with optional filtering by survey_id
 */
export const getGroundTruths = async (surveyId?: string): Promise<GroundTruthMetadata[]> => {
  const url = surveyId ? `/api/ground-truths?survey_id=${surveyId}` : '/api/ground-truths';
  const response = await api.get<GroundTruthMetadata[]>(url);
  return response.data;
};

/**
 * Get detailed ground truth information
 */
export const getGroundTruth = async (gtId: string): Promise<GroundTruth> => {
  const response = await api.get<GroundTruth>(`/api/ground-truths/${gtId}`);
  return response.data;
};

/**
 * Create ground truth from SSR pipeline
 */
export const createGroundTruthFromSSR = async (
  request: CreateGroundTruthFromSSRRequest
): Promise<{ id: string; status: string; name: string; num_profiles: number; num_responses: number }> => {
  const response = await api.post('/api/ground-truths/from-ssr', request, {
    timeout: 600000, // 10 minutes for ground truth generation
  });
  return response.data;
};

/**
 * Upload ground truth data
 */
export const uploadGroundTruth = async (
  request: UploadGroundTruthRequest
): Promise<{ id: string; status: string; name: string }> => {
  const response = await api.post('/api/ground-truths/from-upload', request);
  return response.data;
};

/**
 * Delete a ground truth
 */
export const deleteGroundTruth = async (gtId: string): Promise<{ id: string; status: string }> => {
  const response = await api.delete(`/api/ground-truths/${gtId}`);
  return response.data;
};

/**
 * Compare survey run to ground truth
 */
export const compareToGroundTruth = async (
  runId: string,
  groundTruthId: string
): Promise<ComparisonResults> => {
  const response = await api.post<ComparisonResults>(
    `/api/ground-truths/compare?run_id=${runId}&ground_truth_id=${groundTruthId}`
  );
  return response.data;
};

// ===================
// Error Handler
// ===================

/**
 * Extract error message from API error response
 */
export const getErrorMessage = (error: unknown): string => {
  if (axios.isAxiosError(error)) {
    if (error.response?.data?.detail) {
      // FastAPI validation error format
      if (typeof error.response.data.detail === 'string') {
        return error.response.data.detail;
      }
      // Array of validation errors
      if (Array.isArray(error.response.data.detail)) {
        return error.response.data.detail
          .map((err: any) => `${err.loc.join('.')}: ${err.msg}`)
          .join(', ');
      }
    }
    if (error.response?.statusText) {
      return error.response.statusText;
    }
    if (error.message) {
      return error.message;
    }
  }
  if (error instanceof Error) {
    return error.message;
  }
  return 'An unexpected error occurred';
};

export default api;
