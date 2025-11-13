/**
 * React Query hooks for SSR Pipeline Demo Application
 * Provides data fetching and mutation hooks with caching and state management
 */

import { useQuery, useMutation, useQueryClient, UseQueryOptions, UseMutationOptions } from '@tanstack/react-query';
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
import * as api from './api';

// ===================
// Query Keys
// ===================

export const queryKeys = {
  surveys: ['surveys'] as const,
  survey: (id: string) => ['survey', id] as const,
  health: ['health'] as const,
  surveyRuns: (surveyId?: string) => surveyId ? ['surveyRuns', surveyId] : ['surveyRuns'] as const,
  surveyRun: (runId: string) => ['surveyRun', runId] as const,
  groundTruths: (surveyId?: string) => surveyId ? ['groundTruths', surveyId] : ['groundTruths'] as const,
  groundTruth: (gtId: string) => ['groundTruth', gtId] as const,
};

// ===================
// Survey Queries
// ===================

/**
 * Fetch list of all surveys
 */
export const useSurveys = (options?: Omit<UseQueryOptions<SurveyListItem[], Error>, 'queryKey' | 'queryFn'>) => {
  return useQuery<SurveyListItem[], Error>({
    queryKey: queryKeys.surveys,
    queryFn: api.getSurveys,
    staleTime: 5 * 60 * 1000, // 5 minutes
    ...options,
  });
};

/**
 * Fetch detailed survey configuration
 */
export const useSurvey = (
  surveyId: string,
  options?: Omit<UseQueryOptions<Survey, Error>, 'queryKey' | 'queryFn'>
) => {
  return useQuery<Survey, Error>({
    queryKey: queryKeys.survey(surveyId),
    queryFn: () => api.getSurvey(surveyId),
    enabled: !!surveyId,
    staleTime: 5 * 60 * 1000, // 5 minutes
    ...options,
  });
};

/**
 * Health check query
 */
export const useHealthCheck = (
  options?: Omit<UseQueryOptions<{ message: string; version: string }, Error>, 'queryKey' | 'queryFn'>
) => {
  return useQuery<{ message: string; version: string }, Error>({
    queryKey: queryKeys.health,
    queryFn: api.healthCheck,
    retry: 3,
    retryDelay: 1000,
    ...options,
  });
};

// ===================
// Survey Mutations
// ===================

/**
 * Create new survey
 */
export const useCreateSurvey = (
  options?: UseMutationOptions<{ survey_id: string; status: string; path: string }, Error, CreateSurveyRequest>
) => {
  const queryClient = useQueryClient();

  return useMutation<{ survey_id: string; status: string; path: string }, Error, CreateSurveyRequest>({
    mutationFn: api.createSurvey,
    onSuccess: () => {
      // Invalidate surveys list to refetch
      queryClient.invalidateQueries({ queryKey: queryKeys.surveys });
    },
    ...options,
  });
};

/**
 * Update existing survey
 */
export const useUpdateSurvey = (
  options?: UseMutationOptions<
    { survey_id: string; status: string; path: string },
    Error,
    { surveyId: string; request: CreateSurveyRequest }
  >
) => {
  const queryClient = useQueryClient();

  return useMutation<
    { survey_id: string; status: string; path: string },
    Error,
    { surveyId: string; request: CreateSurveyRequest }
  >({
    mutationFn: ({ surveyId, request }) => api.updateSurvey(surveyId, request),
    onSuccess: (data) => {
      // Invalidate surveys list and specific survey to refetch
      queryClient.invalidateQueries({ queryKey: queryKeys.surveys });
      queryClient.invalidateQueries({ queryKey: queryKeys.survey(data.survey_id) });
    },
    ...options,
  });
};

/**
 * Delete survey
 */
export const useDeleteSurvey = (
  options?: UseMutationOptions<{ survey_id: string; status: string }, Error, string>
) => {
  const queryClient = useQueryClient();

  return useMutation<{ survey_id: string; status: string }, Error, string>({
    mutationFn: api.deleteSurvey,
    onSuccess: (data) => {
      // Invalidate surveys list and remove specific survey from cache
      queryClient.invalidateQueries({ queryKey: queryKeys.surveys });
      queryClient.removeQueries({ queryKey: queryKeys.survey(data.survey_id) });
    },
    ...options,
  });
};

// ===================
// Profile Generation Mutations
// ===================

/**
 * Generate respondent profiles
 */
export const useGenerateProfiles = (
  options?: UseMutationOptions<GenerateProfilesResponse, Error, GenerateProfilesRequest>
) => {
  return useMutation<GenerateProfilesResponse, Error, GenerateProfilesRequest>({
    mutationFn: api.generateProfiles,
    ...options,
  });
};

// ===================
// Response Generation Mutations
// ===================

/**
 * Generate LLM text responses
 */
export const useGenerateResponses = (
  options?: UseMutationOptions<GenerateResponsesResponse, Error, GenerateResponsesRequest>
) => {
  return useMutation<GenerateResponsesResponse, Error, GenerateResponsesRequest>({
    mutationFn: api.generateResponses,
    ...options,
  });
};

// ===================
// SSR Application Mutations
// ===================

/**
 * Apply SSR to convert text to distributions
 */
export const useApplySSR = (options?: UseMutationOptions<ApplySSRResponse, Error, ApplySSRRequest>) => {
  return useMutation<ApplySSRResponse, Error, ApplySSRRequest>({
    mutationFn: api.applySSR,
    ...options,
  });
};

// ===================
// Complete Pipeline Mutations
// ===================

/**
 * Run complete survey pipeline
 */
export const useRunSurvey = (options?: UseMutationOptions<RunSurveyResponse, Error, RunSurveyRequest>) => {
  const queryClient = useQueryClient();

  return useMutation<RunSurveyResponse, Error, RunSurveyRequest>({
    mutationFn: api.runSurvey,
    onSuccess: () => {
      // Invalidate survey runs list to show the new run
      queryClient.invalidateQueries({ queryKey: queryKeys.surveyRuns() });
    },
    ...options,
  });
};

// ===================
// Survey History Queries
// ===================

/**
 * Fetch list of all survey runs with optional filtering by survey_id
 */
export const useSurveyRuns = (
  surveyId?: string,
  options?: Omit<UseQueryOptions<SurveyRunMetadata[], Error>, 'queryKey' | 'queryFn'>
) => {
  return useQuery<SurveyRunMetadata[], Error>({
    queryKey: queryKeys.surveyRuns(surveyId),
    queryFn: () => api.getSurveyRuns(surveyId),
    staleTime: 1 * 60 * 1000, // 1 minute
    ...options,
  });
};

/**
 * Fetch detailed survey run information
 */
export const useSurveyRun = (
  runId: string,
  options?: Omit<UseQueryOptions<SurveyRunDetail, Error>, 'queryKey' | 'queryFn'>
) => {
  return useQuery<SurveyRunDetail, Error>({
    queryKey: queryKeys.surveyRun(runId),
    queryFn: () => api.getSurveyRun(runId),
    enabled: !!runId,
    staleTime: 5 * 60 * 1000, // 5 minutes
    ...options,
  });
};

/**
 * Delete a survey run
 */
export const useDeleteSurveyRun = (
  options?: UseMutationOptions<{ run_id: string; status: string }, Error, string>
) => {
  const queryClient = useQueryClient();

  return useMutation<{ run_id: string; status: string }, Error, string>({
    mutationFn: api.deleteSurveyRun,
    onSuccess: () => {
      // Invalidate all survey runs lists
      queryClient.invalidateQueries({ queryKey: ['surveyRuns'] });
    },
    ...options,
  });
};

// ===================
// Ground Truth Queries
// ===================

/**
 * Fetch list of all ground truths with optional filtering by survey_id
 */
export const useGroundTruths = (
  surveyId?: string,
  options?: Omit<UseQueryOptions<GroundTruthMetadata[], Error>, 'queryKey' | 'queryFn'>
) => {
  return useQuery<GroundTruthMetadata[], Error>({
    queryKey: queryKeys.groundTruths(surveyId),
    queryFn: () => api.getGroundTruths(surveyId),
    staleTime: 1 * 60 * 1000, // 1 minute
    ...options,
  });
};

/**
 * Fetch detailed ground truth information
 */
export const useGroundTruth = (
  gtId: string,
  options?: Omit<UseQueryOptions<GroundTruth, Error>, 'queryKey' | 'queryFn'>
) => {
  return useQuery<GroundTruth, Error>({
    queryKey: queryKeys.groundTruth(gtId),
    queryFn: () => api.getGroundTruth(gtId),
    enabled: !!gtId,
    staleTime: 5 * 60 * 1000, // 5 minutes
    ...options,
  });
};

// ===================
// Ground Truth Mutations
// ===================

/**
 * Create ground truth from SSR pipeline
 */
export const useCreateGroundTruthFromSSR = (
  options?: UseMutationOptions<
    { id: string; status: string; name: string; num_profiles: number; num_responses: number },
    Error,
    CreateGroundTruthFromSSRRequest
  >
) => {
  const queryClient = useQueryClient();

  return useMutation<
    { id: string; status: string; name: string; num_profiles: number; num_responses: number },
    Error,
    CreateGroundTruthFromSSRRequest
  >({
    mutationFn: api.createGroundTruthFromSSR,
    onSuccess: () => {
      // Invalidate all ground truths lists
      queryClient.invalidateQueries({ queryKey: ['groundTruths'] });
    },
    ...options,
  });
};

/**
 * Upload ground truth data
 */
export const useUploadGroundTruth = (
  options?: UseMutationOptions<{ id: string; status: string; name: string }, Error, UploadGroundTruthRequest>
) => {
  const queryClient = useQueryClient();

  return useMutation<{ id: string; status: string; name: string }, Error, UploadGroundTruthRequest>({
    mutationFn: api.uploadGroundTruth,
    onSuccess: () => {
      // Invalidate all ground truths lists
      queryClient.invalidateQueries({ queryKey: ['groundTruths'] });
    },
    ...options,
  });
};

/**
 * Delete a ground truth
 */
export const useDeleteGroundTruth = (
  options?: UseMutationOptions<{ id: string; status: string }, Error, string>
) => {
  const queryClient = useQueryClient();

  return useMutation<{ id: string; status: string }, Error, string>({
    mutationFn: api.deleteGroundTruth,
    onSuccess: () => {
      // Invalidate all ground truths lists
      queryClient.invalidateQueries({ queryKey: ['groundTruths'] });
    },
    ...options,
  });
};

/**
 * Compare survey run to ground truth
 */
export const useCompareToGroundTruth = (
  options?: UseMutationOptions<ComparisonResults, Error, { runId: string; groundTruthId: string }>
) => {
  return useMutation<ComparisonResults, Error, { runId: string; groundTruthId: string }>({
    mutationFn: ({ runId, groundTruthId }) => api.compareToGroundTruth(runId, groundTruthId),
    ...options,
  });
};
