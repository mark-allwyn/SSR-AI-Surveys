/**
 * TypeScript interfaces for SSR Pipeline Demo Application
 * These types match the backend API schemas
 */

// Survey Types
export interface Survey {
  name: string;
  description: string;
  context: string;
  questions: Question[];
  persona_groups: PersonaGroup[];
  categories?: Category[];
  demographics: string[];
  sample_size: number;
}

export interface SurveyListItem {
  id: string;
  name: string;
  description: string;
  num_questions: number;
  num_persona_groups: number;
  has_categories: boolean;
}

export interface Question {
  id: string;
  text: string;
  type: QuestionType;
  scale?: { [key: number]: string };
  options?: string[];
  category?: string;
  categories_compared?: string[];
}

export type QuestionType = 'yes_no' | 'likert_5' | 'likert_7' | 'multiple_choice' | 'preference_scale';

export interface PersonaGroup {
  name: string;
  description: string;
  personas: string[];
  target_demographics: {
    gender: string[];
    age_group: string[];
    occupation: string[];
  };
  weight: number;
}

export interface Category {
  id: string;
  name: string;
  description: string;
  context: string;
}

// Response Types
export interface RespondentProfile {
  respondent_id: string;
  description: string;
  gender: string;
  age_group: string;
  persona_group: string;
  occupation: string;
}

export interface Response {
  respondent_id: string;
  question_id: string;
  text_response: string;
  respondent_profile: RespondentProfile;
  category?: string;
}

// SSR Types
export interface RatingDistribution {
  probabilities: number[];
  mode: number;
  expected_value: number;
  entropy: number;
  text_response: string;
  gender: string;
  age_group: string;
  persona_group: string;
  occupation: string;
}

export interface DistributionData {
  [category: string]: {
    [question_id: string]: {
      [respondent_id: string]: RatingDistribution;
    };
  };
}

// API Request Types
export interface CreateSurveyRequest {
  yaml_content: string;
  filename: string;
}

export interface GenerateProfilesRequest {
  survey_id: string;
  num_profiles: number;
}

export interface GenerateResponsesRequest {
  survey_id: string;
  profiles: RespondentProfile[];
  llm_provider: 'openai' | 'anthropic';
  model: string;
  temperature: number;
}

export interface ApplySSRRequest {
  survey_id: string;
  responses: Response[];
  temperature: number;
  normalize_method: 'paper' | 'softmax' | 'linear';
}

export interface RunSurveyRequest {
  survey_id: string;
  num_profiles: number;
  llm_provider: 'openai' | 'anthropic';
  model: string;
  llm_temperature: number;
  ssr_temperature: number;
  normalize_method: 'paper' | 'softmax' | 'linear';
  seed: number;
}

// API Response Types
export interface GenerateProfilesResponse {
  survey_id: string;
  num_profiles: number;
  profiles: RespondentProfile[];
}

export interface GenerateResponsesResponse {
  survey_id: string;
  num_responses: number;
  responses: Response[];
}

export interface ApplySSRResponse {
  survey_id: string;
  num_distributions: number;
  distributions: DistributionData;
}

export interface RunSurveyResponse {
  run_id: string;
  survey_id: string;
  num_profiles: number;
  num_responses: number;
  num_distributions: number;
  distributions: DistributionData;
  config: {
    llm_provider: string;
    model: string;
    llm_temperature: number;
    ssr_temperature: number;
    normalize_method: string;
    seed: number;
  };
}

// UI State Types
export interface SurveyBuilderState {
  name: string;
  description: string;
  context: string;
  questions: Question[];
  persona_groups: PersonaGroup[];
  categories: Category[];
  demographics: string[];
  sample_size: number;
}

export interface RunSurveyConfig {
  survey_id: string;
  num_profiles: number;
  llm_provider: 'openai' | 'anthropic';
  model: string;
  llm_temperature: number;
  ssr_temperature: number;
  normalize_method: 'paper' | 'softmax' | 'linear';
  seed: number;
}

export interface RunStatus {
  status: 'idle' | 'running' | 'complete' | 'error';
  progress: number;
  message?: string;
  error?: string;
}

// Constants
export const QUESTION_TYPES: { value: QuestionType; label: string }[] = [
  { value: 'yes_no', label: 'Yes/No' },
  { value: 'likert_5', label: 'Likert 5-Point Scale' },
  { value: 'likert_7', label: 'Likert 7-Point Scale' },
  { value: 'multiple_choice', label: 'Multiple Choice' },
  { value: 'preference_scale', label: 'Preference Scale (Comparative)' },
];

export const LLM_PROVIDERS: { value: 'openai' | 'anthropic'; label: string }[] = [
  { value: 'openai', label: 'OpenAI' },
  { value: 'anthropic', label: 'Anthropic (Claude)' },
];

export const OPENAI_MODELS = [
  { value: 'gpt-3.5-turbo', label: 'GPT-3.5 Turbo (Fast & Affordable)' },
  { value: 'gpt-4-turbo-preview', label: 'GPT-4 Turbo (Balanced)' },
  { value: 'gpt-4', label: 'GPT-4 (Most Capable)' },
];

export const ANTHROPIC_MODELS = [
  { value: 'claude-3-haiku-20240307', label: 'Claude 3 Haiku (Fast & Affordable)' },
  { value: 'claude-3-sonnet-20240229', label: 'Claude 3 Sonnet (Balanced)' },
  { value: 'claude-3-opus-20240229', label: 'Claude 3 Opus (Most Capable)' },
];

export const NORMALIZE_METHODS: { value: 'paper' | 'softmax' | 'linear'; label: string; description: string }[] = [
  { value: 'paper', label: 'Paper Method', description: 'Normalization method from original SSR paper' },
  { value: 'softmax', label: 'Softmax', description: 'Standard softmax normalization' },
  { value: 'linear', label: 'Linear', description: 'Simple linear normalization' },
];

export const AGE_GROUPS = ['18-24', '25-34', '35-44', '45-54', '55-64', '65+'];

export const GENDERS = ['Male', 'Female', 'Non-binary', 'Prefer not to say', 'Other'];

export const OCCUPATIONS = [
  'Student',
  'Professional',
  'Manager',
  'Technical',
  'Service',
  'Sales',
  'Administrative',
  'Skilled Trade',
  'Retired',
  'Self-employed',
  'Other',
];

export const DEMOGRAPHICS_OPTIONS = ['age_group', 'gender', 'occupation', 'persona_group'];
