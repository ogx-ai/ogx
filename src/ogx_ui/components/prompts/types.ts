export interface Prompt {
  prompt_id: string;
  prompt: string | null;
  version: number;
  variables: string[];
  is_default: boolean;
}

export interface PromptFormData {
  prompt: string;
  variables: string[];
}

export interface PromptFilters {
  searchTerm?: string;
}
