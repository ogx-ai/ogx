import type { VectorStoreFormData } from "./vector-store-editor";

export function buildVectorStoreParams(
  formData: VectorStoreFormData
): Record<string, unknown> {
  const params: Record<string, unknown> = {
    name: formData.name || undefined,
  };

  if (formData.provider_id) params.provider_id = formData.provider_id;
  if (formData.embedding_model)
    params.embedding_model = formData.embedding_model;
  if (formData.embedding_dimension) {
    params.embedding_dimension = formData.embedding_dimension;
  }

  return params;
}
