import type { Model } from "ogx-client/resources/models";

export function parseModelAllowlist(value: string | undefined): string[] {
  return (value ?? "")
    .split(",")
    .map(modelId => modelId.trim())
    .filter(Boolean);
}

export function filterModels(
  models: Model[],
  allowedModelIds: string[]
): Model[] {
  if (allowedModelIds.length === 0) {
    return models;
  }

  const modelsById = new Map(models.map(model => [model.id, model]));
  const seenModelIds = new Set<string>();

  return allowedModelIds.flatMap(modelId => {
    const model = modelsById.get(modelId);
    if (!model || seenModelIds.has(modelId)) {
      return [];
    }

    seenModelIds.add(modelId);
    return [model];
  });
}
