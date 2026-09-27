import type { Model } from "ogx-client/resources/models";

export function normalizeModelList(result: unknown): Model[] {
  if (Array.isArray(result)) {
    return result as Model[];
  }

  if (result && typeof result === "object" && "data" in result) {
    const data = (result as { data?: unknown }).data;
    return Array.isArray(data) ? (data as Model[]) : [];
  }

  return [];
}
