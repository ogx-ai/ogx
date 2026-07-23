import type { Model } from "ogx-client/resources/models";
import { normalizeModelList } from "./model-list";

const embeddingModel = { id: "embedding-model" } as Model;

describe("normalizeModelList", () => {
  test("unwraps the OpenAI-style data response", () => {
    expect(normalizeModelList({ data: [embeddingModel] })).toEqual([
      embeddingModel,
    ]);
  });

  test("preserves array responses", () => {
    expect(normalizeModelList([embeddingModel])).toEqual([embeddingModel]);
  });

  test("returns an empty list for an invalid response", () => {
    expect(normalizeModelList({ models: [embeddingModel] })).toEqual([]);
  });
});
