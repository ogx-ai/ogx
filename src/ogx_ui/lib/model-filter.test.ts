import type { Model } from "ogx-client/resources/models";
import { filterModels, parseModelAllowlist } from "./model-filter";

const models = [
  { id: "model-a" },
  { id: "model-b" },
  { id: "model-c" },
] as Model[];

describe("model filtering", () => {
  test("keeps the API model list unchanged when no allowlist is configured", () => {
    expect(parseModelAllowlist(undefined)).toEqual([]);
    expect(filterModels(models, [])).toEqual(models);
  });

  test("filters models in allowlist order", () => {
    const allowed = parseModelAllowlist(" model-c, model-a ");

    expect(filterModels(models, allowed).map(model => model.id)).toEqual([
      "model-c",
      "model-a",
    ]);
  });

  test("omits unknown model IDs", () => {
    const allowed = parseModelAllowlist("missing, model-b");

    expect(filterModels(models, allowed).map(model => model.id)).toEqual([
      "model-b",
    ]);
  });
});
