import { buildVectorStoreParams } from "./vector-store-params";

describe("buildVectorStoreParams", () => {
  it("leaves embedding defaults to the server when the form leaves them empty", () => {
    expect(
      buildVectorStoreParams({ name: "Neo4j store", provider_id: "neo4j" })
    ).toEqual({ name: "Neo4j store", provider_id: "neo4j" });
  });

  it("includes explicitly selected embedding settings", () => {
    expect(
      buildVectorStoreParams({
        name: "Custom store",
        provider_id: "neo4j",
        embedding_model: "sentence-transformers/example",
        embedding_dimension: 384,
      })
    ).toEqual({
      name: "Custom store",
      provider_id: "neo4j",
      embedding_model: "sentence-transformers/example",
      embedding_dimension: 384,
    });
  });
});
