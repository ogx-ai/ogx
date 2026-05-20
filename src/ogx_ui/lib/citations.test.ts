import { processCitations, type FileCitation } from "./citations";

describe("processCitations", () => {
  test("replaces citation markers with numbered links", () => {
    const text = "See this source<|doc-uuid-1|> for details.";
    const fileIdMap = { "doc-uuid-1": "file-abc123" };

    const result = processCitations(text, undefined, fileIdMap);

    expect(result.cleaned).toBe(
      "See this source[1](/logs/files/file-abc123) for details."
    );
    expect(result.citations).toHaveLength(1);
    expect(result.citations[0].file_id).toBe("file-abc123");
  });

  test("resolves multiple citations with unique numbering", () => {
    const text = "Source A<|uuid-a|> and Source B<|uuid-b|>.";
    const fileIdMap = {
      "uuid-a": "file-aaa",
      "uuid-b": "file-bbb",
    };

    const result = processCitations(text, undefined, fileIdMap);

    expect(result.cleaned).toBe(
      "Source A[1](/logs/files/file-aaa) and Source B[2](/logs/files/file-bbb)."
    );
    expect(result.citations).toHaveLength(2);
  });

  test("reuses same number for repeated citations", () => {
    const text = "First<|uuid-a|> then again<|uuid-a|>.";
    const fileIdMap = { "uuid-a": "file-aaa" };

    const result = processCitations(text, undefined, fileIdMap);

    expect(result.cleaned).toBe(
      "First[1](/logs/files/file-aaa) then again[1](/logs/files/file-aaa)."
    );
    expect(result.citations).toHaveLength(1);
  });

  test("falls back to raw id when no fileIdMap is provided", () => {
    const text = "Reference<|some-uuid|>.";

    const result = processCitations(text);

    expect(result.cleaned).toBe("Reference[1](/logs/files/some-uuid).");
    expect(result.citations[0].file_id).toBe("some-uuid");
  });

  test("resolves from annotations when fileIdMap is missing", () => {
    const text = "Cite<|doc-id|>.";
    const annotations: FileCitation[] = [
      {
        type: "file_citation",
        file_id: "doc-id",
        filename: "report.pdf",
        index: 0,
      },
    ];

    const result = processCitations(text, annotations);

    expect(result.citations[0].filename).toBe("report.pdf");
    expect(result.citations[0].file_id).toBe("doc-id");
  });

  test("prefers fileIdMap over annotations for file_id resolution", () => {
    const text = "Cite<|doc-id|>.";
    const annotations: FileCitation[] = [
      {
        type: "file_citation",
        file_id: "doc-id",
        filename: "report.pdf",
        index: 0,
      },
    ];
    const fileIdMap = { "doc-id": "file-resolved" };

    const result = processCitations(text, annotations, fileIdMap);

    expect(result.citations[0].file_id).toBe("file-resolved");
    expect(result.citations[0].filename).toBe("report.pdf");
  });

  test("strips raw tool call JSON from text", () => {
    const text =
      'Answer here.{"name": "file_search", "parameters": {"query": "test"}}';

    const result = processCitations(text);

    expect(result.cleaned).toBe("Answer here.");
  });

  test("handles text with no citations", () => {
    const text = "Just a regular message.";

    const result = processCitations(text);

    expect(result.cleaned).toBe("Just a regular message.");
    expect(result.citations).toHaveLength(0);
  });
});
