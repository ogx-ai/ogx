import type { VectorStoreFile } from "ogx-client/resources/vector-stores/files";
import type { File } from "ogx-client/resources/files";
import { getAvailableFiles } from "./available-files";

const uploadedFile = (id: string): File =>
  ({ id, filename: `${id}.txt` }) as File;

const attachedFile = (id: string): VectorStoreFile => ({
  id,
  created_at: 0,
  status: "completed",
  usage_bytes: 0,
  vector_store_id: "vs_test",
  chunking_strategy: { type: "auto" },
});

describe("getAvailableFiles", () => {
  test("filters out files already attached to the vector store", () => {
    expect(
      getAvailableFiles(
        [uploadedFile("file_1"), uploadedFile("file_2")],
        [attachedFile("file_1")]
      ).map(file => file.id)
    ).toEqual(["file_2"]);
  });

  test("returns no files when every uploaded file is attached", () => {
    expect(
      getAvailableFiles([uploadedFile("file_1")], [attachedFile("file_1")])
    ).toEqual([]);
  });
});
