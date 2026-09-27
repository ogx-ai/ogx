import type { File } from "ogx-client/resources/files";
import type { VectorStoreFile } from "ogx-client/resources/vector-stores/files";

export function getAvailableFiles(
  uploadedFiles: File[],
  attachedFiles: VectorStoreFile[]
): File[] {
  const attachedFileIds = new Set(attachedFiles.map(file => file.id));
  return uploadedFiles.filter(file => !attachedFileIds.has(file.id));
}
