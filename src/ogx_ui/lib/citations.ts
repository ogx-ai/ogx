export interface FileCitation {
  type: "file_citation";
  file_id: string;
  filename: string;
  index: number;
}

export function processCitations(
  text: string,
  annotations?: FileCitation[],
  fileIdMap?: Record<string, string>
): { cleaned: string; citations: FileCitation[] } {
  const idToIndex = new Map<string, number>();
  const citations: FileCitation[] = [];

  const withNumbers = text.replace(/<\|([^|]+)\|>/g, (_match, id: string) => {
    if (!idToIndex.has(id)) {
      const idx = idToIndex.size;
      idToIndex.set(id, idx);
      const ann = annotations?.find(a => a.file_id === id);
      const resolvedFileId = fileIdMap?.[id] || ann?.file_id || id;
      citations.push({
        type: "file_citation",
        file_id: resolvedFileId,
        filename: ann?.filename || id,
        index: idx,
      });
    }
    const num = idToIndex.get(id)! + 1;
    return `[${num}](/logs/files/${citations[idToIndex.get(id)!].file_id})`;
  });

  let cleaned = withNumbers;
  cleaned = cleaned.replace(
    /\{"name":\s*"[^"]+",\s*"parameters":\s*\{[^}]*\}\}/g,
    ""
  );
  return { cleaned: cleaned.trim(), citations };
}
