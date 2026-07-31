// check if content contains function call JSON
export const containsToolCall = (content: string): boolean => {
  return (
    content.includes('"type": "function"') ||
    content.includes('"name": "file_search"') ||
    content.includes('"parameters":') ||
    !!content.match(/\{"type":\s*"function".*?\}/)
  );
};

export const extractCleanText = (content: string): string | null => {
  if (containsToolCall(content)) {
    try {
      // parse and extract non-function call parts
      const jsonMatch = content.match(/\{"type":\s*"function"[^}]*\}[^}]*\}/);
      if (jsonMatch) {
        const jsonPart = jsonMatch[0];
        const parsedJson = JSON.parse(jsonPart);

        // if function call, extract text after JSON
        if (parsedJson.type === "function") {
          const textAfterJson = content
            .substring(content.indexOf(jsonPart) + jsonPart.length)
            .trim();
          return textAfterJson || null;
        }
      }
      return null;
    } catch {
      return null;
    }
  }
  return content;
};

// removes function call JSON handling different content types
export const cleanMessageContent = (
  content: string | unknown[] | unknown
): string => {
  if (typeof content === "string") {
    const cleaned = extractCleanText(content);
    return cleaned || "";
  } else if (Array.isArray(content)) {
    return content
      .filter((item: { type: string }) => item.type === "text")
      .map((item: { text: string }) => item.text)
      .join("");
  } else {
    return JSON.stringify(content);
  }
};
