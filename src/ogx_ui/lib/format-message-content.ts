import { ChatMessage, ChatMessageContentPart } from "@/lib/types";
import { formatToolCallToString } from "@/lib/format-tool-call";

export function extractTextFromContentPart(
  content: string | ChatMessageContentPart[] | null | undefined
): string {
  if (content === null || content === undefined) {
    return "";
  }
  if (typeof content === "string") {
    return content;
  } else if (Array.isArray(content)) {
    const parts: string[] = [];
    for (const item of content) {
      if (
        item &&
        typeof item === "object" &&
        item.type === "text" &&
        typeof item.text === "string"
      ) {
        parts.push(item.text);
      } else if (
        item &&
        typeof item === "object" &&
        item.type === "image_url"
      ) {
        parts.push("[Image]"); // Placeholder for images
      } else if (typeof item === "string") {
        // Handle cases where an array might contain plain strings
        parts.push(item);
      }
    }
    return parts.join(" ");
  } else {
    return content;
  }
}

export function extractDisplayableText(
  message: ChatMessage | undefined | null
): string {
  if (!message) {
    return "";
  }

  const textPart = extractTextFromContentPart(message.content);
  let toolCallPart = "";

  if (
    message.tool_calls &&
    Array.isArray(message.tool_calls) &&
    message.tool_calls.length > 0
  ) {
    // For summary, usually the first tool call is sufficient
    toolCallPart = formatToolCallToString(message.tool_calls[0]);
  }

  if (textPart && toolCallPart) {
    return `${textPart} ${toolCallPart}`;
  } else if (toolCallPart) {
    return toolCallPart;
  } else {
    return textPart; // textPart will be "" if message.content was initially null/undefined/empty array etc.
  }
}
