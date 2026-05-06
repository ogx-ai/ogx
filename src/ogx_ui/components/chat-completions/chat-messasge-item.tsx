"use client";

import { ChatMessage } from "@/lib/types";
import React from "react";
import { formatToolCallToString } from "@/lib/format-tool-call";
import { extractTextFromContentPart } from "@/lib/format-message-content";
import {
  MessageBlock,
  ToolCallBlock,
} from "@/components/chat-playground/message-components";

interface ChatMessageItemProps {
  message: ChatMessage;
}
export function ChatMessageItem({ message }: ChatMessageItemProps) {
  switch (message.role) {
    case "system":
      return (
        <MessageBlock
          label="System"
          content={extractTextFromContentPart(message.content)}
        />
      );
    case "user":
      return (
        <MessageBlock
          label="User"
          content={extractTextFromContentPart(message.content)}
        />
      );

    case "assistant":
      if (
        message.tool_calls &&
        Array.isArray(message.tool_calls) &&
        message.tool_calls.length > 0
      ) {
        return (
          <>
            {message.tool_calls.map(
              (
                toolCall: { function?: { name?: string; arguments?: unknown } },
                index: number
              ) => {
                const formattedToolCall = formatToolCallToString(toolCall);
                const toolCallContent = (
                  <ToolCallBlock>
                    {formattedToolCall || "Error: Could not display tool call"}
                  </ToolCallBlock>
                );
                return (
                  <MessageBlock
                    key={index}
                    label="Tool Call"
                    content={toolCallContent}
                  />
                );
              }
            )}
          </>
        );
      } else {
        return (
          <MessageBlock
            label="Assistant"
            content={extractTextFromContentPart(message.content)}
          />
        );
      }
    case "tool":
      const toolOutputContent = (
        <ToolCallBlock>
          {extractTextFromContentPart(message.content)}
        </ToolCallBlock>
      );
      return (
        <MessageBlock label="Tool Call Output" content={toolOutputContent} />
      );
  }
  return null;
}
