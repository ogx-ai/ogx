"use client";

import { ChatMessage, ChatCompletion } from "@/lib/types";
import { ChatMessageItem } from "@/components/chat-completions/chat-messasge-item";
import { Card, CardContent, CardHeader, CardTitle } from "@/components/ui/card";
import {
  DetailLoadingView,
  DetailErrorView,
  DetailNotFoundView,
  DetailLayout,
  PropertiesCard,
  PropertyItem,
} from "@/components/layout/detail-layout";

interface ChatCompletionDetailViewProps {
  completion: ChatCompletion | null;
  isLoading: boolean;
  error: Error | null;
  id: string;
}

export function ChatCompletionDetailView({
  completion,
  isLoading,
  error,
  id,
}: ChatCompletionDetailViewProps) {
  const title = "Chat Completion Details";

  if (error) {
    return <DetailErrorView title={title} id={id} error={error} />;
  }

  if (isLoading) {
    return <DetailLoadingView title={title} />;
  }

  if (!completion) {
    return <DetailNotFoundView title={title} id={id} />;
  }

  // Main content cards
  const mainContent = (
    <>
      <Card>
        <CardHeader>
          <CardTitle>Input</CardTitle>
        </CardHeader>
        <CardContent>
          {completion.input_messages?.map((msg, index) => (
            <ChatMessageItem key={`input-msg-${index}`} message={msg} />
          ))}
          {completion.choices?.[0]?.message?.tool_calls &&
          Array.isArray(completion.choices[0].message.tool_calls) &&
          !completion.input_messages?.some(
            im =>
              im.role === "assistant" &&
              im.tool_calls &&
              Array.isArray(im.tool_calls) &&
              im.tool_calls.length > 0
          )
            ? completion.choices[0].message.tool_calls.map(
                (toolCall: { function?: { name?: string } }, index: number) => {
                  const assistantToolCallMessage: ChatMessage = {
                    role: "assistant",
                    tool_calls: [toolCall],
                    content: "", // Ensure content is defined, even if empty
                  };
                  return (
                    <ChatMessageItem
                      key={`choice-tool-call-${index}`}
                      message={assistantToolCallMessage}
                    />
                  );
                }
              )
            : null}
        </CardContent>
      </Card>

      <Card>
        <CardHeader>
          <CardTitle>Output</CardTitle>
        </CardHeader>
        <CardContent>
          {completion.choices?.[0]?.message ? (
            <ChatMessageItem
              message={completion.choices[0].message as ChatMessage}
            />
          ) : (
            <p className="text-gray-500 italic text-sm">
              No message found in assistant&apos;s choice.
            </p>
          )}
        </CardContent>
      </Card>
    </>
  );

  // Properties sidebar
  const sidebar = (
    <PropertiesCard>
      <PropertyItem
        label="Created"
        value={new Date(completion.created * 1000).toLocaleString()}
      />
      <PropertyItem label="ID" value={completion.id} />
      <PropertyItem label="Model" value={completion.model} />
      <PropertyItem
        label="Finish Reason"
        value={completion.choices?.[0]?.finish_reason || "N/A"}
        hasBorder
      />
      {(() => {
        const toolCalls = completion.choices?.[0]?.message?.tool_calls;
        if (toolCalls && Array.isArray(toolCalls) && toolCalls.length > 0) {
          return (
            <PropertyItem
              label="Functions/Tools Called"
              value={
                <div>
                  <ul className="list-disc list-inside pl-4 mt-1">
                    {toolCalls.map(
                      (
                        toolCall: { function?: { name?: string } },
                        index: number
                      ) => (
                        <li key={index}>
                          <span className="text-gray-900 font-medium">
                            {toolCall.function?.name || "N/A"}
                          </span>
                        </li>
                      )
                    )}
                  </ul>
                </div>
              }
              hasBorder
            />
          );
        }
        return null;
      })()}
    </PropertiesCard>
  );

  return (
    <DetailLayout title={title} mainContent={mainContent} sidebar={sidebar} />
  );
}
