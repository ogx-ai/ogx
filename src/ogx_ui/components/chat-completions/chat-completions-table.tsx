"use client";

import {
  ChatCompletion,
  UsePaginationOptions,
  ListChatCompletionsResponse,
} from "@/lib/types";
import { ListChatCompletionsParams } from "@/lib/llama-stack-client";
import { LogsTable, LogTableRow } from "@/components/logs/logs-table";
import {
  extractTextFromContentPart,
  extractDisplayableText,
} from "@/lib/format-message-content";
import { usePagination } from "@/hooks/use-pagination";

interface ChatCompletionsTableProps {
  /** Optional pagination configuration */
  paginationOptions?: UsePaginationOptions;
}

function formatChatCompletionToRow(completion: ChatCompletion): LogTableRow {
  return {
    id: completion.id,
    input: extractTextFromContentPart(completion.input_messages?.[0]?.content),
    output: extractDisplayableText(completion.choices?.[0]?.message),
    model: completion.model,
    createdTime: new Date(completion.created * 1000).toLocaleString(),
    detailPath: `/logs/chat-completions/${completion.id}`,
  };
}

export function ChatCompletionsTable({
  paginationOptions,
}: ChatCompletionsTableProps) {
  const fetchFunction = async (
    client: ReturnType<typeof import("@/hooks/use-auth-client").useAuthClient>,
    params: {
      after?: string;
      limit: number;
      model?: string;
      order?: string;
    }
  ) => {
    const response = await client.chat.completions.list({
      after: params.after,
      limit: params.limit,
      ...(params.model && { model: params.model }),
      ...(params.order && { order: params.order }),
    } as ListChatCompletionsParams);

    return response as ListChatCompletionsResponse;
  };

  const { data, status, hasMore, error, loadMore } = usePagination({
    ...paginationOptions,
    fetchFunction,
    errorMessagePrefix: "chat completions",
  });

  const formattedData = data.map(formatChatCompletionToRow);

  return (
    <LogsTable
      data={formattedData}
      status={status}
      hasMore={hasMore}
      error={error}
      onLoadMore={loadMore}
      caption="A list of your recent chat completions."
      emptyMessage="No chat completions found."
    />
  );
}
