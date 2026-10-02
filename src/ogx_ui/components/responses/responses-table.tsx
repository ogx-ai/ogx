"use client";

import {
  OpenAIResponse,
  ResponseInputMessageContent,
  UsePaginationOptions,
} from "@/lib/types";
import { LogsTable, LogTableRow } from "@/components/logs/logs-table";
import { usePagination } from "@/hooks/use-pagination";
import type { ResponseListResponse } from "ogx-client/resources/responses/responses";
import {
  isMessageInput,
  isMessageItem,
  isFunctionCallItem,
  isWebSearchCallItem,
  FunctionCallItem,
  WebSearchCallItem,
} from "./utils/item-types";

interface ResponsesTableProps {
  /** Optional pagination configuration */
  paginationOptions?: UsePaginationOptions;
}

/**
 * Helper function to convert ResponseListResponse to OpenAIResponse
 */
const convertResponseListData = (
  responseData: ResponseListResponse
): OpenAIResponse => {
  return {
    id: responseData.id,
    created_at: responseData.created_at,
    model: responseData.model,
    object: responseData.object ?? "response",
    status: responseData.status,
    output: responseData.output as OpenAIResponse["output"],
    input: responseData.input as OpenAIResponse["input"],
    error: responseData.error ?? undefined,
    parallel_tool_calls: responseData.parallel_tool_calls,
    previous_response_id: responseData.previous_response_id ?? undefined,
    temperature: responseData.temperature,
    top_p: responseData.top_p,
    truncation: responseData.truncation ?? undefined,
  };
};

function getInputText(response: OpenAIResponse): string {
  const firstInput = response.input.find(isMessageInput);
  if (firstInput) {
    return extractContentFromItem(firstInput);
  }
  return "";
}

function getOutputText(response: OpenAIResponse): string {
  const firstMessage = response.output.find(isMessageItem);
  if (firstMessage) {
    const content = extractContentFromItem(firstMessage);
    if (content) {
      return content;
    }
  }

  const functionCall = response.output.find(isFunctionCallItem);
  if (functionCall) {
    return formatFunctionCall(functionCall);
  }

  const webSearchCall = response.output.find(isWebSearchCallItem);
  if (webSearchCall) {
    return formatWebSearchCall(webSearchCall);
  }

  return JSON.stringify(response.output);
}

function extractContentFromItem(item: {
  content?: string | ResponseInputMessageContent[];
}): string {
  if (!item.content) {
    return "";
  }

  if (typeof item.content === "string") {
    return item.content;
  } else if (Array.isArray(item.content)) {
    const textContent = item.content.find(
      (c: ResponseInputMessageContent) =>
        c.type === "input_text" || c.type === "output_text"
    );
    return textContent?.text || "";
  }
  return "";
}

function formatFunctionCall(functionCall: FunctionCallItem): string {
  const args = functionCall.arguments || "{}";
  const name = functionCall.name || "unknown";
  return `${name}(${args})`;
}

function formatWebSearchCall(webSearchCall: WebSearchCallItem): string {
  return `web_search_call(status: ${webSearchCall.status})`;
}

function formatResponseToRow(response: OpenAIResponse): LogTableRow {
  return {
    id: response.id,
    input: getInputText(response),
    output: getOutputText(response),
    model: response.model,
    createdTime: new Date(response.created_at * 1000).toLocaleString(),
    detailPath: `/logs/responses/${response.id}`,
  };
}

export function ResponsesTable({ paginationOptions }: ResponsesTableProps) {
  const fetchFunction = async (
    client: ReturnType<typeof import("@/hooks/use-auth-client").useAuthClient>,
    params: {
      after?: string;
      limit: number;
      model?: string;
      order?: string;
    }
  ) => {
    const response = await client.responses.list({
      after: params.after,
      limit: params.limit,
      ...(params.model && { model: params.model }),
      ...(params.order && { order: params.order }),
    } as Parameters<typeof client.responses.list>[0]);

    return {
      data: response.data.map(convertResponseListData),
      has_more: response.has_more,
      last_id: response.last_id,
    };
  };

  const { data, status, hasMore, error, loadMore } = usePagination({
    ...paginationOptions,
    fetchFunction,
    errorMessagePrefix: "responses",
  });

  const formattedData = data.map(formatResponseToRow);

  return (
    <LogsTable
      data={formattedData}
      status={status}
      hasMore={hasMore}
      error={error}
      onLoadMore={loadMore}
      caption="A list of your recent responses."
      emptyMessage="No responses found."
    />
  );
}
