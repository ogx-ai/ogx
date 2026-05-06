"use client";

import { useEffect, useState } from "react";
import { useParams } from "next/navigation";
import type { ResponseObject } from "llama-stack-client/resources/responses/responses";
import { OpenAIResponse, InputItemListResponse } from "@/lib/types";
import { ResponseDetailView } from "@/components/responses/responses-detail";
import { useAuthClient } from "@/hooks/use-auth-client";

export default function ResponseDetailPage() {
  const params = useParams();
  const id = params.id as string;
  const client = useAuthClient();

  const [responseDetail, setResponseDetail] = useState<OpenAIResponse | null>(
    null
  );
  const [inputItems, setInputItems] = useState<InputItemListResponse | null>(
    null
  );
  const [isLoading, setIsLoading] = useState<boolean>(true);
  const [isLoadingInputItems, setIsLoadingInputItems] = useState<boolean>(true);
  const [error, setError] = useState<Error | null>(null);
  const [inputItemsError, setInputItemsError] = useState<Error | null>(null);

  // Helper function to convert ResponseObject to OpenAIResponse
  const convertResponseObject = (
    responseData: ResponseObject
  ): OpenAIResponse => {
    return {
      id: responseData.id,
      created_at: responseData.created_at,
      model: responseData.model,
      object: responseData.object,
      status: responseData.status,
      output: responseData.output as OpenAIResponse["output"],
      input: [], // ResponseObject doesn't include input; component uses inputItems prop instead
      error: responseData.error,
      parallel_tool_calls: responseData.parallel_tool_calls,
      previous_response_id: responseData.previous_response_id,
      temperature: responseData.temperature,
      top_p: responseData.top_p,
      truncation: responseData.truncation,
    };
  };

  useEffect(() => {
    if (!id) {
      setError(new Error("Response ID is missing."));
      setIsLoading(false);
      return;
    }

    const fetchResponseDetail = async () => {
      setIsLoading(true);
      setIsLoadingInputItems(true);
      setError(null);
      setInputItemsError(null);
      setResponseDetail(null);
      setInputItems(null);

      try {
        const [responseResult, inputItemsResult] = await Promise.allSettled([
          client.responses.retrieve(id),
          client.responses.inputItems.list(id, { order: "asc" }),
        ]);

        // Handle response detail result
        if (responseResult.status === "fulfilled") {
          const convertedResponse = convertResponseObject(responseResult.value);
          setResponseDetail(convertedResponse);
        } else {
          console.error(
            `Error fetching response detail for ID ${id}:`,
            responseResult.reason
          );
          setError(
            responseResult.reason instanceof Error
              ? responseResult.reason
              : new Error("Failed to fetch response detail")
          );
        }

        // Handle input items result
        if (inputItemsResult.status === "fulfilled") {
          const inputItemsData =
            inputItemsResult.value as unknown as InputItemListResponse;
          setInputItems(inputItemsData);
        } else {
          console.error(
            `Error fetching input items for response ID ${id}:`,
            inputItemsResult.reason
          );
          setInputItemsError(
            inputItemsResult.reason instanceof Error
              ? inputItemsResult.reason
              : new Error("Failed to fetch input items")
          );
        }
      } catch (err) {
        console.error(`Unexpected error fetching data for ID ${id}:`, err);
        setError(
          err instanceof Error ? err : new Error("Unexpected error occurred")
        );
      } finally {
        setIsLoading(false);
        setIsLoadingInputItems(false);
      }
    };

    fetchResponseDetail();
  }, [id, client]);

  return (
    <ResponseDetailView
      response={responseDetail}
      inputItems={inputItems}
      isLoading={isLoading}
      isLoadingInputItems={isLoadingInputItems}
      error={error}
      inputItemsError={inputItemsError}
      id={id}
    />
  );
}
