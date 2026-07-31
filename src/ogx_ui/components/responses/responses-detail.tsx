"use client";

import { OpenAIResponse, InputItemListResponse } from "@/lib/types";
import { Card, CardContent, CardHeader, CardTitle } from "@/components/ui/card";
import { Skeleton } from "@/components/ui/skeleton";
import {
  DetailLoadingView,
  DetailErrorView,
  DetailNotFoundView,
  DetailLayout,
  PropertiesCard,
  PropertyItem,
} from "@/components/layout/detail-layout";
import { GroupedItemsDisplay } from "./grouping/grouped-items-display";

interface ResponseDetailViewProps {
  response: OpenAIResponse | null;
  inputItems: InputItemListResponse | null;
  isLoading: boolean;
  isLoadingInputItems: boolean;
  error: Error | null;
  inputItemsError: Error | null;
  id: string;
}

export function ResponseDetailView({
  response,
  inputItems,
  isLoading,
  isLoadingInputItems,
  error,
  inputItemsError,
  id,
}: ResponseDetailViewProps) {
  const title = "Responses Details";

  if (error) {
    return <DetailErrorView title={title} id={id} error={error} />;
  }

  if (isLoading) {
    return <DetailLoadingView title={title} />;
  }

  if (!response) {
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
          {/* Show loading state for input items */}
          {isLoadingInputItems ? (
            <div className="space-y-2">
              <Skeleton className="h-4 w-full" />
              <Skeleton className="h-4 w-3/4" />
              <Skeleton className="h-4 w-1/2" />
            </div>
          ) : inputItemsError ? (
            <div className="text-red-500 text-sm">
              Error loading input items: {inputItemsError.message}
              <br />
              <span className="text-gray-500 text-xs">
                Falling back to response input data.
              </span>
            </div>
          ) : null}

          {/* Display input items if available, otherwise fall back to response.input */}
          {(() => {
            const dataToDisplay =
              inputItems?.data && inputItems.data.length > 0
                ? inputItems.data
                : response.input;

            if (dataToDisplay && dataToDisplay.length > 0) {
              return (
                <GroupedItemsDisplay
                  items={dataToDisplay}
                  keyPrefix="input"
                  defaultRole="unknown"
                />
              );
            } else {
              return (
                <p className="text-gray-500 italic text-sm">
                  No input data available.
                </p>
              );
            }
          })()}
        </CardContent>
      </Card>

      <Card>
        <CardHeader>
          <CardTitle>Output</CardTitle>
        </CardHeader>
        <CardContent>
          {response.output?.length > 0 ? (
            <GroupedItemsDisplay
              items={response.output}
              keyPrefix="output"
              defaultRole="assistant"
            />
          ) : (
            <p className="text-gray-500 italic text-sm">
              No output data available.
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
        value={new Date(response.created_at * 1000).toLocaleString()}
      />
      <PropertyItem label="ID" value={response.id} />
      <PropertyItem label="Model" value={response.model} />
      <PropertyItem label="Status" value={response.status} hasBorder />
      {response.temperature && (
        <PropertyItem
          label="Temperature"
          value={response.temperature}
          hasBorder
        />
      )}
      {response.top_p && <PropertyItem label="Top P" value={response.top_p} />}
      {response.parallel_tool_calls && (
        <PropertyItem
          label="Parallel Tool Calls"
          value={response.parallel_tool_calls ? "Yes" : "No"}
        />
      )}
      {response.previous_response_id && (
        <PropertyItem
          label="Previous Response ID"
          value={
            <span className="text-xs">{response.previous_response_id}</span>
          }
          hasBorder
        />
      )}
      {response.error && (
        <PropertyItem
          label="Error"
          value={
            <span className="text-red-900 font-medium">
              {response.error.code}: {response.error.message}
            </span>
          }
          className="pt-1 mt-1 border-t border-red-200"
        />
      )}
    </PropertiesCard>
  );

  return (
    <DetailLayout title={title} mainContent={mainContent} sidebar={sidebar} />
  );
}
