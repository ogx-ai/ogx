"use client";

import React, { useState, useCallback, useEffect, useRef } from "react";
import { useAuthClient } from "@/hooks/use-auth-client";
import { useSession } from "next-auth/react";
import { Badge } from "@/components/ui/badge";
import { Skeleton } from "@/components/ui/skeleton";
import {
  Table,
  TableBody,
  TableCell,
  TableHead,
  TableHeader,
  TableRow,
} from "@/components/ui/table";
import { Card, CardContent, CardHeader, CardTitle } from "@/components/ui/card";
import { ChevronDown, ChevronRight, AlertCircle } from "lucide-react";

interface BatchRequestCounts {
  total: number;
  completed: number;
  failed: number;
}

interface Batch {
  id: string;
  object: string;
  endpoint: string;
  status: string;
  created_at: number;
  completed_at: number | null;
  cancelled_at: number | null;
  expired_at: number | null;
  failed_at: number | null;
  input_file_id: string;
  output_file_id: string | null;
  error_file_id: string | null;
  request_counts: BatchRequestCounts | null;
  errors: { object: string; data: unknown[] } | null;
}

type FetchStatus = "loading" | "idle" | "error";

function formatTimestamp(timestamp: number): string {
  return new Date(timestamp * 1000).toLocaleString();
}

function getStatusBadge(status: string) {
  const statusConfig: Record<string, { className: string; label: string }> = {
    completed: {
      className:
        "bg-green-100 text-green-800 dark:bg-green-900 dark:text-green-200 border-green-200 dark:border-green-800",
      label: "Completed",
    },
    in_progress: {
      className:
        "bg-blue-100 text-blue-800 dark:bg-blue-900 dark:text-blue-200 border-blue-200 dark:border-blue-800",
      label: "In Progress",
    },
    failed: {
      className:
        "bg-red-100 text-red-800 dark:bg-red-900 dark:text-red-200 border-red-200 dark:border-red-800",
      label: "Failed",
    },
    cancelled: {
      className:
        "bg-gray-100 text-gray-800 dark:bg-gray-700 dark:text-gray-200 border-gray-200 dark:border-gray-600",
      label: "Cancelled",
    },
    validating: {
      className:
        "bg-yellow-100 text-yellow-800 dark:bg-yellow-900 dark:text-yellow-200 border-yellow-200 dark:border-yellow-800",
      label: "Validating",
    },
    cancelling: {
      className:
        "bg-gray-100 text-gray-800 dark:bg-gray-700 dark:text-gray-200 border-gray-200 dark:border-gray-600",
      label: "Cancelling",
    },
    expired: {
      className:
        "bg-gray-100 text-gray-800 dark:bg-gray-700 dark:text-gray-200 border-gray-200 dark:border-gray-600",
      label: "Expired",
    },
    finalizing: {
      className:
        "bg-blue-100 text-blue-800 dark:bg-blue-900 dark:text-blue-200 border-blue-200 dark:border-blue-800",
      label: "Finalizing",
    },
  };

  const config = statusConfig[status] || {
    className:
      "bg-gray-100 text-gray-800 dark:bg-gray-700 dark:text-gray-200 border-gray-200 dark:border-gray-600",
    label: status,
  };

  return (
    <Badge variant="outline" className={config.className}>
      {config.label}
    </Badge>
  );
}

export function BatchesManagement() {
  const client = useAuthClient();
  const { status: sessionStatus } = useSession();
  const [batches, setBatches] = useState<Batch[]>([]);
  const [status, setStatus] = useState<FetchStatus>("loading");
  const [error, setError] = useState<string | null>(null);
  const [expandedBatchId, setExpandedBatchId] = useState<string | null>(null);
  const hasFetched = useRef(false);

  const fetchBatches = useCallback(async () => {
    setStatus("loading");
    setError(null);

    try {
      const response = await client.batches.list();
      const data = response.data ?? (response as unknown as Batch[]);
      setBatches(Array.isArray(data) ? data : []);
      setStatus("idle");
    } catch (err: unknown) {
      const errorMessage = err instanceof Error ? err.message : "Unknown error";

      if (
        errorMessage.includes("501") ||
        errorMessage.includes("not implemented") ||
        errorMessage.includes("Not Implemented")
      ) {
        setError(
          "Batches API is not available on this server. The provider may not support batch operations."
        );
      } else if (errorMessage.includes("404")) {
        setError(
          "Batches API endpoint not found. This server may not support batch operations."
        );
      } else {
        setError(`Failed to load batches: ${errorMessage}`);
      }
      setStatus("error");
    }
  }, [client]);

  useEffect(() => {
    if (sessionStatus === "loading") return;
    if (hasFetched.current) return;
    hasFetched.current = true;
    fetchBatches();
  }, [fetchBatches, sessionStatus]);

  const toggleExpand = (batchId: string) => {
    setExpandedBatchId(prev => (prev === batchId ? null : batchId));
  };

  const renderContent = () => {
    if (status === "loading") {
      return (
        <div className="space-y-2">
          <Skeleton className="h-8 w-full" />
          <Skeleton className="h-4 w-full" />
          <Skeleton className="h-4 w-full" />
          <Skeleton className="h-4 w-full" />
          <Skeleton className="h-4 w-full" />
        </div>
      );
    }

    if (status === "error") {
      return (
        <Card>
          <CardContent className="flex items-center gap-3 py-8">
            <AlertCircle className="h-5 w-5 text-muted-foreground" />
            <p className="text-muted-foreground">{error}</p>
          </CardContent>
        </Card>
      );
    }

    if (batches.length === 0) {
      return (
        <div className="text-center py-8">
          <p className="text-muted-foreground">
            No batches found. Batch jobs will appear here once created via the
            API.
          </p>
        </div>
      );
    }

    return (
      <div className="overflow-auto flex-1 min-h-0">
        <Table>
          <TableHeader>
            <TableRow>
              <TableHead className="w-8"></TableHead>
              <TableHead>Batch ID</TableHead>
              <TableHead>Status</TableHead>
              <TableHead>Endpoint</TableHead>
              <TableHead>Created At</TableHead>
              <TableHead>Completed At</TableHead>
              <TableHead>Input File ID</TableHead>
              <TableHead>Output File ID</TableHead>
            </TableRow>
          </TableHeader>
          <TableBody>
            {batches.map(batch => {
              const isExpanded = expandedBatchId === batch.id;

              return (
                <React.Fragment key={batch.id}>
                  <TableRow
                    onClick={() => toggleExpand(batch.id)}
                    className="cursor-pointer hover:bg-muted/50"
                  >
                    <TableCell>
                      {isExpanded ? (
                        <ChevronDown className="h-4 w-4" />
                      ) : (
                        <ChevronRight className="h-4 w-4" />
                      )}
                    </TableCell>
                    <TableCell className="font-mono text-sm">
                      {batch.id}
                    </TableCell>
                    <TableCell>{getStatusBadge(batch.status)}</TableCell>
                    <TableCell className="font-mono text-sm">
                      {batch.endpoint}
                    </TableCell>
                    <TableCell>{formatTimestamp(batch.created_at)}</TableCell>
                    <TableCell>
                      {batch.completed_at
                        ? formatTimestamp(batch.completed_at)
                        : "-"}
                    </TableCell>
                    <TableCell className="font-mono text-sm">
                      {batch.input_file_id}
                    </TableCell>
                    <TableCell className="font-mono text-sm">
                      {batch.output_file_id || "-"}
                    </TableCell>
                  </TableRow>

                  {isExpanded && (
                    <TableRow>
                      <TableCell colSpan={8} className="bg-muted/30 p-0">
                        <div className="p-4 space-y-4">
                          <div className="grid grid-cols-1 md:grid-cols-2 lg:grid-cols-3 gap-4">
                            {batch.request_counts && (
                              <Card>
                                <CardHeader className="pb-2">
                                  <CardTitle className="text-sm">
                                    Request Counts
                                  </CardTitle>
                                </CardHeader>
                                <CardContent className="space-y-1 text-sm">
                                  <div className="flex justify-between">
                                    <span className="text-muted-foreground">
                                      Total
                                    </span>
                                    <span className="font-medium">
                                      {batch.request_counts.total}
                                    </span>
                                  </div>
                                  <div className="flex justify-between">
                                    <span className="text-muted-foreground">
                                      Completed
                                    </span>
                                    <span className="font-medium text-green-600 dark:text-green-400">
                                      {batch.request_counts.completed}
                                    </span>
                                  </div>
                                  <div className="flex justify-between">
                                    <span className="text-muted-foreground">
                                      Failed
                                    </span>
                                    <span className="font-medium text-red-600 dark:text-red-400">
                                      {batch.request_counts.failed}
                                    </span>
                                  </div>
                                </CardContent>
                              </Card>
                            )}

                            <Card>
                              <CardHeader className="pb-2">
                                <CardTitle className="text-sm">
                                  Timestamps
                                </CardTitle>
                              </CardHeader>
                              <CardContent className="space-y-1 text-sm">
                                <div className="flex justify-between">
                                  <span className="text-muted-foreground">
                                    Created
                                  </span>
                                  <span>
                                    {formatTimestamp(batch.created_at)}
                                  </span>
                                </div>
                                {batch.completed_at && (
                                  <div className="flex justify-between">
                                    <span className="text-muted-foreground">
                                      Completed
                                    </span>
                                    <span>
                                      {formatTimestamp(batch.completed_at)}
                                    </span>
                                  </div>
                                )}
                                {batch.cancelled_at && (
                                  <div className="flex justify-between">
                                    <span className="text-muted-foreground">
                                      Cancelled
                                    </span>
                                    <span>
                                      {formatTimestamp(batch.cancelled_at)}
                                    </span>
                                  </div>
                                )}
                                {batch.failed_at && (
                                  <div className="flex justify-between">
                                    <span className="text-muted-foreground">
                                      Failed
                                    </span>
                                    <span>
                                      {formatTimestamp(batch.failed_at)}
                                    </span>
                                  </div>
                                )}
                                {batch.expired_at && (
                                  <div className="flex justify-between">
                                    <span className="text-muted-foreground">
                                      Expired
                                    </span>
                                    <span>
                                      {formatTimestamp(batch.expired_at)}
                                    </span>
                                  </div>
                                )}
                              </CardContent>
                            </Card>

                            <Card>
                              <CardHeader className="pb-2">
                                <CardTitle className="text-sm">
                                  File References
                                </CardTitle>
                              </CardHeader>
                              <CardContent className="space-y-1 text-sm">
                                <div className="flex justify-between">
                                  <span className="text-muted-foreground">
                                    Input File
                                  </span>
                                  <span className="font-mono">
                                    {batch.input_file_id}
                                  </span>
                                </div>
                                <div className="flex justify-between">
                                  <span className="text-muted-foreground">
                                    Output File
                                  </span>
                                  <span className="font-mono">
                                    {batch.output_file_id || "-"}
                                  </span>
                                </div>
                                <div className="flex justify-between">
                                  <span className="text-muted-foreground">
                                    Error File
                                  </span>
                                  <span className="font-mono">
                                    {batch.error_file_id || "-"}
                                  </span>
                                </div>
                              </CardContent>
                            </Card>
                          </div>
                        </div>
                      </TableCell>
                    </TableRow>
                  )}
                </React.Fragment>
              );
            })}
          </TableBody>
        </Table>
      </div>
    );
  };

  return (
    <div className="space-y-4">
      <div className="flex items-center justify-between">
        <h1 className="text-2xl font-semibold">Batches</h1>
      </div>
      {renderContent()}
    </div>
  );
}
