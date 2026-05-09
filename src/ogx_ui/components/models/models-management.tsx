"use client";

import React, { useState, useEffect, useCallback } from "react";
import { Search } from "lucide-react";
import { Input } from "@/components/ui/input";
import { Badge } from "@/components/ui/badge";
import {
  Table,
  TableBody,
  TableCell,
  TableHead,
  TableHeader,
  TableRow,
} from "@/components/ui/table";
import { Skeleton } from "@/components/ui/skeleton";
import { useAuthClient } from "@/hooks/use-auth-client";

type Status = "idle" | "loading" | "error" | "success";

function getModelTypeBadge(modelType: string) {
  switch (modelType) {
    case "llm":
      return (
        <Badge className="bg-blue-100 text-blue-800 border-blue-200 dark:bg-blue-900 dark:text-blue-200 dark:border-blue-800">
          LLM
        </Badge>
      );
    case "embedding":
      return (
        <Badge className="bg-emerald-100 text-emerald-800 border-emerald-200 dark:bg-emerald-900 dark:text-emerald-200 dark:border-emerald-800">
          Embedding
        </Badge>
      );
    case "moderation":
      return (
        <Badge className="bg-amber-100 text-amber-800 border-amber-200 dark:bg-amber-900 dark:text-amber-200 dark:border-amber-800">
          Moderation
        </Badge>
      );
    default:
      return <Badge variant="outline">{modelType}</Badge>;
  }
}

export function ModelsManagement() {
  const client = useAuthClient();
  const [models, setModels] = useState<Record<string, unknown>[]>([]);
  const [status, setStatus] = useState<Status>("idle");
  const [error, setError] = useState<Error | null>(null);
  const [searchTerm, setSearchTerm] = useState("");

  const fetchModels = useCallback(async () => {
    setStatus("loading");
    setError(null);
    try {
      const response = await client.models.list();
      const raw = Array.isArray(response)
        ? response
        : response &&
            typeof response === "object" &&
            "data" in (response as Record<string, unknown>)
          ? ((response as Record<string, unknown>).data as Record<
              string,
              unknown
            >[])
          : [];
      const items = raw.map((m: Record<string, unknown>) => {
        const meta = (m.custom_metadata ?? {}) as Record<string, unknown>;
        return {
          ...m,
          identifier: m.identifier ?? m.id,
          model_type: m.model_type ?? meta.model_type,
          provider_id: m.provider_id ?? meta.provider_id,
          provider_resource_id:
            m.provider_resource_id ?? meta.provider_resource_id,
        };
      });
      setModels(items);
      setStatus("success");
    } catch (err: unknown) {
      console.error("Failed to fetch models:", err);
      setError(err instanceof Error ? err : new Error("Unknown error"));
      setStatus("error");
    }
  }, [client]);

  useEffect(() => {
    fetchModels();
  }, [fetchModels]);

  const filteredModels = models.filter(model => {
    if (!searchTerm) return true;
    const term = searchTerm.toLowerCase();
    return (
      (model.identifier ?? model.id ?? "").toLowerCase().includes(term) ||
      (model.model_type ?? "").toLowerCase().includes(term) ||
      (model.provider_id ?? "").toLowerCase().includes(term) ||
      (model.provider_resource_id ?? "").toLowerCase().includes(term)
    );
  });

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
      return <div className="text-destructive">Error: {error?.message}</div>;
    }

    if (models.length === 0) {
      return (
        <div className="text-center py-8">
          <p className="text-muted-foreground">No models registered.</p>
        </div>
      );
    }

    return (
      <div className="space-y-4">
        <div className="relative flex-1 max-w-md">
          <Search className="absolute left-3 top-1/2 transform -translate-y-1/2 text-muted-foreground h-4 w-4" />
          <Input
            placeholder="Search models..."
            value={searchTerm}
            onChange={e => setSearchTerm(e.target.value)}
            className="pl-10"
          />
        </div>

        <div className="overflow-auto flex-1 min-h-0">
          <Table>
            <TableHeader>
              <TableRow>
                <TableHead>Model ID</TableHead>
                <TableHead>Model Type</TableHead>
                <TableHead>Provider</TableHead>
                <TableHead>Provider Resource ID</TableHead>
              </TableRow>
            </TableHeader>
            <TableBody>
              {filteredModels.map(model => {
                const modelId = model.identifier ?? model.id ?? "unknown";
                const modelType = model.model_type ?? "unknown";
                const providerId = model.provider_id ?? "-";
                const providerResourceId =
                  model.provider_resource_id ??
                  model.metadata?.provider_resource_id ??
                  "-";

                return (
                  <TableRow key={modelId}>
                    <TableCell className="font-mono text-sm">
                      {modelId}
                    </TableCell>
                    <TableCell>{getModelTypeBadge(modelType)}</TableCell>
                    <TableCell>{providerId}</TableCell>
                    <TableCell className="font-mono text-sm text-muted-foreground">
                      {providerResourceId}
                    </TableCell>
                  </TableRow>
                );
              })}
              {filteredModels.length === 0 && (
                <TableRow>
                  <TableCell
                    colSpan={4}
                    className="text-center text-muted-foreground py-8"
                  >
                    No models match your search.
                  </TableCell>
                </TableRow>
              )}
            </TableBody>
          </Table>
        </div>
      </div>
    );
  };

  return (
    <div className="space-y-4">
      <div className="flex items-center justify-between">
        <h1 className="text-2xl font-semibold">Models</h1>
      </div>
      {renderContent()}
    </div>
  );
}
