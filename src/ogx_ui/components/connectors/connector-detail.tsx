"use client";

import React, { useState } from "react";
import { ChevronDown, ChevronRight, ArrowLeft } from "lucide-react";
import { Badge } from "@/components/ui/badge";
import { Button } from "@/components/ui/button";
import {
  Card,
  CardContent,
  CardDescription,
  CardHeader,
  CardTitle,
} from "@/components/ui/card";
import {
  Collapsible,
  CollapsibleContent,
  CollapsibleTrigger,
} from "@/components/ui/collapsible";
import { Skeleton } from "@/components/ui/skeleton";

interface ConnectorTool {
  identifier: string;
  description?: string;
  parameters?: Record<string, unknown>;
  metadata?: Record<string, unknown>;
}

interface Connector {
  identifier: string;
  type: string;
  provider_id?: string;
  metadata?: Record<string, unknown>;
}

interface ConnectorDetailProps {
  connector: Connector;
  onBack: () => void;
}

export function ConnectorDetail({ connector, onBack }: ConnectorDetailProps) {
  const [tools, setTools] = useState<ConnectorTool[]>([]);
  const [loading, setLoading] = useState(true);
  const [error, setError] = useState<string | null>(null);
  const [expandedTools, setExpandedTools] = useState<Set<string>>(new Set());

  React.useEffect(() => {
    async function fetchTools() {
      setLoading(true);
      setError(null);
      try {
        const response = await fetch(
          `/api/v1/connectors/${encodeURIComponent(connector.identifier)}/tools`
        );
        if (!response.ok) {
          throw new Error(
            `Failed to fetch tools: ${response.status} ${response.statusText}`
          );
        }
        const data = await response.json();
        setTools(Array.isArray(data) ? data : (data.data ?? []));
      } catch (err) {
        const message =
          err instanceof Error ? err.message : "Unknown error occurred";
        setError(message);
      } finally {
        setLoading(false);
      }
    }

    fetchTools();
  }, [connector.identifier]);

  const toggleTool = (toolName: string) => {
    setExpandedTools(prev => {
      const next = new Set(prev);
      if (next.has(toolName)) {
        next.delete(toolName);
      } else {
        next.add(toolName);
      }
      return next;
    });
  };

  return (
    <div className="space-y-6">
      <Button variant="ghost" onClick={onBack} className="gap-2">
        <ArrowLeft className="h-4 w-4" />
        Back to Connectors
      </Button>

      <Card>
        <CardHeader>
          <CardTitle className="flex items-center gap-2">
            {connector.identifier}
            <Badge variant="secondary">{connector.type}</Badge>
          </CardTitle>
          {connector.provider_id && (
            <CardDescription>Provider: {connector.provider_id}</CardDescription>
          )}
        </CardHeader>
        <CardContent>
          <div className="grid grid-cols-2 gap-4 text-sm">
            <div>
              <span className="font-medium text-muted-foreground">ID</span>
              <p className="font-mono mt-1">{connector.identifier}</p>
            </div>
            <div>
              <span className="font-medium text-muted-foreground">Type</span>
              <p className="mt-1">{connector.type}</p>
            </div>
            {connector.provider_id && (
              <div>
                <span className="font-medium text-muted-foreground">
                  Provider
                </span>
                <p className="mt-1">{connector.provider_id}</p>
              </div>
            )}
            {connector.metadata &&
              Object.entries(connector.metadata).map(([key, value]) => (
                <div key={key}>
                  <span className="font-medium text-muted-foreground">
                    {key}
                  </span>
                  <p className="mt-1 break-all">
                    {typeof value === "string" ? value : JSON.stringify(value)}
                  </p>
                </div>
              ))}
          </div>
        </CardContent>
      </Card>

      <div className="space-y-2">
        <h2 className="text-lg font-semibold">Tools</h2>

        {loading && (
          <div className="space-y-2">
            <Skeleton className="h-16 w-full" />
            <Skeleton className="h-16 w-full" />
            <Skeleton className="h-16 w-full" />
          </div>
        )}

        {error && <div className="text-destructive">Error: {error}</div>}

        {!loading && !error && tools.length === 0 && (
          <p className="text-muted-foreground py-4">
            No tools found for this connector.
          </p>
        )}

        {!loading &&
          !error &&
          tools.map(tool => (
            <Card key={tool.identifier}>
              <Collapsible
                open={expandedTools.has(tool.identifier)}
                onOpenChange={() => toggleTool(tool.identifier)}
              >
                <CollapsibleTrigger className="w-full">
                  <CardHeader className="cursor-pointer hover:bg-muted/50 rounded-t-xl">
                    <div className="flex items-center gap-2">
                      {expandedTools.has(tool.identifier) ? (
                        <ChevronDown className="h-4 w-4 shrink-0" />
                      ) : (
                        <ChevronRight className="h-4 w-4 shrink-0" />
                      )}
                      <CardTitle className="text-sm font-mono">
                        {tool.identifier}
                      </CardTitle>
                    </div>
                    {tool.description && (
                      <CardDescription className="text-left ml-6">
                        {tool.description}
                      </CardDescription>
                    )}
                  </CardHeader>
                </CollapsibleTrigger>
                <CollapsibleContent>
                  <CardContent>
                    {tool.parameters ? (
                      <pre className="bg-muted p-4 rounded-md overflow-auto text-xs font-mono max-h-96">
                        {JSON.stringify(tool.parameters, null, 2)}
                      </pre>
                    ) : (
                      <p className="text-muted-foreground text-sm">
                        No input schema available.
                      </p>
                    )}
                  </CardContent>
                </CollapsibleContent>
              </Collapsible>
            </Card>
          ))}
      </div>
    </div>
  );
}
