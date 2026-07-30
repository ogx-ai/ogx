"use client";

import React, { useState, useEffect, useCallback } from "react";
import { Search } from "lucide-react";
import { Badge } from "@/components/ui/badge";
import { Input } from "@/components/ui/input";
import { Skeleton } from "@/components/ui/skeleton";
import {
  Table,
  TableBody,
  TableCell,
  TableHead,
  TableHeader,
  TableRow,
} from "@/components/ui/table";
import { ConnectorDetail } from "./connector-detail";

interface Connector {
  identifier: string;
  type: string;
  provider_id?: string;
  metadata?: Record<string, unknown>;
}

type Status = "loading" | "error" | "success";

export function ConnectorsManagement() {
  const [connectors, setConnectors] = useState<Connector[]>([]);
  const [status, setStatus] = useState<Status>("loading");
  const [error, setError] = useState<Error | null>(null);
  const [searchTerm, setSearchTerm] = useState("");
  const [selectedConnector, setSelectedConnector] = useState<Connector | null>(
    null
  );

  const fetchConnectors = useCallback(async () => {
    setStatus("loading");
    setError(null);
    try {
      const response = await fetch("/api/v1/connectors");
      if (response.status === 404 || response.status === 501) {
        setConnectors([]);
        setStatus("success");
        return;
      }
      if (!response.ok) {
        throw new Error(
          `Failed to fetch connectors: ${response.status} ${response.statusText}`
        );
      }
      const data = await response.json();
      setConnectors(Array.isArray(data) ? data : (data.data ?? []));
      setStatus("success");
    } catch (err) {
      const errorObj =
        err instanceof Error ? err : new Error("Unknown error occurred");
      setError(errorObj);
      setStatus("error");
    }
  }, []);

  useEffect(() => {
    fetchConnectors();
  }, [fetchConnectors]);

  if (selectedConnector) {
    return (
      <ConnectorDetail
        connector={selectedConnector}
        onBack={() => setSelectedConnector(null)}
      />
    );
  }

  const renderContent = () => {
    if (status === "loading") {
      return (
        <div className="space-y-2">
          <Skeleton className="h-8 w-full" />
          <Skeleton className="h-4 w-full" />
          <Skeleton className="h-4 w-full" />
          <Skeleton className="h-4 w-full" />
        </div>
      );
    }

    if (status === "error") {
      return <div className="text-destructive">Error: {error?.message}</div>;
    }

    if (connectors.length === 0) {
      return (
        <div className="text-center py-8">
          <p className="text-muted-foreground">No connectors found.</p>
        </div>
      );
    }

    const filteredConnectors = connectors.filter(connector => {
      if (!searchTerm) return true;
      const searchLower = searchTerm.toLowerCase();
      return (
        connector.identifier.toLowerCase().includes(searchLower) ||
        connector.type.toLowerCase().includes(searchLower) ||
        (connector.provider_id ?? "").toLowerCase().includes(searchLower)
      );
    });

    return (
      <div className="space-y-4">
        <div className="relative flex-1 max-w-md">
          <Search className="absolute left-3 top-1/2 transform -translate-y-1/2 text-muted-foreground h-4 w-4" />
          <Input
            placeholder="Search connectors..."
            value={searchTerm}
            onChange={e => setSearchTerm(e.target.value)}
            className="pl-10"
          />
        </div>

        <div className="overflow-auto flex-1 min-h-0">
          <Table>
            <TableHeader>
              <TableRow>
                <TableHead>Connector ID</TableHead>
                <TableHead>Type</TableHead>
                <TableHead>Provider</TableHead>
              </TableRow>
            </TableHeader>
            <TableBody>
              {filteredConnectors.map(connector => (
                <TableRow
                  key={connector.identifier}
                  onClick={() => setSelectedConnector(connector)}
                  className="cursor-pointer hover:bg-muted/50"
                >
                  <TableCell className="font-mono text-blue-600 dark:text-blue-400">
                    {connector.identifier}
                  </TableCell>
                  <TableCell>
                    <Badge variant="secondary">{connector.type}</Badge>
                  </TableCell>
                  <TableCell>{connector.provider_id ?? "-"}</TableCell>
                </TableRow>
              ))}
              {filteredConnectors.length === 0 && (
                <TableRow>
                  <TableCell
                    colSpan={3}
                    className="text-center text-muted-foreground py-8"
                  >
                    No connectors match your search.
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
        <h1 className="text-2xl font-semibold">Connectors</h1>
      </div>
      {renderContent()}
    </div>
  );
}
