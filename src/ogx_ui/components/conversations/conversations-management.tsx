"use client";

import React, { useState, useEffect, useCallback } from "react";
import { useRouter } from "next/navigation";
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

interface Conversation {
  id: string;
  object: string;
  created_at: number;
  metadata?: Record<string, string> | null;
}

type Status = "loading" | "error" | "success";

function formatTimestamp(epochSeconds: number): string {
  return new Date(epochSeconds * 1000).toLocaleString();
}

export function ConversationsManagement() {
  const router = useRouter();
  const [conversations, setConversations] = useState<Conversation[]>([]);
  const [status, setStatus] = useState<Status>("loading");
  const [error, setError] = useState<Error | null>(null);
  const [searchTerm, setSearchTerm] = useState("");

  const fetchConversations = useCallback(async () => {
    setStatus("loading");
    setError(null);
    try {
      const response = await fetch("/api/v1/conversations");
      if (
        response.status === 404 ||
        response.status === 405 ||
        response.status === 501
      ) {
        setConversations([]);
        setStatus("success");
        return;
      }
      if (!response.ok) {
        throw new Error(
          `Failed to fetch conversations: ${response.status} ${response.statusText}`
        );
      }
      const data = await response.json();
      setConversations(Array.isArray(data) ? data : (data.data ?? []));
      setStatus("success");
    } catch (err) {
      const errorObj =
        err instanceof Error ? err : new Error("Unknown error occurred");
      setError(errorObj);
      setStatus("error");
    }
  }, []);

  useEffect(() => {
    fetchConversations();
  }, [fetchConversations]);

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
      return (
        <div className="text-center py-8">
          <p className="text-destructive mb-2">Failed to load conversations</p>
          <p className="text-sm text-muted-foreground">
            {error?.message ?? "Unknown error"}
          </p>
          <p className="text-sm text-muted-foreground mt-2">
            The conversations API may not be available on this server.
          </p>
        </div>
      );
    }

    if (conversations.length === 0) {
      return (
        <div className="text-center py-8">
          <p className="text-muted-foreground">No conversations found.</p>
        </div>
      );
    }

    const filteredConversations = conversations.filter(conversation => {
      if (!searchTerm) return true;
      const searchLower = searchTerm.toLowerCase();
      return (
        conversation.id.toLowerCase().includes(searchLower) ||
        (conversation.metadata &&
          JSON.stringify(conversation.metadata)
            .toLowerCase()
            .includes(searchLower))
      );
    });

    return (
      <div className="space-y-4">
        <div className="relative flex-1 max-w-md">
          <Search className="absolute left-3 top-1/2 transform -translate-y-1/2 text-muted-foreground h-4 w-4" />
          <Input
            placeholder="Search conversations..."
            value={searchTerm}
            onChange={e => setSearchTerm(e.target.value)}
            className="pl-10"
          />
        </div>

        <div className="overflow-auto flex-1 min-h-0">
          <Table>
            <TableHeader>
              <TableRow>
                <TableHead>Conversation ID</TableHead>
                <TableHead>Created At</TableHead>
                <TableHead>Metadata</TableHead>
              </TableRow>
            </TableHeader>
            <TableBody>
              {filteredConversations.map(conversation => (
                <TableRow
                  key={conversation.id}
                  onClick={() =>
                    router.push(`/conversations/${conversation.id}`)
                  }
                  className="cursor-pointer hover:bg-muted/50"
                >
                  <TableCell className="font-mono text-blue-600 dark:text-blue-400">
                    {conversation.id}
                  </TableCell>
                  <TableCell>
                    {formatTimestamp(conversation.created_at)}
                  </TableCell>
                  <TableCell>
                    {conversation.metadata &&
                    Object.keys(conversation.metadata).length > 0 ? (
                      <div className="flex flex-wrap gap-1">
                        {Object.entries(conversation.metadata)
                          .slice(0, 3)
                          .map(([key, value]) => (
                            <Badge key={key} variant="secondary">
                              {key}: {value}
                            </Badge>
                          ))}
                        {Object.keys(conversation.metadata).length > 3 && (
                          <Badge variant="outline">
                            +{Object.keys(conversation.metadata).length - 3}{" "}
                            more
                          </Badge>
                        )}
                      </div>
                    ) : (
                      <span className="text-muted-foreground">-</span>
                    )}
                  </TableCell>
                </TableRow>
              ))}
              {filteredConversations.length === 0 && (
                <TableRow>
                  <TableCell
                    colSpan={3}
                    className="text-center text-muted-foreground py-8"
                  >
                    No conversations match your search.
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
        <h1 className="text-2xl font-semibold">Conversations</h1>
      </div>
      {renderContent()}
    </div>
  );
}
