"use client";

import React, { useState, useEffect, useCallback } from "react";
import { useParams, useRouter } from "next/navigation";
import { ArrowLeft } from "lucide-react";
import { Button } from "@/components/ui/button";
import { Badge } from "@/components/ui/badge";
import { Card, CardContent, CardHeader, CardTitle } from "@/components/ui/card";
import { Skeleton } from "@/components/ui/skeleton";

interface Conversation {
  id: string;
  object: string;
  created_at: number;
  metadata?: Record<string, string> | null;
}

interface ConversationItem {
  id: string;
  type: string;
  role?: string;
  content?: ContentPart[] | string;
  status?: string;
  [key: string]: unknown;
}

interface ContentPart {
  type: string;
  text?: string;
  [key: string]: unknown;
}

interface ConversationItemList {
  object: string;
  data: ConversationItem[];
  first_id: string | null;
  last_id: string | null;
  has_more: boolean;
}

type Status = "loading" | "error" | "success";

function formatTimestamp(epochSeconds: number): string {
  return new Date(epochSeconds * 1000).toLocaleString();
}

function getRoleBadgeVariant(
  role: string
): "default" | "secondary" | "outline" | "destructive" {
  switch (role) {
    case "assistant":
      return "default";
    case "user":
      return "secondary";
    case "system":
      return "outline";
    default:
      return "secondary";
  }
}

function getRoleStyles(role: string): string {
  switch (role) {
    case "assistant":
      return "bg-blue-50 dark:bg-blue-950/30 border-blue-200 dark:border-blue-800";
    case "user":
      return "bg-gray-50 dark:bg-gray-900/30 border-gray-200 dark:border-gray-800";
    case "system":
      return "bg-amber-50 dark:bg-amber-950/30 border-amber-200 dark:border-amber-800";
    default:
      return "bg-muted/30 border-border";
  }
}

function extractTextContent(item: ConversationItem): string {
  if (typeof item.content === "string") {
    return item.content;
  }

  if (Array.isArray(item.content)) {
    return item.content
      .map(part => {
        if (part.type === "output_text" || part.type === "input_text") {
          return part.text ?? "";
        }
        if (part.type === "text") {
          return part.text ?? "";
        }
        return `[${part.type}]`;
      })
      .join("\n");
  }

  if (item.type === "function_call") {
    const name = (item.name as string) ?? "unknown";
    const args = (item.arguments as string) ?? "";
    return `Function call: ${name}(${args})`;
  }

  if (item.type === "function_call_output") {
    return `Function output: ${(item.output as string) ?? ""}`;
  }

  return JSON.stringify(item, null, 2);
}

export default function ConversationDetailPage() {
  const params = useParams();
  const router = useRouter();
  const id = params.id as string;

  const [conversation, setConversation] = useState<Conversation | null>(null);
  const [items, setItems] = useState<ConversationItem[]>([]);
  const [convStatus, setConvStatus] = useState<Status>("loading");
  const [itemsStatus, setItemsStatus] = useState<Status>("loading");
  const [convError, setConvError] = useState<Error | null>(null);
  const [itemsError, setItemsError] = useState<Error | null>(null);

  const fetchData = useCallback(async () => {
    if (!id) return;

    setConvStatus("loading");
    setItemsStatus("loading");
    setConvError(null);
    setItemsError(null);

    const [convResult, itemsResult] = await Promise.allSettled([
      fetch(`/api/v1/conversations/${encodeURIComponent(id)}`),
      fetch(
        `/api/v1/conversations/${encodeURIComponent(id)}/items?order=asc&limit=100`
      ),
    ]);

    if (convResult.status === "fulfilled" && convResult.value.ok) {
      const data = await convResult.value.json();
      setConversation(data);
      setConvStatus("success");
    } else {
      const msg =
        convResult.status === "rejected"
          ? (convResult.reason?.message ?? "Network error")
          : `Failed to fetch conversation: ${convResult.value.status}`;
      setConvError(new Error(msg));
      setConvStatus("error");
    }

    if (itemsResult.status === "fulfilled" && itemsResult.value.ok) {
      const data: ConversationItemList = await itemsResult.value.json();
      setItems(data.data ?? []);
      setItemsStatus("success");
    } else {
      const msg =
        itemsResult.status === "rejected"
          ? (itemsResult.reason?.message ?? "Network error")
          : `Failed to fetch items: ${itemsResult.value.status}`;
      setItemsError(new Error(msg));
      setItemsStatus("error");
    }
  }, [id]);

  useEffect(() => {
    fetchData();
  }, [fetchData]);

  if (convStatus === "loading") {
    return (
      <div className="space-y-4">
        <Skeleton className="h-8 w-48" />
        <Skeleton className="h-24 w-full" />
        <Skeleton className="h-4 w-full" />
        <Skeleton className="h-4 w-full" />
        <Skeleton className="h-4 w-3/4" />
      </div>
    );
  }

  if (convStatus === "error") {
    return (
      <div className="space-y-4">
        <Button variant="ghost" onClick={() => router.push("/conversations")}>
          <ArrowLeft className="h-4 w-4 mr-2" />
          Back to Conversations
        </Button>
        <div className="text-center py-8">
          <p className="text-destructive mb-2">Failed to load conversation</p>
          <p className="text-sm text-muted-foreground">{convError?.message}</p>
        </div>
      </div>
    );
  }

  return (
    <div className="space-y-6">
      <div className="flex items-center gap-4">
        <Button variant="ghost" onClick={() => router.push("/conversations")}>
          <ArrowLeft className="h-4 w-4 mr-2" />
          Back
        </Button>
        <h1 className="text-2xl font-semibold">Conversation Detail</h1>
      </div>

      {conversation && (
        <Card>
          <CardHeader>
            <CardTitle className="text-lg">Properties</CardTitle>
          </CardHeader>
          <CardContent>
            <dl className="grid grid-cols-1 sm:grid-cols-2 gap-x-6 gap-y-3 text-sm">
              <div>
                <dt className="text-muted-foreground">ID</dt>
                <dd className="font-mono mt-0.5">{conversation.id}</dd>
              </div>
              <div>
                <dt className="text-muted-foreground">Created At</dt>
                <dd className="mt-0.5">
                  {formatTimestamp(conversation.created_at)}
                </dd>
              </div>
              {conversation.metadata &&
                Object.keys(conversation.metadata).length > 0 && (
                  <div className="sm:col-span-2">
                    <dt className="text-muted-foreground">Metadata</dt>
                    <dd className="mt-1 flex flex-wrap gap-1">
                      {Object.entries(conversation.metadata).map(
                        ([key, value]) => (
                          <Badge key={key} variant="secondary">
                            {key}: {value}
                          </Badge>
                        )
                      )}
                    </dd>
                  </div>
                )}
            </dl>
          </CardContent>
        </Card>
      )}

      <Card>
        <CardHeader>
          <CardTitle className="text-lg">
            Items{" "}
            {itemsStatus === "success" && (
              <Badge variant="outline" className="ml-2">
                {items.length}
              </Badge>
            )}
          </CardTitle>
        </CardHeader>
        <CardContent>
          {itemsStatus === "loading" && (
            <div className="space-y-3">
              <Skeleton className="h-16 w-full" />
              <Skeleton className="h-16 w-full" />
              <Skeleton className="h-16 w-full" />
            </div>
          )}

          {itemsStatus === "error" && (
            <div className="text-center py-4">
              <p className="text-destructive text-sm">Failed to load items</p>
              <p className="text-xs text-muted-foreground mt-1">
                {itemsError?.message}
              </p>
            </div>
          )}

          {itemsStatus === "success" && items.length === 0 && (
            <p className="text-muted-foreground text-center py-4">
              No items in this conversation.
            </p>
          )}

          {itemsStatus === "success" && items.length > 0 && (
            <div className="space-y-3 max-h-[600px] overflow-y-auto pr-1">
              {items.map((item, index) => {
                const role = item.role ?? item.type ?? "unknown";
                const textContent = extractTextContent(item);

                return (
                  <div
                    key={item.id ?? index}
                    className={`rounded-lg border p-4 ${getRoleStyles(role)}`}
                  >
                    <div className="flex items-center gap-2 mb-2">
                      <Badge variant={getRoleBadgeVariant(role)}>{role}</Badge>
                      <span className="text-xs text-muted-foreground font-mono">
                        {item.id}
                      </span>
                      {item.type && item.type !== "message" && (
                        <Badge variant="outline" className="text-xs">
                          {item.type}
                        </Badge>
                      )}
                      {item.status && (
                        <Badge variant="outline" className="text-xs">
                          {item.status}
                        </Badge>
                      )}
                    </div>
                    <div className="text-sm whitespace-pre-wrap break-words">
                      {textContent}
                    </div>
                  </div>
                );
              })}
            </div>
          )}
        </CardContent>
      </Card>
    </div>
  );
}
