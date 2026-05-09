"use client";

import React, { useState, useEffect, useCallback } from "react";
import { useAuthClient } from "@/hooks/use-auth-client";
import { Search, ChevronRight, ChevronDown } from "lucide-react";
import { Input } from "@/components/ui/input";
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
import { Tabs, TabsContent, TabsList, TabsTrigger } from "@/components/ui/tabs";
import {
  Collapsible,
  CollapsibleContent,
  CollapsibleTrigger,
} from "@/components/ui/collapsible";

interface ToolGroup {
  identifier: string;
  provider_id: string;
  type: string;
  description?: string;
  provider_resource_id?: string;
  [key: string]: unknown;
}

interface ToolParam {
  name: string;
  parameter_type: string;
  description?: string;
  required?: boolean;
  default?: unknown;
  [key: string]: unknown;
}

interface Tool {
  identifier: string;
  tool_name?: string;
  description?: string;
  toolgroup_id?: string;
  provider_id?: string;
  parameters?: ToolParam[];
  metadata?: Record<string, unknown>;
  [key: string]: unknown;
}

type LoadStatus = "loading" | "idle" | "error";

export function ToolsManagement() {
  const client = useAuthClient();
  const [toolGroups, setToolGroups] = useState<ToolGroup[]>([]);
  const [tools, setTools] = useState<Tool[]>([]);
  const [toolGroupsStatus, setToolGroupsStatus] =
    useState<LoadStatus>("loading");
  const [toolsStatus, setToolsStatus] = useState<LoadStatus>("loading");
  const [toolGroupsError, setToolGroupsError] = useState<string | null>(null);
  const [toolsError, setToolsError] = useState<string | null>(null);
  const [searchTerm, setSearchTerm] = useState("");
  const [expandedToolGroups, setExpandedToolGroups] = useState<Set<string>>(
    new Set()
  );
  const [expandedTools, setExpandedTools] = useState<Set<string>>(new Set());
  const [toolGroupTools, setToolGroupTools] = useState<Record<string, Tool[]>>(
    {}
  );

  const fetchToolGroups = useCallback(async () => {
    try {
      setToolGroupsStatus("loading");
      const response = await client.toolgroups.list();

      const groupsArray = Array.isArray(response)
        ? response
        : response &&
            typeof response === "object" &&
            "data" in response &&
            Array.isArray((response as { data: unknown }).data)
          ? (response as { data: ToolGroup[] }).data
          : [];

      setToolGroups(groupsArray);
      setToolGroupsStatus("idle");
    } catch (err) {
      const msg = err instanceof Error ? err.message : String(err);
      if (
        msg.includes("404") ||
        msg.includes("Not Found") ||
        msg.includes("501")
      ) {
        setToolGroups([]);
        setToolGroupsStatus("idle");
      } else {
        console.error("Failed to fetch tool groups:", err);
        setToolGroupsError(msg);
        setToolGroupsStatus("error");
      }
    }
  }, [client]);

  const fetchTools = useCallback(async () => {
    try {
      setToolsStatus("loading");
      const res = await fetch("/api/v1/tools");
      if (!res.ok) {
        if (res.status === 404 || res.status === 501) {
          setTools([]);
          setToolsStatus("idle");
          return;
        }
        throw new Error(`${res.status} ${res.statusText}`);
      }
      const response = await res.json();

      const toolsArray = Array.isArray(response)
        ? response
        : response &&
            typeof response === "object" &&
            "data" in response &&
            Array.isArray((response as { data: unknown }).data)
          ? (response as { data: Tool[] }).data
          : [];

      setTools(toolsArray);
      setToolsStatus("idle");
    } catch (err) {
      const msg = err instanceof Error ? err.message : String(err);
      if (
        msg.includes("404") ||
        msg.includes("Not Found") ||
        msg.includes("501")
      ) {
        setTools([]);
        setToolsStatus("idle");
      } else {
        console.error("Failed to fetch tools:", err);
        setToolsError(msg);
        setToolsStatus("error");
      }
    }
  }, [client]);

  useEffect(() => {
    fetchToolGroups();
    fetchTools();
  }, [fetchToolGroups, fetchTools]);

  const toggleToolGroup = useCallback(
    (identifier: string) => {
      setExpandedToolGroups(prev => {
        const next = new Set(prev);
        if (next.has(identifier)) {
          next.delete(identifier);
        } else {
          next.add(identifier);

          // Load tools for this group if not already loaded
          if (!toolGroupTools[identifier]) {
            const groupTools = tools.filter(t => t.toolgroup_id === identifier);
            setToolGroupTools(prev => ({
              ...prev,
              [identifier]: groupTools,
            }));
          }
        }
        return next;
      });
    },
    [tools, toolGroupTools]
  );

  const toggleTool = useCallback((identifier: string) => {
    setExpandedTools(prev => {
      const next = new Set(prev);
      if (next.has(identifier)) {
        next.delete(identifier);
      } else {
        next.add(identifier);
      }
      return next;
    });
  }, []);

  const filteredTools = tools.filter(tool => {
    if (!searchTerm) return true;
    const lower = searchTerm.toLowerCase();
    return (
      (tool.identifier || "").toLowerCase().includes(lower) ||
      (tool.tool_name || "").toLowerCase().includes(lower) ||
      (tool.description || "").toLowerCase().includes(lower) ||
      (tool.toolgroup_id || "").toLowerCase().includes(lower)
    );
  });

  const renderToolGroupsTab = () => {
    if (toolGroupsStatus === "loading") {
      return (
        <div className="space-y-2">
          <Skeleton className="h-8 w-full" />
          <Skeleton className="h-4 w-full" />
          <Skeleton className="h-4 w-full" />
        </div>
      );
    }

    if (toolGroupsStatus === "error") {
      return <div className="text-destructive">Error: {toolGroupsError}</div>;
    }

    if (toolGroups.length === 0) {
      return (
        <div className="text-center py-8">
          <p className="text-muted-foreground">No tool groups found.</p>
        </div>
      );
    }

    return (
      <div className="overflow-auto flex-1 min-h-0">
        <Table>
          <TableHeader>
            <TableRow>
              <TableHead className="w-8"></TableHead>
              <TableHead>Toolgroup ID</TableHead>
              <TableHead>Provider ID</TableHead>
              <TableHead>Description</TableHead>
            </TableRow>
          </TableHeader>
          <TableBody>
            {toolGroups.map(group => {
              const isExpanded = expandedToolGroups.has(group.identifier);
              const groupTools = toolGroupTools[group.identifier] || [];

              return (
                <React.Fragment key={group.identifier}>
                  <TableRow
                    className="cursor-pointer hover:bg-muted/50"
                    onClick={() => toggleToolGroup(group.identifier)}
                  >
                    <TableCell className="w-8 p-2">
                      {isExpanded ? (
                        <ChevronDown className="h-4 w-4" />
                      ) : (
                        <ChevronRight className="h-4 w-4" />
                      )}
                    </TableCell>
                    <TableCell>
                      <code className="text-sm bg-muted px-1.5 py-0.5 rounded">
                        {group.identifier}
                      </code>
                    </TableCell>
                    <TableCell>
                      <Badge variant="secondary">{group.provider_id}</Badge>
                    </TableCell>
                    <TableCell className="text-muted-foreground">
                      {(group.description as string) || "-"}
                    </TableCell>
                  </TableRow>

                  {isExpanded && (
                    <TableRow>
                      <TableCell colSpan={4} className="p-0">
                        <div className="bg-muted/30 border-t border-b p-4">
                          {groupTools.length === 0 ? (
                            <p className="text-sm text-muted-foreground">
                              No tools found in this group.
                            </p>
                          ) : (
                            <Table>
                              <TableHeader>
                                <TableRow>
                                  <TableHead>Tool Name</TableHead>
                                  <TableHead>Description</TableHead>
                                </TableRow>
                              </TableHeader>
                              <TableBody>
                                {groupTools.map(tool => (
                                  <TableRow key={tool.identifier}>
                                    <TableCell>
                                      <code className="text-sm">
                                        {tool.identifier ||
                                          tool.tool_name ||
                                          "-"}
                                      </code>
                                    </TableCell>
                                    <TableCell className="text-muted-foreground">
                                      {tool.description || "-"}
                                    </TableCell>
                                  </TableRow>
                                ))}
                              </TableBody>
                            </Table>
                          )}
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

  const renderToolsTab = () => {
    if (toolsStatus === "loading") {
      return (
        <div className="space-y-2">
          <Skeleton className="h-8 w-full" />
          <Skeleton className="h-4 w-full" />
          <Skeleton className="h-4 w-full" />
        </div>
      );
    }

    if (toolsStatus === "error") {
      return <div className="text-destructive">Error: {toolsError}</div>;
    }

    if (tools.length === 0) {
      return (
        <div className="text-center py-8">
          <p className="text-muted-foreground">No tools found.</p>
        </div>
      );
    }

    return (
      <div className="space-y-4">
        <div className="relative flex-1 max-w-md">
          <Search className="absolute left-3 top-1/2 transform -translate-y-1/2 text-muted-foreground h-4 w-4" />
          <Input
            placeholder="Search tools..."
            value={searchTerm}
            onChange={e => setSearchTerm(e.target.value)}
            className="pl-10"
          />
        </div>

        <div className="overflow-auto flex-1 min-h-0">
          <Table>
            <TableHeader>
              <TableRow>
                <TableHead className="w-8"></TableHead>
                <TableHead>Tool Name</TableHead>
                <TableHead>Description</TableHead>
                <TableHead>Toolgroup ID</TableHead>
              </TableRow>
            </TableHeader>
            <TableBody>
              {filteredTools.map(tool => {
                const toolId = tool.identifier || tool.tool_name || "";
                const isExpanded = expandedTools.has(toolId);
                const hasParams = tool.parameters && tool.parameters.length > 0;

                return (
                  <React.Fragment key={toolId}>
                    <Collapsible
                      open={isExpanded}
                      onOpenChange={() => toggleTool(toolId)}
                      asChild
                    >
                      <>
                        <CollapsibleTrigger asChild>
                          <TableRow
                            className={
                              hasParams
                                ? "cursor-pointer hover:bg-muted/50"
                                : "hover:bg-muted/50"
                            }
                          >
                            <TableCell className="w-8 p-2">
                              {hasParams &&
                                (isExpanded ? (
                                  <ChevronDown className="h-4 w-4" />
                                ) : (
                                  <ChevronRight className="h-4 w-4" />
                                ))}
                            </TableCell>
                            <TableCell>
                              <code className="text-sm bg-muted px-1.5 py-0.5 rounded">
                                {tool.identifier || tool.tool_name || "-"}
                              </code>
                            </TableCell>
                            <TableCell className="text-muted-foreground max-w-md truncate">
                              {tool.description || "-"}
                            </TableCell>
                            <TableCell>
                              {tool.toolgroup_id ? (
                                <Badge variant="outline">
                                  {tool.toolgroup_id}
                                </Badge>
                              ) : (
                                "-"
                              )}
                            </TableCell>
                          </TableRow>
                        </CollapsibleTrigger>

                        {hasParams && (
                          <CollapsibleContent asChild>
                            <TableRow>
                              <TableCell colSpan={4} className="p-0">
                                <div className="bg-muted/30 border-t border-b p-4">
                                  <h4 className="text-sm font-medium mb-2">
                                    Parameters
                                  </h4>
                                  <div className="rounded border bg-background">
                                    <Table>
                                      <TableHeader>
                                        <TableRow>
                                          <TableHead>Name</TableHead>
                                          <TableHead>Type</TableHead>
                                          <TableHead>Required</TableHead>
                                          <TableHead>Description</TableHead>
                                        </TableRow>
                                      </TableHeader>
                                      <TableBody>
                                        {tool.parameters!.map((param, idx) => (
                                          <TableRow key={idx}>
                                            <TableCell>
                                              <code className="text-sm">
                                                {param.name}
                                              </code>
                                            </TableCell>
                                            <TableCell>
                                              <Badge
                                                variant="secondary"
                                                className="font-mono text-xs"
                                              >
                                                {param.parameter_type}
                                              </Badge>
                                            </TableCell>
                                            <TableCell>
                                              {param.required ? (
                                                <Badge variant="default">
                                                  required
                                                </Badge>
                                              ) : (
                                                <span className="text-muted-foreground text-sm">
                                                  optional
                                                </span>
                                              )}
                                            </TableCell>
                                            <TableCell className="text-muted-foreground text-sm">
                                              {param.description || "-"}
                                            </TableCell>
                                          </TableRow>
                                        ))}
                                      </TableBody>
                                    </Table>
                                  </div>
                                </div>
                              </TableCell>
                            </TableRow>
                          </CollapsibleContent>
                        )}
                      </>
                    </Collapsible>
                  </React.Fragment>
                );
              })}
            </TableBody>
          </Table>
        </div>
      </div>
    );
  };

  return (
    <div className="space-y-4">
      <div className="flex items-center justify-between">
        <h1 className="text-2xl font-semibold">Tools & Toolgroups</h1>
      </div>

      <Tabs defaultValue="toolgroups">
        <TabsList>
          <TabsTrigger value="toolgroups">
            Tool Groups
            {toolGroupsStatus === "idle" && (
              <Badge variant="secondary" className="ml-2">
                {toolGroups.length}
              </Badge>
            )}
          </TabsTrigger>
          <TabsTrigger value="tools">
            All Tools
            {toolsStatus === "idle" && (
              <Badge variant="secondary" className="ml-2">
                {tools.length}
              </Badge>
            )}
          </TabsTrigger>
        </TabsList>

        <TabsContent value="toolgroups">{renderToolGroupsTab()}</TabsContent>
        <TabsContent value="tools">{renderToolsTab()}</TabsContent>
      </Tabs>
    </div>
  );
}
