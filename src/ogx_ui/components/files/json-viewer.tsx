"use client";

/* eslint-disable react/no-unescaped-entities */
import React, { useMemo, useState } from "react";
import { Button } from "@/components/ui/button";
import { ChevronDown, ChevronRight } from "lucide-react";

interface JsonViewerProps {
  content: string;
  filename?: string;
}

type JsonValue =
  | string
  | number
  | boolean
  | null
  | JsonValue[]
  | { [key: string]: JsonValue };

interface JsonNode {
  key?: string;
  value: JsonValue;
  type: "object" | "array" | "string" | "number" | "boolean" | "null";
  isRoot?: boolean;
}

export function JsonViewer({ content }: JsonViewerProps) {
  const [expandedKeys, setExpandedKeys] = useState<Set<string>>(
    new Set(["root"])
  );

  const parsedData = useMemo(() => {
    try {
      const parsed = JSON.parse(content.trim());
      return { data: parsed, error: null };
    } catch (error) {
      return {
        data: null,
        error: error instanceof Error ? error.message : "Invalid JSON",
      };
    }
  }, [content]);

  const toggleExpanded = (key: string) => {
    const newExpanded = new Set(expandedKeys);
    if (newExpanded.has(key)) {
      newExpanded.delete(key);
    } else {
      newExpanded.add(key);
    }
    setExpandedKeys(newExpanded);
  };

  const renderValue = (
    node: JsonNode,
    path: string = "root",
    depth: number = 0
  ): React.ReactNode => {
    const isExpanded = expandedKeys.has(path);
    const indent = depth * 20;

    if (node.type === "object" && node.value !== null) {
      const entries = Object.entries(node.value);
      return (
        <div key={path}>
          <div
            className="flex items-center cursor-pointer hover:bg-muted/50 py-1 px-2 rounded"
            style={{ marginLeft: indent }}
            onClick={() => toggleExpanded(path)}
          >
            {isExpanded ? (
              <ChevronDown className="h-4 w-4 mr-2 text-muted-foreground" />
            ) : (
              <ChevronRight className="h-4 w-4 mr-2 text-muted-foreground" />
            )}
            {node.key && (
              <span className="font-medium text-blue-700 dark:text-blue-300 mr-2">
                "{node.key}":
              </span>
            )}
            <span className="text-muted-foreground">
              {`{${entries.length} ${entries.length === 1 ? "property" : "properties"}}`}
            </span>
          </div>
          {isExpanded && (
            <div>
              {entries.map(([key, value]) =>
                renderValue(
                  {
                    key,
                    value,
                    type: getValueType(value),
                  },
                  `${path}.${key}`,
                  depth + 1
                )
              )}
            </div>
          )}
        </div>
      );
    }

    if (node.type === "array") {
      return (
        <div key={path}>
          <div
            className="flex items-center cursor-pointer hover:bg-muted/50 py-1 px-2 rounded"
            style={{ marginLeft: indent }}
            onClick={() => toggleExpanded(path)}
          >
            {isExpanded ? (
              <ChevronDown className="h-4 w-4 mr-2 text-muted-foreground" />
            ) : (
              <ChevronRight className="h-4 w-4 mr-2 text-muted-foreground" />
            )}
            {node.key && (
              <span className="font-medium text-blue-700 dark:text-blue-300 mr-2">
                "{node.key}":
              </span>
            )}
            <span className="text-muted-foreground">
              [{node.value.length} {node.value.length === 1 ? "item" : "items"}]
            </span>
          </div>
          {isExpanded && (
            <div>
              {node.value.map((item: JsonValue, index: number) =>
                renderValue(
                  {
                    key: index.toString(),
                    value: item,
                    type: getValueType(item),
                  },
                  `${path}[${index}]`,
                  depth + 1
                )
              )}
            </div>
          )}
        </div>
      );
    }

    // Primitive values
    return (
      <div
        key={path}
        className="flex items-center py-1 px-2 hover:bg-muted/50 rounded"
        style={{ marginLeft: indent }}
      >
        {node.key && (
          <span className="font-medium text-blue-700 dark:text-blue-300 mr-2">
            "{node.key}":
          </span>
        )}
        <span className={getValueColor(node.type)}>
          {formatPrimitiveValue(node.value, node.type)}
        </span>
      </div>
    );
  };

  const getValueType = (value: JsonValue): JsonNode["type"] => {
    if (value === null) return "null";
    if (Array.isArray(value)) return "array";
    return typeof value as JsonNode["type"];
  };

  const getValueColor = (type: JsonNode["type"]): string => {
    switch (type) {
      case "string":
        return "text-green-700 dark:text-green-300";
      case "number":
        return "text-purple-700 dark:text-purple-300";
      case "boolean":
        return "text-orange-700 dark:text-orange-300";
      case "null":
        return "text-gray-500 dark:text-gray-400";
      default:
        return "text-foreground";
    }
  };

  const formatPrimitiveValue = (
    value: JsonValue,
    type: JsonNode["type"]
  ): string => {
    if (type === "string") return `"${value}"`;
    if (type === "null") return "null";
    return String(value);
  };

  if (parsedData.error) {
    return (
      <div className="space-y-4">
        <div className="p-4 border border-red-200 rounded-lg bg-red-50 dark:bg-red-900/20 dark:border-red-800">
          <h3 className="font-semibold text-red-800 dark:text-red-300 mb-2">
            JSON Parsing Error
          </h3>
          <p className="text-sm text-red-600 dark:text-red-400">
            Failed to parse JSON file. Please check the file format.
          </p>
          <p className="text-xs text-red-500 dark:text-red-400 mt-1 font-mono">
            {parsedData.error}
          </p>
        </div>
        {/* Show raw content as fallback */}
        <div className="border rounded-lg p-4 bg-muted/50">
          <h4 className="font-medium mb-2">Raw Content:</h4>
          <pre className="text-sm font-mono whitespace-pre-wrap overflow-auto max-h-96 text-muted-foreground">
            {content}
          </pre>
        </div>
      </div>
    );
  }

  return (
    <div className="space-y-4">
      {/* JSON Info */}
      <div className="flex items-center justify-between gap-4">
        <div className="text-sm text-muted-foreground">
          JSON Document • {Object.keys(parsedData.data || {}).length} root{" "}
          {Object.keys(parsedData.data || {}).length === 1
            ? "property"
            : "properties"}
        </div>
        <div className="flex gap-2">
          <Button
            variant="outline"
            size="sm"
            onClick={() => setExpandedKeys(new Set(["root"]))}
          >
            Collapse All
          </Button>
          <Button
            variant="outline"
            size="sm"
            onClick={() => {
              // Expand all nodes (this is a simple version - could be more sophisticated)
              const allKeys = new Set<string>();
              const addKeys = (obj: JsonValue, prefix: string = "root") => {
                allKeys.add(prefix);
                if (obj && typeof obj === "object") {
                  Object.keys(obj).forEach(key => {
                    addKeys(obj[key], `${prefix}.${key}`);
                  });
                }
              };
              addKeys(parsedData.data);
              setExpandedKeys(allKeys);
            }}
          >
            Expand All
          </Button>
        </div>
      </div>

      {/* JSON Tree */}
      <div className="border rounded-lg overflow-hidden">
        <div className="max-h-[500px] overflow-auto scrollbar-thin scrollbar-track-muted scrollbar-thumb-muted-foreground p-4 bg-muted/20">
          {renderValue({
            value: parsedData.data,
            type: getValueType(parsedData.data),
            isRoot: true,
          })}
        </div>
      </div>
    </div>
  );
}
