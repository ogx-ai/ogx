"use client";

import React, { useState, useEffect, useCallback } from "react";
import { useAuthClient } from "@/hooks/use-auth-client";
import { Button } from "@/components/ui/button";
import { Textarea } from "@/components/ui/textarea";
import { Card } from "@/components/ui/card";
import { Badge } from "@/components/ui/badge";
import { Skeleton } from "@/components/ui/skeleton";
import {
  Select,
  SelectContent,
  SelectItem,
  SelectTrigger,
  SelectValue,
} from "@/components/ui/select";
import { ShieldCheck, ShieldAlert, ChevronDown, ChevronUp } from "lucide-react";
import type { Model } from "llama-stack-client/resources/models";

type ModelWithMetadata = Model & {
  id: string;
  custom_metadata?: {
    model_type?: string;
    [key: string]: unknown;
  };
};

interface ModerationCategory {
  [key: string]: boolean | number | string;
}

interface ModerationResult {
  flagged: boolean;
  categories?: ModerationCategory;
  category_scores?: ModerationCategory;
  [key: string]: unknown;
}

interface ModerationResponse {
  id?: string;
  model?: string;
  results?: ModerationResult[];
  [key: string]: unknown;
}

export function SafetyPlayground() {
  const client = useAuthClient();
  const [models, setModels] = useState<ModelWithMetadata[]>([]);
  const [selectedModel, setSelectedModel] = useState<string>("");
  const [modelsLoading, setModelsLoading] = useState(true);
  const [modelsError, setModelsError] = useState<string | null>(null);
  const [inputText, setInputText] = useState("");
  const [checking, setChecking] = useState(false);
  const [result, setResult] = useState<ModerationResponse | null>(null);
  const [error, setError] = useState<string | null>(null);
  const [showRawResponse, setShowRawResponse] = useState(false);

  const fetchModels = useCallback(async () => {
    try {
      setModelsLoading(true);
      setModelsError(null);
      const modelList = await client.models.list();

      const moderationModels = modelList.filter(
        (model): model is ModelWithMetadata => {
          const m = model as ModelWithMetadata;
          const modelType = m.custom_metadata?.model_type?.toLowerCase() || "";
          const modelId = m.id?.toLowerCase() || "";
          return (
            modelType.includes("moderation") ||
            modelType.includes("guard") ||
            modelId.includes("guard") ||
            modelId.includes("moderation") ||
            modelId.includes("safety")
          );
        }
      );

      setModels(moderationModels);
      if (moderationModels.length > 0) {
        setSelectedModel(moderationModels[0].id);
      }
    } catch (err) {
      console.error("Failed to fetch models:", err);
      setModelsError("Failed to fetch available models");
    } finally {
      setModelsLoading(false);
    }
  }, [client]);

  useEffect(() => {
    fetchModels();
  }, [fetchModels]);

  const handleCheck = async () => {
    if (!inputText.trim() || !selectedModel) return;

    setChecking(true);
    setError(null);
    setResult(null);

    try {
      const response = await fetch("/api/v1/moderations", {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({
          model: selectedModel,
          input: inputText.trim(),
        }),
      });

      if (!response.ok) {
        const errorData = await response.text();
        throw new Error(
          `Failed to run moderation check (${response.status}): ${errorData}`
        );
      }

      const data: ModerationResponse = await response.json();
      setResult(data);
    } catch (err) {
      console.error("Failed to run moderation check:", err);
      setError(
        err instanceof Error ? err.message : "Failed to run moderation check"
      );
    } finally {
      setChecking(false);
    }
  };

  const renderResult = () => {
    if (!result) return null;

    const results = result.results || [];
    if (results.length === 0) {
      return (
        <Card className="p-6">
          <p className="text-muted-foreground">
            No moderation results returned. Check the raw response for details.
          </p>
          <button
            onClick={() => setShowRawResponse(!showRawResponse)}
            className="mt-2 text-sm text-blue-600 hover:text-blue-800 dark:text-blue-400 dark:hover:text-blue-300 flex items-center gap-1"
          >
            {showRawResponse ? (
              <ChevronUp className="h-3 w-3" />
            ) : (
              <ChevronDown className="h-3 w-3" />
            )}
            {showRawResponse ? "Hide" : "Show"} raw response
          </button>
          {showRawResponse && (
            <pre className="mt-2 p-3 bg-muted rounded-md text-xs overflow-auto max-h-64">
              {JSON.stringify(result, null, 2)}
            </pre>
          )}
        </Card>
      );
    }

    return (
      <div className="space-y-4">
        {results.map((moderationResult, index) => {
          const isFlagged = moderationResult.flagged;

          return (
            <Card key={index} className="p-6">
              <div className="flex items-center gap-3 mb-4">
                {isFlagged ? (
                  <ShieldAlert className="h-6 w-6 text-destructive" />
                ) : (
                  <ShieldCheck className="h-6 w-6 text-green-600 dark:text-green-400" />
                )}
                <Badge
                  variant={isFlagged ? "destructive" : "default"}
                  className={
                    !isFlagged
                      ? "bg-green-100 text-green-800 dark:bg-green-900 dark:text-green-200 hover:bg-green-100 dark:hover:bg-green-900"
                      : ""
                  }
                >
                  {isFlagged ? "Flagged" : "Safe"}
                </Badge>
                {results.length > 1 && (
                  <span className="text-sm text-muted-foreground">
                    Result {index + 1} of {results.length}
                  </span>
                )}
              </div>

              {moderationResult.categories &&
                Object.keys(moderationResult.categories).length > 0 && (
                  <div className="mb-4">
                    <h4 className="text-sm font-medium mb-2">Categories</h4>
                    <div className="flex flex-wrap gap-2">
                      {Object.entries(moderationResult.categories).map(
                        ([category, value]) => {
                          const isTriggered = Boolean(value);
                          return (
                            <Badge
                              key={category}
                              variant={isTriggered ? "destructive" : "outline"}
                            >
                              {category}
                            </Badge>
                          );
                        }
                      )}
                    </div>
                  </div>
                )}

              {moderationResult.category_scores &&
                Object.keys(moderationResult.category_scores).length > 0 && (
                  <div className="mb-4">
                    <h4 className="text-sm font-medium mb-2">
                      Category Scores
                    </h4>
                    <div className="space-y-2">
                      {Object.entries(moderationResult.category_scores).map(
                        ([category, score]) => {
                          const numericScore =
                            typeof score === "number" ? score : 0;
                          const percentage = Math.round(numericScore * 100);
                          return (
                            <div key={category} className="space-y-1">
                              <div className="flex justify-between text-sm">
                                <span className="text-muted-foreground">
                                  {category}
                                </span>
                                <span className="font-mono">
                                  {numericScore.toFixed(4)}
                                </span>
                              </div>
                              <div className="w-full bg-muted rounded-full h-2">
                                <div
                                  className={`h-2 rounded-full transition-all ${
                                    numericScore > 0.5
                                      ? "bg-destructive"
                                      : numericScore > 0.2
                                        ? "bg-yellow-500"
                                        : "bg-green-500"
                                  }`}
                                  style={{
                                    width: `${Math.max(percentage, 1)}%`,
                                  }}
                                />
                              </div>
                            </div>
                          );
                        }
                      )}
                    </div>
                  </div>
                )}

              <button
                onClick={() => setShowRawResponse(!showRawResponse)}
                className="text-sm text-blue-600 hover:text-blue-800 dark:text-blue-400 dark:hover:text-blue-300 flex items-center gap-1"
              >
                {showRawResponse ? (
                  <ChevronUp className="h-3 w-3" />
                ) : (
                  <ChevronDown className="h-3 w-3" />
                )}
                {showRawResponse ? "Hide" : "Show"} raw response
              </button>
              {showRawResponse && (
                <pre className="mt-2 p-3 bg-muted rounded-md text-xs overflow-auto max-h-64">
                  {JSON.stringify(result, null, 2)}
                </pre>
              )}
            </Card>
          );
        })}
      </div>
    );
  };

  return (
    <div className="space-y-6">
      <div>
        <h1 className="text-2xl font-semibold">Safety</h1>
        <p className="text-muted-foreground mt-1">
          Test content moderation by checking text against a safety model.
        </p>
      </div>

      <Card className="p-6 space-y-4">
        <div>
          <label className="text-sm font-medium block mb-2">Model</label>
          {modelsLoading ? (
            <Skeleton className="h-10 w-full" />
          ) : modelsError ? (
            <div className="text-destructive text-sm">{modelsError}</div>
          ) : models.length === 0 ? (
            <div className="text-muted-foreground text-sm">
              No moderation models found. Register a model with a type
              containing &quot;moderation&quot; or &quot;guard&quot; to use this
              feature.
            </div>
          ) : (
            <Select value={selectedModel} onValueChange={setSelectedModel}>
              <SelectTrigger className="w-full">
                <SelectValue placeholder="Select a moderation model" />
              </SelectTrigger>
              <SelectContent>
                {models.map(model => (
                  <SelectItem key={model.id} value={model.id}>
                    {model.id}
                  </SelectItem>
                ))}
              </SelectContent>
            </Select>
          )}
        </div>

        <div>
          <label className="text-sm font-medium block mb-2">
            Content to check
          </label>
          <Textarea
            value={inputText}
            onChange={e => setInputText(e.target.value)}
            placeholder="Enter or paste text content to check for safety..."
            className="min-h-[120px] resize-y"
          />
        </div>

        <Button
          onClick={handleCheck}
          disabled={checking || !inputText.trim() || !selectedModel}
        >
          {checking ? "Checking..." : "Check Content"}
        </Button>
      </Card>

      {error && (
        <Card className="p-4 border-destructive">
          <p className="text-destructive text-sm">{error}</p>
        </Card>
      )}

      {checking && (
        <div className="space-y-2">
          <Skeleton className="h-8 w-full" />
          <Skeleton className="h-4 w-3/4" />
          <Skeleton className="h-4 w-1/2" />
        </div>
      )}

      {result && !checking && renderResult()}
    </div>
  );
}
