"use client";

import { useState, useEffect } from "react";
import { Button } from "@/components/ui/button";
import { Input } from "@/components/ui/input";
import { Card } from "@/components/ui/card";
import {
  Select,
  SelectContent,
  SelectItem,
  SelectTrigger,
  SelectValue,
} from "@/components/ui/select";
import { useAuthClient } from "@/hooks/use-auth-client";
import type { Model } from "llama-stack-client/resources/models";

interface VectorDBCreatorProps {
  models: Model[];
  onVectorDBCreated?: (vectorDbId: string) => void;
  onCancel?: () => void;
}

interface VectorDBProvider {
  api: string;
  provider_id: string;
  provider_type: string;
}

export function VectorDBCreator({
  models,
  onVectorDBCreated,
  onCancel,
}: VectorDBCreatorProps) {
  const [vectorDbName, setVectorDbName] = useState("");
  const [selectedEmbeddingModel, setSelectedEmbeddingModel] = useState("");
  const [selectedProvider, setSelectedProvider] = useState("faiss");
  const [availableProviders, setAvailableProviders] = useState<
    VectorDBProvider[]
  >([]);
  const [isCreating, setIsCreating] = useState(false);
  const [isLoadingProviders, setIsLoadingProviders] = useState(false);
  const [error, setError] = useState<string | null>(null);
  const client = useAuthClient();

  const embeddingModels = models.filter(
    model => model.model_type === "embedding"
  );

  useEffect(() => {
    const fetchProviders = async () => {
      setIsLoadingProviders(true);
      try {
        const providersResponse = await client.providers.list();

        const vectorIoProviders = providersResponse.filter(
          (provider: VectorDBProvider) => provider.api === "vector_io"
        );

        setAvailableProviders(vectorIoProviders);

        if (vectorIoProviders.length > 0) {
          const faissProvider = vectorIoProviders.find(
            (p: VectorDBProvider) => p.provider_id === "faiss"
          );
          setSelectedProvider(
            faissProvider?.provider_id || vectorIoProviders[0].provider_id
          );
        }
      } catch (err) {
        console.error("Error fetching providers:", err);
        setAvailableProviders([
          {
            api: "vector_io",
            provider_id: "faiss",
            provider_type: "inline::faiss",
          },
        ]);
      } finally {
        setIsLoadingProviders(false);
      }
    };

    fetchProviders();
  }, [client]);

  const handleCreate = async () => {
    if (!vectorDbName.trim() || !selectedEmbeddingModel) {
      setError("Please provide a name and select an embedding model");
      return;
    }

    setIsCreating(true);
    setError(null);

    try {
      const embeddingModel = embeddingModels.find(
        m => m.identifier === selectedEmbeddingModel
      );

      if (!embeddingModel) {
        throw new Error("Selected embedding model not found");
      }

      const embeddingDimension = embeddingModel.metadata
        ?.embedding_dimension as number;

      if (!embeddingDimension) {
        throw new Error("Embedding dimension not available for selected model");
      }

      const vectorDbId = vectorDbName.trim() || `vector_db_${Date.now()}`;

      const response = await client.vectorDBs.register({
        vector_db_id: vectorDbId,
        embedding_model: selectedEmbeddingModel,
        embedding_dimension: embeddingDimension,
        provider_id: selectedProvider,
      });

      onVectorDBCreated?.(response.identifier || vectorDbId);
    } catch (err) {
      console.error("Error creating vector DB:", err);
      setError(
        err instanceof Error ? err.message : "Failed to create vector DB"
      );
    } finally {
      setIsCreating(false);
    }
  };

  return (
    <Card className="p-6 space-y-4">
      <h3 className="text-lg font-semibold">Create Vector Database</h3>

      <div className="space-y-4">
        <div>
          <label className="text-sm font-medium block mb-2">
            Vector DB Name
          </label>
          <Input
            value={vectorDbName}
            onChange={e => setVectorDbName(e.target.value)}
            placeholder="My Vector Database"
          />
        </div>

        <div>
          <label className="text-sm font-medium block mb-2">
            Embedding Model
          </label>
          <Select
            value={selectedEmbeddingModel}
            onValueChange={setSelectedEmbeddingModel}
          >
            <SelectTrigger>
              <SelectValue placeholder="Select Embedding Model" />
            </SelectTrigger>
            <SelectContent>
              {embeddingModels.map(model => (
                <SelectItem key={model.identifier} value={model.identifier}>
                  {model.identifier}
                </SelectItem>
              ))}
            </SelectContent>
          </Select>
          {selectedEmbeddingModel && (
            <p className="text-xs text-muted-foreground mt-1">
              Dimension:{" "}
              {embeddingModels.find(
                m => m.identifier === selectedEmbeddingModel
              )?.metadata?.embedding_dimension || "Unknown"}
            </p>
          )}
        </div>

        <div>
          <label className="text-sm font-medium block mb-2">
            Vector Database Provider
          </label>
          <Select
            value={selectedProvider}
            onValueChange={setSelectedProvider}
            disabled={isLoadingProviders}
          >
            <SelectTrigger>
              <SelectValue
                placeholder={
                  isLoadingProviders
                    ? "Loading providers..."
                    : "Select Provider"
                }
              />
            </SelectTrigger>
            <SelectContent>
              {availableProviders.map(provider => (
                <SelectItem
                  key={provider.provider_id}
                  value={provider.provider_id}
                >
                  {provider.provider_id}
                </SelectItem>
              ))}
            </SelectContent>
          </Select>
          {selectedProvider && (
            <p className="text-xs text-muted-foreground mt-1">
              Selected provider: {selectedProvider}
            </p>
          )}
        </div>

        {error && (
          <div className="text-destructive text-sm bg-destructive/10 p-2 rounded">
            {error}
          </div>
        )}

        <div className="flex gap-2 pt-2">
          <Button
            onClick={handleCreate}
            disabled={
              isCreating || !vectorDbName.trim() || !selectedEmbeddingModel
            }
            className="flex-1"
          >
            {isCreating ? "Creating..." : "Create Vector DB"}
          </Button>
          {onCancel && (
            <Button variant="outline" onClick={onCancel} className="flex-1">
              Cancel
            </Button>
          )}
        </div>
      </div>

      <div className="text-xs text-muted-foreground bg-muted/50 p-3 rounded">
        <strong>Note:</strong> This will create a new vector database that can
        be used with RAG tools. After creation, you&apos;ll be able to upload
        documents and use it for knowledge search in your agent conversations.
      </div>
    </Card>
  );
}
