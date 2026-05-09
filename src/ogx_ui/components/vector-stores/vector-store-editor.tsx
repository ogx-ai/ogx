"use client";

import { useState, useEffect } from "react";
import { Button } from "@/components/ui/button";
import { Input } from "@/components/ui/input";
import { Label } from "@/components/ui/label";
import { Card, CardContent } from "@/components/ui/card";
import {
  Select,
  SelectContent,
  SelectItem,
  SelectTrigger,
  SelectValue,
} from "@/components/ui/select";
import { useAuthClient } from "@/hooks/use-auth-client";
import type { Model } from "llama-stack-client/resources/models";

export interface VectorStoreFormData {
  name: string;
  embedding_model?: string;
  embedding_dimension?: number;
  provider_id?: string;
}

interface VectorStoreEditorProps {
  onSave: (formData: VectorStoreFormData) => Promise<void>;
  onCancel: () => void;
  error?: string | null;
  initialData?: VectorStoreFormData;
  showSuccessState?: boolean;
  isEditing?: boolean;
}

export function VectorStoreEditor({
  onSave,
  onCancel,
  error,
  initialData,
  showSuccessState,
  isEditing = false,
}: VectorStoreEditorProps) {
  const client = useAuthClient();
  const [formData, setFormData] = useState<VectorStoreFormData>(
    initialData || {
      name: "",
      embedding_model: "",
      embedding_dimension: 768,
      provider_id: "",
    }
  );
  const [loading, setLoading] = useState(false);
  const [models, setModels] = useState<Model[]>([]);
  const [modelsLoading, setModelsLoading] = useState(true);
  const [modelsError, setModelsError] = useState<string | null>(null);

  const embeddingModels = models.filter(
    model => model.custom_metadata?.model_type === "embedding"
  );

  useEffect(() => {
    const fetchModels = async () => {
      try {
        setModelsLoading(true);
        setModelsError(null);
        const modelList = await client.models.list();
        setModels(modelList);

        // Set default embedding model if available
        const embeddingModelsList = modelList.filter(model => {
          return model.custom_metadata?.model_type === "embedding";
        });
        if (embeddingModelsList.length > 0 && !formData.embedding_model) {
          setFormData(prev => ({
            ...prev,
            embedding_model: embeddingModelsList[0].id,
          }));
        }
      } catch (err) {
        console.error("Failed to load models:", err);
        setModelsError(
          err instanceof Error ? err.message : "Failed to load models"
        );
      } finally {
        setModelsLoading(false);
      }
    };

    fetchModels();
  }, [client, formData.embedding_model]);

  const handleSubmit = async (e: React.FormEvent) => {
    e.preventDefault();
    setLoading(true);

    try {
      await onSave(formData);
    } finally {
      setLoading(false);
    }
  };

  return (
    <Card>
      <CardContent className="pt-6">
        <form onSubmit={handleSubmit} className="space-y-4">
          <div className="space-y-2">
            <Label htmlFor="name">Name</Label>
            <Input
              id="name"
              value={formData.name}
              onChange={e => setFormData({ ...formData, name: e.target.value })}
              placeholder="Enter vector store name"
              required
            />
          </div>

          <div className="space-y-2">
            <Label htmlFor="embedding_model">Embedding Model (Optional)</Label>
            {modelsLoading ? (
              <div className="text-sm text-muted-foreground">
                Loading models... ({models.length} loaded)
              </div>
            ) : modelsError ? (
              <div className="text-sm text-destructive">
                Error: {modelsError}
              </div>
            ) : embeddingModels.length === 0 ? (
              <div className="text-sm text-muted-foreground">
                No embedding models available ({models.length} total models)
              </div>
            ) : (
              <Select
                value={formData.embedding_model}
                onValueChange={value =>
                  setFormData({ ...formData, embedding_model: value })
                }
              >
                <SelectTrigger>
                  <SelectValue placeholder="Select Embedding Model" />
                </SelectTrigger>
                <SelectContent>
                  {embeddingModels.map(model => (
                    <SelectItem key={model.id} value={model.id}>
                      {model.id}
                    </SelectItem>
                  ))}
                </SelectContent>
              </Select>
            )}
            {formData.embedding_model && (
              <p className="text-xs text-muted-foreground mt-1">
                Dimension:{" "}
                {embeddingModels.find(m => m.id === formData.embedding_model)
                  ?.custom_metadata?.embedding_dimension || "Unknown"}
              </p>
            )}
          </div>

          <div className="space-y-2">
            <Label htmlFor="embedding_dimension">
              Embedding Dimension (Optional)
            </Label>
            <Input
              id="embedding_dimension"
              type="number"
              value={formData.embedding_dimension}
              onChange={e =>
                setFormData({
                  ...formData,
                  embedding_dimension: parseInt(e.target.value) || 768,
                })
              }
              placeholder="768"
            />
          </div>

          <div className="space-y-2">
            <Label htmlFor="provider_id">
              Provider ID {isEditing ? "(Read-only)" : "(Optional)"}
            </Label>
            <Input
              id="provider_id"
              value={formData.provider_id}
              onChange={e =>
                setFormData({ ...formData, provider_id: e.target.value })
              }
              placeholder="e.g., faiss, chroma, sqlite"
              disabled={isEditing}
            />
            {isEditing && (
              <p className="text-xs text-muted-foreground">
                Provider ID cannot be changed after vector store creation
              </p>
            )}
          </div>

          {error && (
            <div
              className={`text-sm p-3 rounded ${
                error.startsWith("✅")
                  ? "text-green-700 bg-green-50 border border-green-200"
                  : "text-destructive bg-destructive/10"
              }`}
            >
              {error}
            </div>
          )}

          <div className="flex gap-2 pt-4">
            {showSuccessState ? (
              <Button type="button" onClick={onCancel}>
                Close
              </Button>
            ) : (
              <>
                <Button type="submit" disabled={loading}>
                  {loading
                    ? initialData
                      ? "Updating..."
                      : "Creating..."
                    : initialData
                      ? "Update Vector Store"
                      : "Create Vector Store"}
                </Button>
                <Button type="button" variant="outline" onClick={onCancel}>
                  Cancel
                </Button>
              </>
            )}
          </div>
        </form>
      </CardContent>
    </Card>
  );
}
