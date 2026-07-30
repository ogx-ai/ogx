"use client";

import { useState, useEffect } from "react";
import { Button } from "@/components/ui/button";
import { Plus } from "lucide-react";
import { PromptList } from "./prompt-list";
import { PromptEditor } from "./prompt-editor";
import { Prompt, PromptFormData } from "./types";
import { useAuthClient } from "@/hooks/use-auth-client";

export function PromptManagement() {
  const [prompts, setPrompts] = useState<Prompt[]>([]);
  const [showPromptModal, setShowPromptModal] = useState(false);
  const [editingPrompt, setEditingPrompt] = useState<Prompt | undefined>();
  const [loading, setLoading] = useState(true);
  const [error, setError] = useState<string | null>(null); // For main page errors (loading, etc.)
  const [modalError, setModalError] = useState<string | null>(null); // For form submission errors
  const client = useAuthClient();

  // Load prompts from API on component mount
  useEffect(() => {
    const fetchPrompts = async () => {
      try {
        setLoading(true);
        setError(null);

        const response = await client.prompts.list();
        setPrompts(response || []);
      } catch (err: unknown) {
        console.error("Failed to load prompts:", err);

        // Handle different types of errors
        const error = err as Error & { status?: number };
        if (error?.message?.includes("404") || error?.status === 404) {
          setError(
            "Prompts API endpoint not found. Please ensure your OGX server supports the prompts API."
          );
        } else if (
          error?.message?.includes("not implemented") ||
          error?.message?.includes("not supported")
        ) {
          setError("Prompts API is not yet implemented on this OGX server.");
        } else {
          setError(
            `Failed to load prompts: ${error?.message || "Unknown error"}`
          );
        }
      } finally {
        setLoading(false);
      }
    };

    fetchPrompts();
  }, [client]);

  const handleSavePrompt = async (formData: PromptFormData) => {
    try {
      setModalError(null);

      if (editingPrompt) {
        // Update existing prompt
        const response = await client.prompts.update(editingPrompt.prompt_id, {
          prompt: formData.prompt,
          variables: formData.variables,
          version: editingPrompt.version,
          set_as_default: true,
        });

        // Update local state
        setPrompts(prev =>
          prev.map(p =>
            p.prompt_id === editingPrompt.prompt_id ? response : p
          )
        );
      } else {
        // Create new prompt
        const response = await client.prompts.create({
          prompt: formData.prompt,
          variables: formData.variables,
        });

        // Add to local state
        setPrompts(prev => [response, ...prev]);
      }

      setShowPromptModal(false);
      setEditingPrompt(undefined);
    } catch (err) {
      console.error("Failed to save prompt:", err);

      // Extract specific error message from API response
      const error = err as Error & {
        message?: string;
        detail?: { errors?: Array<{ msg?: string }> };
      };

      // Try to parse JSON from error message if it's a string
      let parsedError = error;
      if (typeof error?.message === "string" && error.message.includes("{")) {
        try {
          const jsonMatch = error.message.match(/\d+\s+(.+)/);
          if (jsonMatch) {
            parsedError = JSON.parse(jsonMatch[1]);
          }
        } catch {
          // If parsing fails, use original error
        }
      }

      // Try to get the specific validation error message
      const validationError = parsedError?.detail?.errors?.[0]?.msg;
      if (validationError) {
        // Clean up validation error messages (remove "Value error, " prefix if present)
        const cleanMessage = validationError.replace(/^Value error,\s*/i, "");
        setModalError(cleanMessage);
      } else {
        // For other errors, format them nicely with line breaks
        const statusMatch = error?.message?.match(/(\d+)\s+(.+)/);
        if (statusMatch) {
          const statusCode = statusMatch[1];
          const response = statusMatch[2];
          setModalError(
            `Failed to save prompt: Status Code ${statusCode}\n\nResponse: ${response}`
          );
        } else {
          const message = error?.message || error?.detail || "Unknown error";
          setModalError(`Failed to save prompt: ${message}`);
        }
      }
    }
  };

  const handleEditPrompt = (prompt: Prompt) => {
    setEditingPrompt(prompt);
    setShowPromptModal(true);
    setModalError(null); // Clear any previous modal errors
  };

  const handleDeletePrompt = async (promptId: string) => {
    try {
      setError(null);
      await client.prompts.delete(promptId);
      setPrompts(prev => prev.filter(p => p.prompt_id !== promptId));

      // If we're deleting the currently editing prompt, close the modal
      if (editingPrompt && editingPrompt.prompt_id === promptId) {
        setShowPromptModal(false);
        setEditingPrompt(undefined);
      }
    } catch (err) {
      console.error("Failed to delete prompt:", err);
      setError("Failed to delete prompt");
    }
  };

  const handleCreateNew = () => {
    setEditingPrompt(undefined);
    setShowPromptModal(true);
    setModalError(null); // Clear any previous modal errors
  };

  const handleCancel = () => {
    setShowPromptModal(false);
    setEditingPrompt(undefined);
  };

  const renderContent = () => {
    if (loading) {
      return <div className="text-muted-foreground">Loading prompts...</div>;
    }

    if (error) {
      return <div className="text-destructive">Error: {error}</div>;
    }

    if (!prompts || prompts.length === 0) {
      return (
        <div className="text-center py-12">
          <p className="text-muted-foreground mb-4">No prompts found.</p>
          <Button onClick={handleCreateNew}>
            <Plus className="h-4 w-4 mr-2" />
            Create Your First Prompt
          </Button>
        </div>
      );
    }

    return (
      <PromptList
        prompts={prompts}
        onEdit={handleEditPrompt}
        onDelete={handleDeletePrompt}
      />
    );
  };

  return (
    <div className="space-y-4">
      <div className="flex items-center justify-between">
        <h1 className="text-2xl font-semibold">Prompts</h1>
        <Button onClick={handleCreateNew} disabled={loading}>
          <Plus className="h-4 w-4 mr-2" />
          New Prompt
        </Button>
      </div>
      {renderContent()}

      {/* Create/Edit Prompt Modal */}
      {showPromptModal && (
        <div className="fixed inset-0 bg-black/50 flex items-center justify-center z-50">
          <div className="bg-background border rounded-lg shadow-lg max-w-4xl w-full mx-4 max-h-[90vh] overflow-hidden">
            <div className="p-6 border-b">
              <h2 className="text-2xl font-bold">
                {editingPrompt ? "Edit Prompt" : "Create New Prompt"}
              </h2>
            </div>
            <div className="p-6 overflow-y-auto max-h-[calc(90vh-120px)]">
              <PromptEditor
                prompt={editingPrompt}
                onSave={handleSavePrompt}
                onCancel={handleCancel}
                onDelete={handleDeletePrompt}
                error={modalError}
              />
            </div>
          </div>
        </div>
      )}
    </div>
  );
}
