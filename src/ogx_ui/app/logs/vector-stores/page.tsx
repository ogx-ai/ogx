"use client";

import React from "react";
import type {
  ListVectorStoresResponse,
  VectorStore,
} from "llama-stack-client/resources/vector-stores/vector-stores";
import { useRouter } from "next/navigation";
import { usePagination } from "@/hooks/use-pagination";
import { Button } from "@/components/ui/button";
import { Plus, Trash2, Search, Edit, X } from "lucide-react";
import { useState } from "react";
import { Input } from "@/components/ui/input";
import {
  Table,
  TableBody,
  TableCell,
  TableHead,
  TableHeader,
  TableRow,
} from "@/components/ui/table";
import { Skeleton } from "@/components/ui/skeleton";
import { useAuthClient } from "@/hooks/use-auth-client";
import {
  VectorStoreEditor,
  VectorStoreFormData,
} from "@/components/vector-stores/vector-store-editor";

export default function VectorStoresPage() {
  const router = useRouter();
  const client = useAuthClient();
  const [deletingStores, setDeletingStores] = useState<Set<string>>(new Set());
  const [searchTerm, setSearchTerm] = useState("");
  const [showVectorStoreModal, setShowVectorStoreModal] = useState(false);
  const [editingStore, setEditingStore] = useState<VectorStore | null>(null);
  const [modalError, setModalError] = useState<string | null>(null);
  const [showSuccessState, setShowSuccessState] = useState(false);
  const {
    data: stores,
    status,
    hasMore,
    error,
    loadMore,
  } = usePagination<VectorStore>({
    limit: 20,
    order: "desc",
    fetchFunction: async (client, params) => {
      const response = await client.vectorStores.list({
        after: params.after,
        limit: params.limit,
        order: params.order,
      } as Parameters<typeof client.vectorStores.list>[0]);
      return response as ListVectorStoresResponse;
    },
    errorMessagePrefix: "vector stores",
  });

  // Auto-load all pages for infinite scroll behavior (like Responses)
  React.useEffect(() => {
    if (status === "idle" && hasMore) {
      loadMore();
    }
  }, [status, hasMore, loadMore]);

  // Handle ESC key to close modal
  React.useEffect(() => {
    const handleEscape = (event: KeyboardEvent) => {
      if (event.key === "Escape" && showVectorStoreModal) {
        handleCancel();
      }
    };

    document.addEventListener("keydown", handleEscape);
    return () => document.removeEventListener("keydown", handleEscape);
  }, [showVectorStoreModal]);

  const handleDeleteVectorStore = async (storeId: string) => {
    if (
      !confirm(
        "Are you sure you want to delete this vector store? This action cannot be undone."
      )
    ) {
      return;
    }

    setDeletingStores(prev => new Set([...prev, storeId]));

    try {
      await client.vectorStores.delete(storeId);
      // Reload the data to reflect the deletion
      window.location.reload();
    } catch (err: unknown) {
      console.error("Failed to delete vector store:", err);
      const errorMessage = err instanceof Error ? err.message : "Unknown error";
      alert(`Failed to delete vector store: ${errorMessage}`);
    } finally {
      setDeletingStores(prev => {
        const newSet = new Set(prev);
        newSet.delete(storeId);
        return newSet;
      });
    }
  };

  const handleSaveVectorStore = async (formData: VectorStoreFormData) => {
    try {
      setModalError(null);

      if (editingStore) {
        // Update existing vector store
        const updateParams: {
          name?: string;
          extra_body?: Record<string, unknown>;
        } = {};

        // Only include fields that have changed or are provided
        if (formData.name && formData.name !== editingStore.name) {
          updateParams.name = formData.name;
        }

        // Add all parameters to extra_body (except provider_id which can't be changed)
        const extraBody: Record<string, unknown> = {};
        if (formData.embedding_model) {
          extraBody.embedding_model = formData.embedding_model;
        }
        if (formData.embedding_dimension) {
          extraBody.embedding_dimension = formData.embedding_dimension;
        }

        if (Object.keys(extraBody).length > 0) {
          updateParams.extra_body = extraBody;
        }

        await client.vectorStores.update(editingStore.id, updateParams);

        // Show success state with close button
        setShowSuccessState(true);
        setModalError(
          "✅ Vector store updated successfully! You can close this modal and refresh the page to see changes."
        );
        return;
      }

      const createParams: {
        name?: string;
        provider_id?: string;
        extra_body?: Record<string, unknown>;
      } = {
        name: formData.name || undefined,
      };

      // Extract provider_id to top-level (like Python client does)
      if (formData.provider_id) {
        createParams.provider_id = formData.provider_id;
      }

      // Add remaining parameters to extra_body
      const extraBody: Record<string, unknown> = {};
      if (formData.provider_id) {
        extraBody.provider_id = formData.provider_id;
      }
      if (formData.embedding_model) {
        extraBody.embedding_model = formData.embedding_model;
      }
      if (formData.embedding_dimension) {
        extraBody.embedding_dimension = formData.embedding_dimension;
      }

      if (Object.keys(extraBody).length > 0) {
        createParams.extra_body = extraBody;
      }

      await client.vectorStores.create(createParams);

      // Show success state with close button
      setShowSuccessState(true);
      setModalError(
        "✅ Vector store created successfully! You can close this modal and refresh the page to see changes."
      );
    } catch (err: unknown) {
      console.error("Failed to create vector store:", err);
      const errorMessage =
        err instanceof Error ? err.message : "Failed to create vector store";
      setModalError(errorMessage);
    }
  };

  const handleEditVectorStore = (store: VectorStore) => {
    setEditingStore(store);
    setShowVectorStoreModal(true);
    setModalError(null);
  };

  const handleCancel = () => {
    setShowVectorStoreModal(false);
    setEditingStore(null);
    setModalError(null);
    setShowSuccessState(false);
  };

  const renderContent = () => {
    if (status === "loading") {
      return (
        <div className="space-y-2">
          <Skeleton className="h-8 w-full" />
          <Skeleton className="h-4 w-full" />
          <Skeleton className="h-4 w-full" />
        </div>
      );
    }

    if (status === "error") {
      return <div className="text-destructive">Error: {error?.message}</div>;
    }

    if (!stores || stores.length === 0) {
      return <p>No vector stores found.</p>;
    }

    // Filter stores based on search term
    const filteredStores = stores.filter(store => {
      if (!searchTerm) return true;

      const searchLower = searchTerm.toLowerCase();
      return (
        store.id.toLowerCase().includes(searchLower) ||
        (store.name && store.name.toLowerCase().includes(searchLower)) ||
        (store.metadata?.provider_id &&
          String(store.metadata.provider_id)
            .toLowerCase()
            .includes(searchLower)) ||
        (store.metadata?.provider_vector_db_id &&
          String(store.metadata.provider_vector_db_id)
            .toLowerCase()
            .includes(searchLower))
      );
    });

    return (
      <div className="space-y-4">
        {/* Search Bar */}
        <div className="relative flex-1 max-w-md">
          <Search className="absolute left-3 top-1/2 transform -translate-y-1/2 text-muted-foreground h-4 w-4" />
          <Input
            placeholder="Search vector stores..."
            value={searchTerm}
            onChange={e => setSearchTerm(e.target.value)}
            className="pl-10"
          />
        </div>

        <div className="overflow-auto flex-1 min-h-0">
          <Table>
            <TableHeader>
              <TableRow>
                <TableHead>ID</TableHead>
                <TableHead>Name</TableHead>
                <TableHead>Created</TableHead>
                <TableHead>Completed</TableHead>
                <TableHead>Cancelled</TableHead>
                <TableHead>Failed</TableHead>
                <TableHead>In Progress</TableHead>
                <TableHead>Total</TableHead>
                <TableHead>Usage Bytes</TableHead>
                <TableHead>Provider ID</TableHead>
                <TableHead>Provider Vector DB ID</TableHead>
                <TableHead>Actions</TableHead>
              </TableRow>
            </TableHeader>
            <TableBody>
              {filteredStores.map(store => {
                const fileCounts = store.file_counts;
                const metadata = store.metadata || {};
                const providerId = metadata.provider_id ?? "";
                const providerDbId = metadata.provider_vector_db_id ?? "";

                return (
                  <TableRow
                    key={store.id}
                    onClick={() =>
                      router.push(`/logs/vector-stores/${store.id}`)
                    }
                    className="cursor-pointer hover:bg-muted/50"
                  >
                    <TableCell>
                      <Button
                        variant="link"
                        className="p-0 h-auto font-mono text-blue-600 hover:text-blue-800 dark:text-blue-400 dark:hover:text-blue-300"
                        onClick={() =>
                          router.push(`/logs/vector-stores/${store.id}`)
                        }
                      >
                        {store.id}
                      </Button>
                    </TableCell>
                    <TableCell>{store.name}</TableCell>
                    <TableCell>
                      {new Date(store.created_at * 1000).toLocaleString()}
                    </TableCell>
                    <TableCell>{fileCounts.completed}</TableCell>
                    <TableCell>{fileCounts.cancelled}</TableCell>
                    <TableCell>{fileCounts.failed}</TableCell>
                    <TableCell>{fileCounts.in_progress}</TableCell>
                    <TableCell>{fileCounts.total}</TableCell>
                    <TableCell>{store.usage_bytes}</TableCell>
                    <TableCell>{providerId}</TableCell>
                    <TableCell>{providerDbId}</TableCell>
                    <TableCell>
                      <div className="flex gap-2">
                        <Button
                          variant="outline"
                          size="sm"
                          onClick={e => {
                            e.stopPropagation();
                            handleEditVectorStore(store);
                          }}
                        >
                          <Edit className="h-4 w-4" />
                        </Button>
                        <Button
                          variant="outline"
                          size="sm"
                          onClick={e => {
                            e.stopPropagation();
                            handleDeleteVectorStore(store.id);
                          }}
                          disabled={deletingStores.has(store.id)}
                        >
                          {deletingStores.has(store.id) ? (
                            "Deleting..."
                          ) : (
                            <>
                              <Trash2 className="h-4 w-4" />
                            </>
                          )}
                        </Button>
                      </div>
                    </TableCell>
                  </TableRow>
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
        <h1 className="text-2xl font-semibold">Vector Stores</h1>
        <Button
          onClick={() => setShowVectorStoreModal(true)}
          disabled={status === "loading"}
        >
          <Plus className="h-4 w-4 mr-2" />
          New Vector Store
        </Button>
      </div>
      {renderContent()}

      {/* Create Vector Store Modal */}
      {showVectorStoreModal && (
        <div className="fixed inset-0 bg-black/50 flex items-center justify-center z-50">
          <div className="bg-background border rounded-lg shadow-lg max-w-2xl w-full mx-4 max-h-[90vh] overflow-hidden">
            <div className="p-6 border-b flex items-center justify-between">
              <h2 className="text-2xl font-bold">
                {editingStore ? "Edit Vector Store" : "Create New Vector Store"}
              </h2>
              <Button
                variant="ghost"
                size="sm"
                onClick={handleCancel}
                className="p-1 h-auto"
              >
                <X className="h-4 w-4" />
              </Button>
            </div>
            <div className="p-6 overflow-y-auto max-h-[calc(90vh-120px)]">
              <VectorStoreEditor
                onSave={handleSaveVectorStore}
                onCancel={handleCancel}
                error={modalError}
                showSuccessState={showSuccessState}
                isEditing={!!editingStore}
                initialData={
                  editingStore
                    ? {
                        name: editingStore.name || "",
                        embedding_model:
                          editingStore.metadata?.embedding_model || "",
                        embedding_dimension:
                          editingStore.metadata?.embedding_dimension || 768,
                        provider_id: editingStore.metadata?.provider_id || "",
                      }
                    : undefined
                }
              />
            </div>
          </div>
        </div>
      )}
    </div>
  );
}
