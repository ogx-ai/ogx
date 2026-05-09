"use client";

import { useRouter } from "next/navigation";
import { useState, useEffect } from "react";
import type { VectorStore } from "llama-stack-client/resources/vector-stores/vector-stores";
import type { VectorStoreFile } from "llama-stack-client/resources/vector-stores/files";
import { Card, CardContent, CardHeader, CardTitle } from "@/components/ui/card";
import { Skeleton } from "@/components/ui/skeleton";
import { Button } from "@/components/ui/button";
import { useAuthClient } from "@/hooks/use-auth-client";
import { Edit2, Trash2, X } from "lucide-react";
import {
  DetailLoadingView,
  DetailErrorView,
  DetailNotFoundView,
  PropertiesCard,
  PropertyItem,
} from "@/components/layout/detail-layout";
import {
  Table,
  TableBody,
  TableCaption,
  TableCell,
  TableHead,
  TableHeader,
  TableRow,
} from "@/components/ui/table";
import { VectorStoreEditor, VectorStoreFormData } from "./vector-store-editor";

interface VectorStoreDetailViewProps {
  store: VectorStore | null;
  files: VectorStoreFile[];
  isLoadingStore: boolean;
  isLoadingFiles: boolean;
  errorStore: Error | null;
  errorFiles: Error | null;
  id: string;
}

export function VectorStoreDetailView({
  store,
  files,
  isLoadingStore,
  isLoadingFiles,
  errorStore,
  errorFiles,
  id,
}: VectorStoreDetailViewProps) {
  const router = useRouter();
  const client = useAuthClient();
  const [isDeleting, setIsDeleting] = useState(false);
  const [showEditModal, setShowEditModal] = useState(false);
  const [modalError, setModalError] = useState<string | null>(null);
  const [showSuccessState, setShowSuccessState] = useState(false);

  // Handle ESC key to close modal
  useEffect(() => {
    const handleEscape = (event: KeyboardEvent) => {
      if (event.key === "Escape" && showEditModal) {
        handleCancel();
      }
    };

    document.addEventListener("keydown", handleEscape);
    return () => document.removeEventListener("keydown", handleEscape);
  }, [showEditModal]);

  const handleFileClick = (fileId: string) => {
    router.push(`/logs/vector-stores/${id}/files/${fileId}`);
  };

  const handleEditVectorStore = () => {
    setShowEditModal(true);
    setModalError(null);
    setShowSuccessState(false);
  };

  const handleCancel = () => {
    setShowEditModal(false);
    setModalError(null);
    setShowSuccessState(false);
  };

  const handleSaveVectorStore = async (formData: VectorStoreFormData) => {
    try {
      setModalError(null);

      // Update existing vector store (same logic as list page)
      const updateParams: {
        name?: string;
        extra_body?: Record<string, unknown>;
      } = {};

      // Only include fields that have changed or are provided
      if (formData.name && formData.name !== store?.name) {
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

      await client.vectorStores.update(id, updateParams);

      // Show success state
      setShowSuccessState(true);
      setModalError(
        "✅ Vector store updated successfully! You can close this modal and refresh the page to see changes."
      );
    } catch (err: unknown) {
      console.error("Failed to update vector store:", err);
      const errorMessage =
        err instanceof Error ? err.message : "Failed to update vector store";
      setModalError(errorMessage);
    }
  };

  const handleDeleteVectorStore = async () => {
    if (
      !confirm(
        "Are you sure you want to delete this vector store? This action cannot be undone."
      )
    ) {
      return;
    }

    setIsDeleting(true);

    try {
      await client.vectorStores.delete(id);
      // Redirect to the vector stores list after successful deletion
      router.push("/logs/vector-stores");
    } catch (err: unknown) {
      console.error("Failed to delete vector store:", err);
      const errorMessage = err instanceof Error ? err.message : "Unknown error";
      alert(`Failed to delete vector store: ${errorMessage}`);
    } finally {
      setIsDeleting(false);
    }
  };

  if (errorStore) {
    return (
      <DetailErrorView
        title="Vector Store Details"
        id={id}
        error={errorStore}
      />
    );
  }
  if (isLoadingStore) {
    return <DetailLoadingView />;
  }
  if (!store) {
    return <DetailNotFoundView title="Vector Store Details" id={id} />;
  }

  const mainContent = (
    <>
      <Card>
        <CardHeader>
          <CardTitle>Files</CardTitle>
        </CardHeader>
        <CardContent>
          {isLoadingFiles ? (
            <Skeleton className="h-4 w-full" />
          ) : errorFiles ? (
            <div className="text-destructive text-sm">
              Error loading files: {errorFiles.message}
            </div>
          ) : files.length > 0 ? (
            <Table>
              <TableCaption>Files in this vector store</TableCaption>
              <TableHeader>
                <TableRow>
                  <TableHead>ID</TableHead>
                  <TableHead>Status</TableHead>
                  <TableHead>Created</TableHead>
                  <TableHead>Usage Bytes</TableHead>
                </TableRow>
              </TableHeader>
              <TableBody>
                {files.map(file => (
                  <TableRow key={file.id}>
                    <TableCell>
                      <Button
                        variant="link"
                        className="p-0 h-auto font-mono text-blue-600 hover:text-blue-800 dark:text-blue-400 dark:hover:text-blue-300"
                        onClick={() => handleFileClick(file.id)}
                      >
                        {file.id}
                      </Button>
                    </TableCell>
                    <TableCell>{file.status}</TableCell>
                    <TableCell>
                      {new Date(file.created_at * 1000).toLocaleString()}
                    </TableCell>
                    <TableCell>{file.usage_bytes}</TableCell>
                  </TableRow>
                ))}
              </TableBody>
            </Table>
          ) : (
            <p className="text-gray-500 italic text-sm">
              No files in this vector store.
            </p>
          )}
        </CardContent>
      </Card>
    </>
  );

  const sidebar = (
    <PropertiesCard>
      <PropertyItem label="ID" value={store.id} />
      <PropertyItem label="Name" value={store.name || ""} />
      <PropertyItem
        label="Created"
        value={new Date(store.created_at * 1000).toLocaleString()}
      />
      <PropertyItem label="Status" value={store.status} />
      <PropertyItem label="Total Files" value={store.file_counts.total} />
      <PropertyItem label="Usage Bytes" value={store.usage_bytes} />
      <PropertyItem
        label="Provider ID"
        value={(store.metadata.provider_id as string) || ""}
      />
      <PropertyItem
        label="Provider DB ID"
        value={(store.metadata.provider_vector_db_id as string) || ""}
      />
    </PropertiesCard>
  );

  return (
    <>
      <div className="flex items-center justify-between mb-6">
        <h1 className="text-2xl font-bold">Vector Store Details</h1>
        <div className="flex gap-2">
          <Button
            variant="outline"
            onClick={handleEditVectorStore}
            disabled={isDeleting}
          >
            <Edit2 className="h-4 w-4 mr-2" />
            Edit
          </Button>
          <Button
            variant="destructive"
            onClick={handleDeleteVectorStore}
            disabled={isDeleting}
          >
            {isDeleting ? (
              "Deleting..."
            ) : (
              <>
                <Trash2 className="h-4 w-4 mr-2" />
                Delete
              </>
            )}
          </Button>
        </div>
      </div>
      <div className="flex flex-col md:flex-row gap-6">
        <div className="flex-grow md:w-2/3 space-y-6">{mainContent}</div>
        <div className="md:w-1/3">{sidebar}</div>
      </div>

      {/* Edit Vector Store Modal */}
      {showEditModal && (
        <div className="fixed inset-0 bg-black/50 flex items-center justify-center z-50">
          <div className="bg-background border rounded-lg shadow-lg max-w-2xl w-full mx-4 max-h-[90vh] overflow-hidden">
            <div className="p-6 border-b flex items-center justify-between">
              <h2 className="text-2xl font-bold">Edit Vector Store</h2>
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
                isEditing={true}
                initialData={{
                  name: store?.name || "",
                  embedding_model: store?.metadata?.embedding_model || "",
                  embedding_dimension:
                    store?.metadata?.embedding_dimension || 768,
                  provider_id: store?.metadata?.provider_id || "",
                }}
              />
            </div>
          </div>
        </div>
      )}
    </>
  );
}
