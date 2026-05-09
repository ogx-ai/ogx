"use client";

import React from "react";
import type { ListFilesResponse, FileResource } from "@/lib/types";
import { useRouter } from "next/navigation";
import { usePagination } from "@/hooks/use-pagination";
import { Button } from "@/components/ui/button";
import { Plus, Trash2, Search, Download, X } from "lucide-react";
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
import { FileEditor } from "./file-editor";
import {
  formatFileSize,
  getFileTypeIcon,
  formatTimestamp,
  formatPurpose,
  truncateFilename,
} from "@/lib/file-utils";

export function FilesManagement() {
  const router = useRouter();
  const client = useAuthClient();
  const [deletingFiles, setDeletingFiles] = useState<Set<string>>(new Set());
  const [searchTerm, setSearchTerm] = useState("");
  const [showUploadModal, setShowUploadModal] = useState(false);
  const [modalError, setModalError] = useState<string | null>(null);
  const [showSuccessState, setShowSuccessState] = useState(false);

  const {
    data: files,
    status,
    hasMore,
    error,
    loadMore,
    refetch,
  } = usePagination<FileResource>({
    limit: 20,
    order: "desc",
    fetchFunction: async (client, params) => {
      const response = await client.files.list({
        after: params.after,
        limit: params.limit,
        order: params.order,
      });
      return response as ListFilesResponse;
    },
    errorMessagePrefix: "files",
  });

  // Auto-load all pages for infinite scroll behavior (like other features)
  React.useEffect(() => {
    if (status === "idle" && hasMore) {
      loadMore();
    }
  }, [status, hasMore, loadMore]);

  // Handle ESC key to close modal
  React.useEffect(() => {
    const handleEscape = (event: KeyboardEvent) => {
      if (event.key === "Escape" && showUploadModal) {
        handleCancel();
      }
    };

    document.addEventListener("keydown", handleEscape);
    return () => document.removeEventListener("keydown", handleEscape);
  }, [showUploadModal]);

  const handleDeleteFile = async (fileId: string) => {
    if (
      !confirm(
        "Are you sure you want to delete this file? This action cannot be undone."
      )
    ) {
      return;
    }

    setDeletingFiles(prev => new Set([...prev, fileId]));

    try {
      await client.files.delete(fileId);
      // Refresh the data to reflect the deletion
      refetch();
    } catch (err: unknown) {
      console.error("Failed to delete file:", err);
      const errorMessage = err instanceof Error ? err.message : "Unknown error";
      alert(`Failed to delete file: ${errorMessage}`);
    } finally {
      setDeletingFiles(prev => {
        const newSet = new Set(prev);
        newSet.delete(fileId);
        return newSet;
      });
    }
  };

  const handleDownloadFile = async (fileId: string, filename: string) => {
    try {
      // Show loading state (could be expanded with UI feedback)
      console.log(`Starting download for file: ${filename}`);

      const response = await client.files.content(fileId);

      // Create download link
      let downloadUrl: string;
      let mimeType = "application/octet-stream";

      // Determine MIME type from file extension
      const extension = filename.split(".").pop()?.toLowerCase();
      switch (extension) {
        case "pdf":
          mimeType = "application/pdf";
          break;
        case "txt":
          mimeType = "text/plain";
          break;
        case "md":
          mimeType = "text/markdown";
          break;
        case "html":
          mimeType = "text/html";
          break;
        case "csv":
          mimeType = "text/csv";
          break;
        case "json":
          mimeType = "application/json";
          break;
        case "docx":
          mimeType =
            "application/vnd.openxmlformats-officedocument.wordprocessingml.document";
          break;
        case "doc":
          mimeType = "application/msword";
          break;
      }

      if (typeof response === "string") {
        const blob = new Blob([response], { type: mimeType });
        downloadUrl = URL.createObjectURL(blob);
      } else if (response instanceof Blob) {
        downloadUrl = URL.createObjectURL(response);
      } else {
        // Handle other response types by converting to JSON string
        const blob = new Blob([JSON.stringify(response, null, 2)], {
          type: "application/json",
        });
        downloadUrl = URL.createObjectURL(blob);
      }

      // Trigger download
      const link = document.createElement("a");
      link.href = downloadUrl;
      link.download = filename;
      document.body.appendChild(link);
      link.click();
      document.body.removeChild(link);

      // Clean up
      setTimeout(() => {
        URL.revokeObjectURL(downloadUrl);
      }, 1000);

      console.log(`Download completed for file: ${filename}`);
    } catch (err: unknown) {
      console.error("Failed to download file:", err);
      let errorMessage = "Unknown error occurred";

      if (err instanceof Error) {
        errorMessage = err.message;

        // Provide more specific error messages
        if (err.message.includes("404") || err.message.includes("not found")) {
          errorMessage = "File not found. It may have been deleted or moved.";
        } else if (
          err.message.includes("403") ||
          err.message.includes("forbidden")
        ) {
          errorMessage =
            "Access denied. You may not have permission to download this file.";
        } else if (
          err.message.includes("network") ||
          err.message.includes("fetch")
        ) {
          errorMessage =
            "Network error. Please check your connection and try again.";
        }
      }

      alert(`Failed to download "${filename}": ${errorMessage}`);
    }
  };

  const handleUploadSuccess = () => {
    // Show success message and refresh data
    setShowSuccessState(true);
    setModalError(
      "✅ Upload successful! File list updated. You can now close this modal."
    );
    refetch();
  };

  const handleCancel = () => {
    setShowUploadModal(false);
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

    if (!files || files.length === 0) {
      return (
        <div className="text-center py-8">
          <p className="text-muted-foreground mb-4">No files found.</p>
          <Button onClick={() => setShowUploadModal(true)}>
            <Plus className="h-4 w-4 mr-2" />
            Upload Your First File
          </Button>
        </div>
      );
    }

    // Filter files based on search term
    const filteredFiles = files.filter(file => {
      if (!searchTerm) return true;

      const searchLower = searchTerm.toLowerCase();
      return (
        file.id.toLowerCase().includes(searchLower) ||
        file.filename.toLowerCase().includes(searchLower) ||
        file.purpose.toLowerCase().includes(searchLower)
      );
    });

    return (
      <div className="space-y-4">
        {/* Search Bar */}
        <div className="relative flex-1 max-w-md">
          <Search className="absolute left-3 top-1/2 transform -translate-y-1/2 text-muted-foreground h-4 w-4" />
          <Input
            placeholder="Search files..."
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
                <TableHead>File Name</TableHead>
                <TableHead>Type</TableHead>
                <TableHead>Size</TableHead>
                <TableHead>Purpose</TableHead>
                <TableHead>Created</TableHead>
                <TableHead>Expires</TableHead>
                <TableHead>Actions</TableHead>
              </TableRow>
            </TableHeader>
            <TableBody>
              {filteredFiles.map(file => {
                const fileIcon = getFileTypeIcon(
                  file.filename.split(".").pop()
                );
                const isExpired =
                  file.expires_at && file.expires_at * 1000 < Date.now();

                return (
                  <TableRow
                    key={file.id}
                    onClick={() => router.push(`/logs/files/${file.id}`)}
                    className="cursor-pointer hover:bg-muted/50"
                  >
                    <TableCell>
                      <Button
                        variant="link"
                        className="p-0 h-auto font-mono text-blue-600 hover:text-blue-800 dark:text-blue-400 dark:hover:text-blue-300"
                        onClick={() => router.push(`/logs/files/${file.id}`)}
                      >
                        {file.id}
                      </Button>
                    </TableCell>
                    <TableCell>
                      <div className="flex items-center gap-2">
                        <span>{fileIcon}</span>
                        <span title={file.filename}>
                          {truncateFilename(file.filename)}
                        </span>
                      </div>
                    </TableCell>
                    <TableCell>
                      {file.filename.split(".").pop()?.toUpperCase() ||
                        "Unknown"}
                    </TableCell>
                    <TableCell>{formatFileSize(file.bytes)}</TableCell>
                    <TableCell>{formatPurpose(file.purpose)}</TableCell>
                    <TableCell>{formatTimestamp(file.created_at)}</TableCell>
                    <TableCell>
                      {file.expires_at ? (
                        <span className={isExpired ? "text-destructive" : ""}>
                          {formatTimestamp(file.expires_at)}
                        </span>
                      ) : (
                        "Never"
                      )}
                    </TableCell>
                    <TableCell>
                      <div className="flex gap-2">
                        <Button
                          variant="outline"
                          size="sm"
                          onClick={e => {
                            e.stopPropagation();
                            handleDownloadFile(file.id, file.filename);
                          }}
                          title="Download file"
                        >
                          <Download className="h-4 w-4" />
                        </Button>
                        <Button
                          variant="outline"
                          size="sm"
                          onClick={e => {
                            e.stopPropagation();
                            handleDeleteFile(file.id);
                          }}
                          disabled={deletingFiles.has(file.id)}
                          title="Delete file"
                        >
                          {deletingFiles.has(file.id) ? (
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
        <h1 className="text-2xl font-semibold">Files</h1>
        <Button
          onClick={() => setShowUploadModal(true)}
          disabled={status === "loading"}
        >
          <Plus className="h-4 w-4 mr-2" />
          Upload File
        </Button>
      </div>
      {renderContent()}

      {/* Upload File Modal */}
      {showUploadModal && (
        <div className="fixed inset-0 bg-black/50 flex items-center justify-center z-50">
          <div className="bg-background border rounded-lg shadow-lg max-w-2xl w-full mx-4 max-h-[90vh] overflow-hidden">
            <div className="p-6 border-b flex items-center justify-between">
              <h2 className="text-2xl font-bold">Upload File</h2>
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
              <FileEditor
                onUploadSuccess={handleUploadSuccess}
                onCancel={handleCancel}
                error={modalError}
                showSuccessState={showSuccessState}
              />
            </div>
          </div>
        </div>
      )}
    </div>
  );
}
