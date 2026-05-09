"use client";

import React, { useState, useEffect } from "react";
import { useParams, useRouter } from "next/navigation";
import { Button } from "@/components/ui/button";
import { Card, CardContent, CardHeader, CardTitle } from "@/components/ui/card";
import { Download, Trash2, ArrowLeft, FileText } from "lucide-react";
import { useAuthClient } from "@/hooks/use-auth-client";
import { FileResource } from "@/lib/types";
import {
  formatFileSize,
  getFileTypeIcon,
  formatTimestamp,
  formatPurpose,
  isTextFile,
} from "@/lib/file-utils";
import {
  DetailLoadingView,
  DetailErrorView,
  DetailNotFoundView,
  DetailLayout,
  PropertiesCard,
  PropertyItem,
} from "@/components/layout/detail-layout";
import { CopyButton } from "@/components/ui/copy-button";
import { CSVViewer } from "./csv-viewer";
import { JsonViewer } from "./json-viewer";

// Content size limits
const MAX_TEXT_PREVIEW_SIZE = 50 * 1024 * 1024; // 50MB for text files
const WARN_TEXT_PREVIEW_SIZE = 10 * 1024 * 1024; // 10MB warning threshold

export function FileDetail() {
  const params = useParams();
  const router = useRouter();
  const client = useAuthClient();
  const fileId = params.id as string;

  const [file, setFile] = useState<FileResource | null>(null);
  const [fileContent, setFileContent] = useState<string | null>(null);
  const [fileContentUrl, setFileContentUrl] = useState<string | null>(null);
  const [loading, setLoading] = useState(true);
  const [contentLoading, setContentLoading] = useState(false);
  const [error, setError] = useState<Error | null>(null);
  const [contentError, setContentError] = useState<string | null>(null);
  const [sizeWarning, setSizeWarning] = useState<string | null>(null);
  const [isDeleting, setIsDeleting] = useState(false);

  useEffect(() => {
    if (!fileId) return;

    const fetchFile = async () => {
      try {
        setLoading(true);
        setError(null);

        const response = await client.files.retrieve(fileId);
        setFile(response as FileResource);
      } catch (err) {
        console.error("Failed to fetch file:", err);
        setError(
          err instanceof Error ? err : new Error("Failed to fetch file")
        );
      } finally {
        setLoading(false);
      }
    };

    fetchFile();
  }, [fileId, client]);

  // Cleanup blob URL when component unmounts or content changes
  useEffect(() => {
    return () => {
      if (fileContentUrl) {
        URL.revokeObjectURL(fileContentUrl);
      }
    };
  }, [fileContentUrl]);

  const handleLoadContent = async () => {
    if (!file) return;

    try {
      setContentLoading(true);
      setContentError(null); // Clear any previous errors
      setSizeWarning(null); // Clear any previous size warnings

      // Check file size before processing
      if (file.bytes > MAX_TEXT_PREVIEW_SIZE) {
        setContentError(
          `File is too large to preview (${formatFileSize(file.bytes)}). Maximum supported size is ${formatFileSize(MAX_TEXT_PREVIEW_SIZE)}.`
        );
        return;
      }

      if (file.bytes > WARN_TEXT_PREVIEW_SIZE) {
        setSizeWarning(
          `Large file detected (${formatFileSize(file.bytes)}). Loading may take longer than usual.`
        );
      }

      // Clean up existing blob URL
      if (fileContentUrl) {
        URL.revokeObjectURL(fileContentUrl);
        setFileContentUrl(null);
      }

      // Determine MIME type from file extension
      const extension = file.filename.split(".").pop()?.toLowerCase();
      let mimeType = "application/octet-stream"; // Default

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

      // For binary files (PDF, Word, images), fetch directly to avoid client parsing
      const isBinaryFile = [
        "pdf",
        "docx",
        "doc",
        "png",
        "jpg",
        "jpeg",
        "gif",
        "bmp",
        "webp",
      ].includes(extension || "");

      let blob: Blob;
      let textContent: string | null = null;

      // TODO: Use llama stack client consistently for all file types
      // Currently using direct fetch for binary files to ensure proper rendering
      if (isBinaryFile) {
        // For binary files, use direct fetch to preserve binary integrity
        const contentResponse = await fetch(`/api/v1/files/${fileId}/content`);
        if (!contentResponse.ok) {
          throw new Error(
            `Failed to fetch content: ${contentResponse.status} ${contentResponse.statusText}`
          );
        }
        const arrayBuffer = await contentResponse.arrayBuffer();
        blob = new Blob([arrayBuffer], { type: mimeType });
      } else {
        // Use llama stack client for text content
        const response = await client.files.content(fileId);

        if (typeof response === "string") {
          blob = new Blob([response], { type: mimeType });
          textContent = response;
        } else if (response instanceof Blob) {
          blob = response;
        } else if (response instanceof ArrayBuffer) {
          blob = new Blob([response], { type: mimeType });
        } else {
          // Handle other response types (convert to JSON)
          const jsonString = JSON.stringify(response, null, 2);
          blob = new Blob([jsonString], { type: "application/json" });
          textContent = jsonString;
        }
      }

      const blobUrl = URL.createObjectURL(blob);
      setFileContentUrl(blobUrl);

      // Keep text content for copy functionality and CSV/JSON viewers
      if (
        textContent &&
        (isTextFile(mimeType) || extension === "csv" || extension === "json")
      ) {
        setFileContent(textContent);
      }
    } catch (err) {
      console.error("Failed to load file content:", err);

      // Clean up any partially created blob URL on error
      if (fileContentUrl) {
        URL.revokeObjectURL(fileContentUrl);
        setFileContentUrl(null);
      }

      const errorMessage = err instanceof Error ? err.message : "Unknown error";
      setContentError(
        `Failed to load file content: ${errorMessage}. Please try again or check if the file still exists.`
      );
    } finally {
      setContentLoading(false);
    }
  };

  const handleDownload = async () => {
    if (!file) return;

    try {
      // Determine MIME type from file extension
      const extension = file.filename.split(".").pop()?.toLowerCase();
      let mimeType = "application/octet-stream";

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

      // For binary files (PDF, Word, images), detect to handle properly
      const isBinaryFile = [
        "pdf",
        "docx",
        "doc",
        "png",
        "jpg",
        "jpeg",
        "gif",
        "bmp",
        "webp",
      ].includes(extension || "");

      let downloadUrl: string;

      // TODO: Use llama stack client consistently for all file types
      // Currently using direct fetch for binary files to ensure proper downloading
      if (isBinaryFile) {
        // For binary files, use direct fetch to preserve binary integrity
        const contentResponse = await fetch(`/api/v1/files/${fileId}/content`);
        if (!contentResponse.ok) {
          throw new Error(
            `Failed to fetch content: ${contentResponse.status} ${contentResponse.statusText}`
          );
        }
        const arrayBuffer = await contentResponse.arrayBuffer();
        const blob = new Blob([arrayBuffer], { type: mimeType });
        downloadUrl = URL.createObjectURL(blob);
      } else {
        // Use llama stack client for text content
        const response = await client.files.content(fileId);

        if (typeof response === "string") {
          const blob = new Blob([response], { type: mimeType });
          downloadUrl = URL.createObjectURL(blob);
        } else if (response instanceof Blob) {
          downloadUrl = URL.createObjectURL(response);
        } else {
          const blob = new Blob([JSON.stringify(response, null, 2)], {
            type: "application/json",
          });
          downloadUrl = URL.createObjectURL(blob);
        }
      }

      // Trigger download
      const link = document.createElement("a");
      link.href = downloadUrl;
      link.download = file.filename;
      document.body.appendChild(link);
      link.click();
      document.body.removeChild(link);
      URL.revokeObjectURL(downloadUrl);
    } catch (err) {
      console.error("Failed to download file:", err);
      setContentError("Failed to download file. Please try again.");
    }
  };

  const handleDelete = async () => {
    if (!file) return;

    if (
      !confirm(
        `Are you sure you want to delete "${file.filename}"? This action cannot be undone.`
      )
    ) {
      return;
    }

    try {
      setIsDeleting(true);
      await client.files.delete(fileId);
      router.push("/logs/files");
    } catch (err) {
      console.error("Failed to delete file:", err);
      setContentError("Failed to delete file. Please try again.");
    } finally {
      setIsDeleting(false);
    }
  };

  if (loading) {
    return <DetailLoadingView />;
  }

  if (error) {
    return <DetailErrorView title="File Details" id={fileId} error={error} />;
  }

  if (!file) {
    return <DetailNotFoundView title="File Details" id={fileId} />;
  }

  const isExpired = file.expires_at && file.expires_at * 1000 < Date.now();
  const fileExtension = file.filename.split(".").pop()?.toLowerCase();
  const fileIcon = getFileTypeIcon(fileExtension);
  const isCSVFile = fileExtension === "csv";
  const isJsonFile = fileExtension === "json";

  // Security: File type whitelist for preview
  // In local development, be permissive but still maintain reasonable security
  const SAFE_PREVIEW_EXTENSIONS = [
    "txt",
    "plain",
    "csv",
    "json",
    "pdf",
    "html",
    "htm",
    "docx",
    "doc",
    "md",
    "markdown",
    "xml",
    "log",
  ];
  const canPreview =
    !fileExtension || SAFE_PREVIEW_EXTENSIONS.includes(fileExtension);

  const mainContent = (
    <div className="space-y-6" data-main-content>
      {/* File Header */}
      <Card>
        <CardHeader>
          <div className="flex items-start justify-between">
            <div className="flex items-center gap-3">
              <div className="text-2xl">{fileIcon}</div>
              <div>
                <CardTitle className="text-xl">{file.filename}</CardTitle>
                <div className="flex items-center gap-4 mt-2 text-sm text-muted-foreground">
                  <span>{formatFileSize(file.bytes)}</span>
                  <span>•</span>
                  <span>
                    {file.filename.split(".").pop()?.toUpperCase() || "Unknown"}
                  </span>
                  <span>•</span>
                  <span>{formatPurpose(file.purpose)}</span>
                </div>
              </div>
            </div>
            <div className="flex gap-2">
              <Button variant="outline" onClick={handleDownload}>
                <Download className="h-4 w-4 mr-2" />
                Download
              </Button>
              <Button
                variant="destructive"
                onClick={handleDelete}
                disabled={isDeleting}
              >
                <Trash2 className="h-4 w-4 mr-2" />
                {isDeleting ? "Deleting..." : "Delete"}
              </Button>
            </div>
          </div>
        </CardHeader>
      </Card>

      {/* File Content Preview */}
      {canPreview && (
        <Card>
          <CardHeader>
            <div className="flex items-center justify-between">
              <CardTitle className="text-lg flex items-center gap-2">
                <FileText className="h-5 w-5" />
                Content Preview
              </CardTitle>
              {!fileContentUrl && (
                <Button
                  variant="outline"
                  onClick={handleLoadContent}
                  disabled={contentLoading}
                >
                  {contentLoading ? "Loading..." : "Load Content"}
                </Button>
              )}
              {fileContentUrl && (
                <div className="flex gap-2">
                  <Button
                    variant="outline"
                    size="sm"
                    onClick={() => window.open(fileContentUrl, "_blank")}
                  >
                    Open in New Tab
                  </Button>
                  <Button
                    variant="outline"
                    size="sm"
                    onClick={() => {
                      const link = document.createElement("a");
                      link.href = fileContentUrl;
                      link.download = file.filename;
                      link.click();
                    }}
                  >
                    Download
                  </Button>
                </div>
              )}
            </div>
          </CardHeader>
          {contentError && (
            <CardContent>
              <div className="p-4 border border-red-200 rounded-lg bg-red-50 dark:bg-red-900/20 dark:border-red-800">
                <div className="flex items-center justify-between">
                  <div>
                    <h4 className="font-semibold text-red-800 dark:text-red-300 mb-1">
                      Content Error
                    </h4>
                    <p className="text-sm text-red-600 dark:text-red-400">
                      {contentError}
                    </p>
                  </div>
                  <Button
                    variant="ghost"
                    size="sm"
                    onClick={() => setContentError(null)}
                    className="text-red-600 hover:text-red-800 dark:text-red-400 dark:hover:text-red-300"
                  >
                    Dismiss
                  </Button>
                </div>
              </div>
            </CardContent>
          )}
          {sizeWarning && (
            <CardContent>
              <div className="p-4 border border-orange-200 rounded-lg bg-orange-50 dark:bg-orange-900/20 dark:border-orange-800">
                <div className="flex items-center justify-between">
                  <div>
                    <h4 className="font-semibold text-orange-800 dark:text-orange-300 mb-1">
                      Large File Warning
                    </h4>
                    <p className="text-sm text-orange-600 dark:text-orange-400">
                      {sizeWarning}
                    </p>
                  </div>
                  <Button
                    variant="ghost"
                    size="sm"
                    onClick={() => setSizeWarning(null)}
                    className="text-orange-600 hover:text-orange-800 dark:text-orange-400 dark:hover:text-orange-300"
                  >
                    Dismiss
                  </Button>
                </div>
              </div>
            </CardContent>
          )}
          {fileContentUrl && (
            <CardContent>
              <div className="w-full">
                {isCSVFile && fileContent ? (
                  // CSV files: Use custom CSV viewer
                  <CSVViewer content={fileContent} />
                ) : isJsonFile && fileContent ? (
                  // JSON files: Use custom JSON viewer
                  <JsonViewer content={fileContent} />
                ) : (
                  // Other files: Use iframe preview
                  <div className="relative">
                    {fileContent && (
                      <div className="absolute top-2 right-2 z-10">
                        <CopyButton
                          content={fileContent}
                          copyMessage="Copied file content to clipboard!"
                        />
                      </div>
                    )}
                    <iframe
                      key={fileContentUrl} // Force iframe reload when URL changes
                      src={fileContentUrl}
                      className="w-full h-[600px] border rounded-lg bg-white dark:bg-gray-900"
                      title="File Preview"
                      onError={() => {
                        console.warn(
                          "Iframe failed to load content, this may be a browser security restriction"
                        );
                      }}
                    />
                  </div>
                )}
              </div>
            </CardContent>
          )}
        </Card>
      )}

      {/* Additional Information */}
      <Card>
        <CardHeader>
          <CardTitle>File Information</CardTitle>
        </CardHeader>
        <CardContent className="space-y-2">
          <div>
            <span className="font-medium">File ID:</span>
            <div className="flex items-center gap-2 mt-1">
              <code className="bg-muted px-2 py-1 rounded text-sm font-mono">
                {file.id}
              </code>
              <CopyButton
                content={file.id}
                copyMessage="Copied file ID to clipboard!"
              />
            </div>
          </div>

          {file.expires_at && (
            <div>
              <span className="font-medium">Status:</span>
              <div className="mt-1">
                <span
                  className={`inline-block px-2 py-1 rounded text-sm ${
                    isExpired
                      ? "bg-red-100 text-red-800 dark:bg-red-900/20 dark:text-red-300"
                      : "bg-green-100 text-green-800 dark:bg-green-900/20 dark:text-green-300"
                  }`}
                >
                  {isExpired ? "Expired" : "Active"}
                </span>
              </div>
            </div>
          )}
        </CardContent>
      </Card>
    </div>
  );

  const sidebar = (
    <div className="space-y-4">
      {/* Navigation */}
      <Card>
        <CardContent className="p-4">
          <Button
            variant="ghost"
            onClick={() => router.push("/logs/files")}
            className="w-full justify-start p-0"
          >
            <ArrowLeft className="h-4 w-4 mr-2" />
            Back to Files
          </Button>
        </CardContent>
      </Card>

      {/* Properties */}
      <PropertiesCard>
        <PropertyItem label="ID" value={file.id} />
        <PropertyItem label="Filename" value={file.filename} />
        <PropertyItem label="Size" value={formatFileSize(file.bytes)} />
        <PropertyItem label="Purpose" value={formatPurpose(file.purpose)} />
        <PropertyItem
          label="Created"
          value={formatTimestamp(file.created_at)}
        />
        {file.expires_at && (
          <PropertyItem
            label="Expires"
            value={
              <span className={isExpired ? "text-destructive" : ""}>
                {formatTimestamp(file.expires_at)}
              </span>
            }
          />
        )}
      </PropertiesCard>
    </div>
  );

  return (
    <DetailLayout
      title="File Details"
      mainContent={mainContent}
      sidebar={sidebar}
    />
  );
}
