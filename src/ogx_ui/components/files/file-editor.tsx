"use client";

import React, { useState } from "react";
import { Button } from "@/components/ui/button";
import { Label } from "@/components/ui/label";
import {
  Select,
  SelectContent,
  SelectItem,
  SelectTrigger,
  SelectValue,
} from "@/components/ui/select";
import { useAuthClient } from "@/hooks/use-auth-client";
import { FileUploadZone } from "./file-upload-zone";
import { FileUploadFormData, DEFAULT_EXPIRES_AFTER } from "@/lib/types";
import {
  validateUploadParams,
  formatValidationErrors,
  formatValidationWarnings,
} from "@/lib/file-validation";
import { getPurposeDescription } from "@/lib/file-utils";
import { toFile } from "llama-stack-client";

interface FileEditorProps {
  onUploadSuccess: () => void;
  onCancel: () => void;
  error?: string | null;
  showSuccessState?: boolean;
}

export function FileEditor({
  onUploadSuccess,
  onCancel,
  error,
  showSuccessState,
}: FileEditorProps) {
  const client = useAuthClient();
  const [selectedFiles, setSelectedFiles] = useState<File[]>([]);
  const [isUploading, setIsUploading] = useState(false);
  const [uploadProgress, setUploadProgress] = useState<Record<string, number>>(
    {}
  );

  const [formData, setFormData] = useState<FileUploadFormData>({
    purpose: "assistants",
    expiresAfter: DEFAULT_EXPIRES_AFTER,
  });

  const handleFilesSelected = (files: File[]) => {
    setSelectedFiles(prev => [...prev, ...files]);
  };

  const handleRemoveFile = (index: number) => {
    setSelectedFiles(prev => prev.filter((_, i) => i !== index));
  };

  const handleUpload = async () => {
    // Comprehensive validation
    const validation = validateUploadParams(
      selectedFiles,
      formData.purpose,
      formData.expiresAfter
    );

    if (!validation.isValid) {
      alert(formatValidationErrors(validation.errors));
      return;
    }

    // Show warnings if any
    if (validation.warnings.length > 0) {
      const proceed = confirm(
        `${formatValidationWarnings(validation.warnings)}\n\nDo you want to continue with the upload?`
      );
      if (!proceed) return;
    }

    setIsUploading(true);
    setUploadProgress({});

    try {
      // Upload files sequentially to show individual progress
      for (let i = 0; i < selectedFiles.length; i++) {
        const file = selectedFiles[i];
        const progressKey = `${file.name}-${i}`;

        setUploadProgress(prev => ({ ...prev, [progressKey]: 0 }));

        // Prepare upload parameters
        const uploadParams: {
          file: Awaited<ReturnType<typeof toFile>>;
          purpose: typeof formData.purpose;
          expires_after?: { anchor: "created_at"; seconds: number };
        } = {
          file: await toFile(file, file.name),
          purpose: formData.purpose,
        };

        // Add expiration if specified
        if (formData.expiresAfter && formData.expiresAfter > 0) {
          uploadParams.expires_after = {
            anchor: "created_at",
            seconds: formData.expiresAfter,
          };
        }

        // Simulate progress (since we don't have real progress from the API)
        const progressInterval = setInterval(() => {
          setUploadProgress(prev => {
            const current = prev[progressKey] || 0;
            if (current < 90) {
              return { ...prev, [progressKey]: current + 10 };
            }
            return prev;
          });
        }, 100);

        // Perform upload
        await client.files.create(uploadParams);

        // Complete progress
        clearInterval(progressInterval);
        setUploadProgress(prev => ({ ...prev, [progressKey]: 100 }));
      }

      onUploadSuccess();
    } catch (err: unknown) {
      console.error("Failed to upload files:", err);
      const errorMessage =
        err instanceof Error ? err.message : "Failed to upload files";
      alert(`Upload failed: ${errorMessage}`);
    } finally {
      setIsUploading(false);
    }
  };

  const formatExpiresAfter = (seconds: number): string => {
    if (seconds < 3600) return `${Math.floor(seconds / 60)} minutes`;
    if (seconds < 86400) return `${Math.floor(seconds / 3600)} hours`;
    return `${Math.floor(seconds / 86400)} days`;
  };

  const canUpload = selectedFiles.length > 0 && !isUploading;

  return (
    <div className="space-y-6">
      {error && (
        <div
          className={`p-4 rounded-lg ${
            showSuccessState
              ? "bg-green-50 dark:bg-green-950/20 text-green-800 dark:text-green-200 border border-green-200 dark:border-green-800"
              : "bg-red-50 dark:bg-red-950/20 text-red-800 dark:text-red-200 border border-red-200 dark:border-red-800"
          }`}
        >
          {error}
        </div>
      )}

      {!showSuccessState && (
        <>
          {/* File Upload Zone */}
          <div>
            <Label className="text-base font-medium mb-3 block">
              Select Files
            </Label>
            <FileUploadZone
              onFilesSelected={handleFilesSelected}
              selectedFiles={selectedFiles}
              onRemoveFile={handleRemoveFile}
              disabled={isUploading}
              maxFiles={10}
            />
          </div>

          {/* Upload Configuration */}
          <div className="grid grid-cols-1 md:grid-cols-2 gap-4">
            {/* Purpose Selection */}
            <div>
              <Label htmlFor="purpose" className="text-sm font-medium">
                Purpose
              </Label>
              <Select
                value={formData.purpose}
                onValueChange={(
                  value:
                    | "fine-tune"
                    | "assistants"
                    | "user_data"
                    | "batch"
                    | "vision"
                    | "evals"
                ) => setFormData(prev => ({ ...prev, purpose: value }))}
                disabled={isUploading}
              >
                <SelectTrigger>
                  <SelectValue placeholder="Select purpose" />
                </SelectTrigger>
                <SelectContent>
                  <SelectItem value="fine-tune">Fine-tuning</SelectItem>
                  <SelectItem value="assistants">Assistants</SelectItem>
                  <SelectItem value="user_data">User Data</SelectItem>
                  <SelectItem value="batch">Batch Processing</SelectItem>
                  <SelectItem value="vision">Vision</SelectItem>
                  <SelectItem value="evals">Evaluations</SelectItem>
                </SelectContent>
              </Select>
              <p className="text-xs text-muted-foreground mt-1">
                {getPurposeDescription(formData.purpose)}
              </p>
            </div>

            {/* Expiration */}
            <div>
              <Label htmlFor="expires" className="text-sm font-medium">
                Expires After
              </Label>
              <Select
                value={String(formData.expiresAfter || 0)}
                onValueChange={value =>
                  setFormData(prev => ({
                    ...prev,
                    expiresAfter: parseInt(value) || undefined,
                  }))
                }
                disabled={isUploading}
              >
                <SelectTrigger>
                  <SelectValue placeholder="Select expiration" />
                </SelectTrigger>
                <SelectContent>
                  <SelectItem value="0">Never</SelectItem>
                  <SelectItem value="3600">1 hour</SelectItem>
                  <SelectItem value="86400">1 day</SelectItem>
                  <SelectItem value="604800">7 days</SelectItem>
                  <SelectItem value="2592000">30 days</SelectItem>
                </SelectContent>
              </Select>
              <p className="text-xs text-muted-foreground mt-1">
                {formData.expiresAfter && formData.expiresAfter > 0
                  ? `Files will be automatically deleted after ${formatExpiresAfter(formData.expiresAfter)}`
                  : "Files will not expire automatically"}
              </p>
            </div>
          </div>

          {/* Upload Progress */}
          {isUploading && Object.keys(uploadProgress).length > 0 && (
            <div className="space-y-2">
              <Label className="text-sm font-medium">Upload Progress</Label>
              {selectedFiles.map((file, index) => {
                const progressKey = `${file.name}-${index}`;
                const progress = uploadProgress[progressKey] || 0;

                return (
                  <div key={progressKey} className="space-y-1">
                    <div className="flex justify-between text-sm">
                      <span>{file.name}</span>
                      <span>{progress}%</span>
                    </div>
                    <div className="w-full bg-gray-200 rounded-full h-2">
                      <div
                        className="bg-blue-600 h-2 rounded-full transition-all duration-300"
                        style={{ width: `${progress}%` }}
                      />
                    </div>
                  </div>
                );
              })}
            </div>
          )}
        </>
      )}

      {/* Action Buttons */}
      <div className="flex justify-end gap-3">
        <Button variant="outline" onClick={onCancel} disabled={isUploading}>
          {showSuccessState ? "Close" : "Cancel"}
        </Button>
        {!showSuccessState && (
          <Button
            onClick={handleUpload}
            disabled={!canUpload}
            className="min-w-[100px]"
          >
            {isUploading
              ? "Uploading..."
              : `Upload ${selectedFiles.length} File${selectedFiles.length !== 1 ? "s" : ""}`}
          </Button>
        )}
      </div>
    </div>
  );
}
