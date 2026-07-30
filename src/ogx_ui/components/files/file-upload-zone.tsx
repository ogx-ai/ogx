"use client";

import React, { useState, useCallback } from "react";
import { Button } from "@/components/ui/button";
import { Upload, X } from "lucide-react";
import { cn } from "@/lib/utils";
import { formatFileSize, getFileTypeIcon } from "@/lib/file-utils";
import { SUPPORTED_FILE_TYPES } from "@/lib/types";
import {
  validateFileForUpload,
  formatValidationErrors,
  detectPotentialCorruption,
} from "@/lib/file-validation";

interface FileUploadZoneProps {
  onFilesSelected: (files: File[]) => void;
  selectedFiles: File[];
  onRemoveFile: (index: number) => void;
  disabled?: boolean;
  maxFiles?: number;
  className?: string;
}

export function FileUploadZone({
  onFilesSelected,
  selectedFiles,
  onRemoveFile,
  disabled = false,
  maxFiles = 10,
  className,
}: FileUploadZoneProps) {
  const [isDragOver, setIsDragOver] = useState(false);
  // eslint-disable-next-line @typescript-eslint/no-unused-vars
  const [dragCounter, setDragCounter] = useState(0);

  const handleDragEnter = useCallback((e: React.DragEvent) => {
    e.preventDefault();
    e.stopPropagation();
    setDragCounter(prev => prev + 1);
    setIsDragOver(true);
  }, []);

  const handleDragLeave = useCallback((e: React.DragEvent) => {
    e.preventDefault();
    e.stopPropagation();
    setDragCounter(prev => {
      const newCount = prev - 1;
      if (newCount === 0) {
        setIsDragOver(false);
      }
      return newCount;
    });
  }, []);

  const handleFiles = useCallback(
    (files: File[]) => {
      if (disabled) return;

      // Check file limit
      const availableSlots = maxFiles - selectedFiles.length;
      if (files.length > availableSlots) {
        alert(
          `You can only upload ${availableSlots} more file(s). Maximum ${maxFiles} files allowed.`
        );
        return;
      }

      // Validate all files first
      const validFiles: File[] = [];
      const allErrors: string[] = [];
      const allWarnings: string[] = [];

      files.forEach(file => {
        const validation = validateFileForUpload(file);
        const corruptionWarnings = detectPotentialCorruption(file);

        if (validation.isValid) {
          // Check for duplicates
          const isDuplicate = selectedFiles.some(
            selected =>
              selected.name === file.name && selected.size === file.size
          );

          if (isDuplicate) {
            allErrors.push(`"${file.name}" is already selected`);
          } else {
            validFiles.push(file);

            // Collect warnings
            if (
              validation.warnings.length > 0 ||
              corruptionWarnings.length > 0
            ) {
              const fileWarnings = [
                ...validation.warnings,
                ...corruptionWarnings,
              ];
              allWarnings.push(`"${file.name}": ${fileWarnings.join(", ")}`);
            }
          }
        } else {
          allErrors.push(
            `"${file.name}": ${formatValidationErrors(validation.errors)}`
          );
        }
      });

      // Show errors if any
      if (allErrors.length > 0) {
        alert(`Some files could not be added:\n\n${allErrors.join("\n")}`);
      }

      // Show warnings if any valid files have warnings
      if (allWarnings.length > 0 && validFiles.length > 0) {
        const proceed = confirm(
          `Warning(s) for some files:\n\n${allWarnings.join("\n")}\n\nDo you want to continue adding these files?`
        );
        if (!proceed) return;
      }

      // Add valid files
      if (validFiles.length > 0) {
        onFilesSelected(validFiles);
      }
    },
    [disabled, maxFiles, selectedFiles, onFilesSelected]
  );

  const handleDragOver = useCallback((e: React.DragEvent) => {
    e.preventDefault();
    e.stopPropagation();
  }, []);

  const handleDrop = useCallback(
    (e: React.DragEvent) => {
      e.preventDefault();
      e.stopPropagation();

      setIsDragOver(false);
      setDragCounter(0);

      if (disabled) return;

      const files = Array.from(e.dataTransfer.files);
      handleFiles(files);
    },
    [disabled, handleFiles]
  );

  const handleFileSelect = useCallback(() => {
    if (disabled) return;

    const input = document.createElement("input");
    input.type = "file";
    input.multiple = maxFiles > 1;
    input.accept = SUPPORTED_FILE_TYPES.join(",");

    input.onchange = e => {
      const target = e.target as HTMLInputElement;
      if (target.files) {
        const files = Array.from(target.files);
        handleFiles(files);
      }
    };

    input.click();
  }, [disabled, maxFiles, handleFiles]);

  const getSupportedFormats = () => {
    return SUPPORTED_FILE_TYPES.map(type => {
      switch (type) {
        case "application/pdf":
          return "PDF";
        case "text/plain":
          return "TXT";
        case "text/markdown":
          return "MD";
        case "text/html":
          return "HTML";
        case "text/csv":
          return "CSV";
        case "application/json":
          return "JSON";
        case "application/vnd.openxmlformats-officedocument.wordprocessingml.document":
          return "DOCX";
        case "application/msword":
          return "DOC";
        default:
          return type;
      }
    }).join(", ");
  };

  return (
    <div className={cn("space-y-4", className)}>
      {/* Upload Zone */}
      <div
        onDragEnter={handleDragEnter}
        onDragLeave={handleDragLeave}
        onDragOver={handleDragOver}
        onDrop={handleDrop}
        className={cn(
          "border-2 border-dashed rounded-lg p-8 text-center transition-colors cursor-pointer",
          isDragOver
            ? "border-blue-500 bg-blue-50 dark:bg-blue-950/20"
            : "border-gray-300 dark:border-gray-600",
          disabled && "opacity-50 cursor-not-allowed"
        )}
        onClick={handleFileSelect}
      >
        <Upload className="h-12 w-12 mx-auto mb-4 text-gray-400" />

        <div className="space-y-2">
          <p className="text-lg font-medium">
            {isDragOver
              ? "Drop files here"
              : "Click to upload or drag and drop"}
          </p>
          <p className="text-sm text-muted-foreground">
            Supported formats: {getSupportedFormats()}
          </p>
          <p className="text-xs text-muted-foreground">
            Max file size: {formatFileSize(100 * 1024 * 1024)} • Max {maxFiles}{" "}
            files
          </p>
        </div>
      </div>

      {/* Selected Files List */}
      {selectedFiles.length > 0 && (
        <div className="space-y-2">
          <h4 className="text-sm font-medium">
            Selected Files ({selectedFiles.length})
          </h4>
          <div className="space-y-2 max-h-40 overflow-y-auto">
            {selectedFiles.map((file, index) => {
              const icon = getFileTypeIcon(file.name.split(".").pop());

              return (
                <div
                  key={`${file.name}-${index}`}
                  className="flex items-center gap-3 p-3 border rounded-lg bg-muted/50"
                >
                  <div className="text-lg">{icon}</div>

                  <div className="flex-1 min-w-0">
                    <div className="flex items-center gap-2">
                      <span className="font-medium truncate" title={file.name}>
                        {file.name}
                      </span>
                      <span className="text-xs bg-gray-200 dark:bg-gray-700 px-2 py-1 rounded">
                        {file.name.split(".").pop()?.toUpperCase()}
                      </span>
                    </div>
                    <div className="text-sm text-muted-foreground">
                      {formatFileSize(file.size)}
                    </div>
                  </div>

                  <Button
                    variant="ghost"
                    size="sm"
                    onClick={e => {
                      e.stopPropagation();
                      onRemoveFile(index);
                    }}
                    disabled={disabled}
                    className="h-8 w-8 p-0"
                  >
                    <X className="h-4 w-4" />
                  </Button>
                </div>
              );
            })}
          </div>
        </div>
      )}
    </div>
  );
}
