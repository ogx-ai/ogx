import {
  SupportedFileType,
  SUPPORTED_FILE_TYPES,
  MAX_FILE_SIZE,
  FileValidationError,
} from "./types";

export interface FileValidationResult {
  isValid: boolean;
  errors: FileValidationError[];
  warnings: string[];
}

/**
 * Comprehensive file validation
 */
export function validateFileForUpload(file: File): FileValidationResult {
  const errors: FileValidationError[] = [];
  const warnings: string[] = [];

  // Check if file exists
  if (!file) {
    errors.push({
      field: "file",
      message: "No file provided",
    });
    return { isValid: false, errors, warnings };
  }

  // Check file size
  if (file.size === 0) {
    errors.push({
      field: "size",
      message: "File is empty",
    });
  } else if (file.size > MAX_FILE_SIZE) {
    errors.push({
      field: "size",
      message: `File size (${formatFileSize(file.size)}) exceeds maximum limit of ${formatFileSize(MAX_FILE_SIZE)}`,
    });
  }

  // Check file type
  if (!file.type) {
    warnings.push("File type could not be determined");
  } else if (!SUPPORTED_FILE_TYPES.includes(file.type as SupportedFileType)) {
    errors.push({
      field: "type",
      message: `Unsupported file type: ${file.type}`,
    });
  }

  // Check filename
  if (!file.name) {
    errors.push({
      field: "name",
      message: "File must have a name",
    });
  } else {
    // Check for potentially problematic characters
    const problematicChars = /[<>:"/\\|?*]/g;
    if (problematicChars.test(file.name)) {
      warnings.push(
        "File name contains special characters that may cause issues"
      );
    }

    // Check filename length
    if (file.name.length > 255) {
      errors.push({
        field: "name",
        message: "File name is too long (maximum 255 characters)",
      });
    }

    // Check for extension
    const hasExtension = file.name.includes(".");
    if (!hasExtension) {
      warnings.push("File has no extension");
    }
  }

  // File-specific validations
  if (file.type === "application/pdf" && file.size > 50 * 1024 * 1024) {
    warnings.push("Large PDF files may take longer to process");
  }

  if (file.type?.startsWith("text/") && file.size > 10 * 1024 * 1024) {
    warnings.push("Large text files may have limited preview capabilities");
  }

  return {
    isValid: errors.length === 0,
    errors,
    warnings,
  };
}

/**
 * Validate file upload parameters
 */
export function validateUploadParams(
  files: File[],
  purpose:
    | "fine-tune"
    | "assistants"
    | "user_data"
    | "batch"
    | "vision"
    | "evals",
  expiresAfter?: number
): FileValidationResult {
  const errors: FileValidationError[] = [];
  const warnings: string[] = [];

  // Check files array
  if (!files || files.length === 0) {
    errors.push({
      field: "files",
      message: "At least one file is required",
    });
    return { isValid: false, errors, warnings };
  }

  // Check file limit
  if (files.length > 10) {
    errors.push({
      field: "files",
      message: "Maximum 10 files can be uploaded at once",
    });
  }

  // Check total size
  const totalSize = files.reduce((sum, file) => sum + file.size, 0);
  const maxTotalSize = MAX_FILE_SIZE * files.length; // Per-file limit applies

  if (totalSize > maxTotalSize) {
    warnings.push(
      `Total upload size is ${formatFileSize(totalSize)}. Consider uploading in smaller batches.`
    );
  }

  // Validate each file
  const duplicateNames = new Set<string>();
  const seenNames = new Set<string>();

  files.forEach((file, index) => {
    const validation = validateFileForUpload(file);

    // Add file-specific errors with context
    validation.errors.forEach(error => {
      errors.push({
        ...error,
        message: `File ${index + 1} (${file.name}): ${error.message}`,
      });
    });

    // Check for duplicate names
    if (seenNames.has(file.name)) {
      duplicateNames.add(file.name);
    } else {
      seenNames.add(file.name);
    }
  });

  // Report duplicate names
  duplicateNames.forEach(name => {
    warnings.push(`Multiple files with name "${name}" detected`);
  });

  // Validate purpose
  const validPurposes = [
    "fine-tune",
    "assistants",
    "user_data",
    "batch",
    "vision",
    "evals",
  ];
  if (!purpose || !validPurposes.includes(purpose)) {
    errors.push({
      field: "purpose",
      message: `Purpose must be one of: ${validPurposes.join(", ")}`,
    });
  }

  // Validate expiration
  if (expiresAfter !== undefined) {
    if (expiresAfter < 0) {
      errors.push({
        field: "expiresAfter",
        message: "Expiration time cannot be negative",
      });
    } else if (expiresAfter > 0 && expiresAfter < 3600) {
      errors.push({
        field: "expiresAfter",
        message: "Minimum expiration time is 1 hour (3600 seconds)",
      });
    } else if (expiresAfter > 2592000) {
      errors.push({
        field: "expiresAfter",
        message: "Maximum expiration time is 30 days (2592000 seconds)",
      });
    }
  }

  return {
    isValid: errors.length === 0,
    errors,
    warnings,
  };
}

/**
 * Format file size helper (duplicated here for independence)
 */
function formatFileSize(bytes: number): string {
  const units = ["B", "KB", "MB", "GB"];
  let size = bytes;
  let unitIndex = 0;

  while (size >= 1024 && unitIndex < units.length - 1) {
    size /= 1024;
    unitIndex++;
  }

  return `${size.toFixed(unitIndex === 0 ? 0 : 1)} ${units[unitIndex]}`;
}

/**
 * Check if file is likely corrupted based on size/type mismatch
 */
export function detectPotentialCorruption(file: File): string[] {
  const warnings: string[] = [];

  // PDF files should be at least a few KB
  if (file.type === "application/pdf" && file.size < 1024) {
    warnings.push("PDF file appears unusually small and may be corrupted");
  }

  // Word documents should be at least a few KB
  if (file.type.includes("word") && file.size < 2048) {
    warnings.push("Word document appears unusually small and may be corrupted");
  }

  // Large files with text/* MIME type might be incorrect
  if (file.type?.startsWith("text/") && file.size > 100 * 1024 * 1024) {
    warnings.push(
      "Very large text file - file type might be incorrectly detected"
    );
  }

  return warnings;
}

/**
 * Get user-friendly error messages
 */
export function formatValidationErrors(errors: FileValidationError[]): string {
  if (errors.length === 0) return "";

  if (errors.length === 1) {
    return errors[0].message;
  }

  return `Multiple issues found:\n${errors.map(e => `• ${e.message}`).join("\n")}`;
}

/**
 * Get user-friendly warning messages
 */
export function formatValidationWarnings(warnings: string[]): string {
  if (warnings.length === 0) return "";

  if (warnings.length === 1) {
    return warnings[0];
  }

  return `${warnings.length} warnings:\n${warnings.map(w => `• ${w}`).join("\n")}`;
}
