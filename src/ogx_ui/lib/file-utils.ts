import {
  SupportedFileType,
  SUPPORTED_FILE_TYPES,
  FILE_TYPE_EXTENSIONS,
  MAX_FILE_SIZE,
} from "./types";

/**
 * Format file size in human readable format
 */
export function formatFileSize(bytes: number): string {
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
 * Get file type icon based on file extension or MIME type
 */
export function getFileTypeIcon(
  fileExtensionOrMimeType: string | undefined
): string {
  if (!fileExtensionOrMimeType) return "📄";

  // If it's a file extension (starts with . or is just the extension)
  let extension = fileExtensionOrMimeType.toLowerCase();
  if (extension.startsWith(".")) {
    extension = extension.substring(1);
  }

  // Check by file extension first (more reliable)
  switch (extension) {
    case "pdf":
      return "📕"; // Red book for PDF
    case "html":
    case "htm":
      return "🌐"; // Globe for HTML
    case "txt":
      return "📄"; // Generic document for text
    case "md":
    case "markdown":
      return "📝"; // Memo for Markdown
    case "csv":
      return "📊"; // Bar chart for CSV
    case "json":
      return "⚙️"; // Gear for JSON
    case "docx":
    case "doc":
      return "📘"; // Blue book for Word documents
    case "xlsx":
    case "xls":
      return "📗"; // Green book for Excel
    case "pptx":
    case "ppt":
      return "📙"; // Orange book for PowerPoint
    case "zip":
    case "rar":
    case "7z":
      return "🗜️"; // Compression for archives
    case "jpg":
    case "jpeg":
    case "png":
    case "gif":
    case "svg":
      return "🖼️"; // Framed picture for images
    case "mp4":
    case "avi":
    case "mov":
      return "🎬"; // Movie camera for videos
    case "mp3":
    case "wav":
    case "flac":
      return "🎵"; // Musical note for audio
    case "js":
    case "ts":
      return "⚡"; // Lightning for JavaScript/TypeScript
    case "py":
      return "🐍"; // Snake for Python
    case "java":
      return "☕"; // Coffee for Java
    case "cpp":
    case "c":
      return "🔧"; // Wrench for C/C++
    case "xml":
      return "🏷️"; // Label for XML
    case "css":
      return "🎨"; // Palette for CSS
    default:
      break;
  }

  // Fallback to MIME type checking for unknown extensions
  const mimeType = fileExtensionOrMimeType.toLowerCase();

  if (mimeType.startsWith("text/")) {
    if (mimeType === "text/markdown") return "📝";
    if (mimeType === "text/html") return "🌐";
    if (mimeType === "text/csv") return "📊";
    if (mimeType === "text/plain") return "📄";
    return "📄";
  }

  if (mimeType === "application/pdf") return "📕";
  if (mimeType === "application/json") return "⚙️";
  if (mimeType.includes("document") || mimeType.includes("word")) return "📘";
  if (mimeType.includes("spreadsheet") || mimeType.includes("excel"))
    return "📗";
  if (mimeType.includes("presentation") || mimeType.includes("powerpoint"))
    return "📙";
  if (mimeType.startsWith("image/")) return "🖼️";
  if (mimeType.startsWith("video/")) return "🎬";
  if (mimeType.startsWith("audio/")) return "🎵";

  return "📄";
}

/**
 * Validate file before upload
 */
export function validateFile(file: File): {
  isValid: boolean;
  errors: string[];
} {
  const errors: string[] = [];

  // Check file size
  if (file.size > MAX_FILE_SIZE) {
    errors.push(
      `File size exceeds maximum limit of ${formatFileSize(MAX_FILE_SIZE)}`
    );
  }

  // Check file type
  if (!SUPPORTED_FILE_TYPES.includes(file.type as SupportedFileType)) {
    const supportedExtensions = Object.values(FILE_TYPE_EXTENSIONS).join(", ");
    errors.push(
      `Unsupported file type. Supported types: ${supportedExtensions}`
    );
  }

  // Check for empty file
  if (file.size === 0) {
    errors.push("File appears to be empty");
  }

  return {
    isValid: errors.length === 0,
    errors,
  };
}

/**
 * Get file extension from filename
 */
export function getFileExtension(filename: string): string {
  const lastDot = filename.lastIndexOf(".");
  return lastDot === -1 ? "" : filename.substring(lastDot);
}

/**
 * Generate a human-readable purpose label
 */
export function formatPurpose(
  purpose:
    | "fine-tune"
    | "assistants"
    | "user_data"
    | "batch"
    | "vision"
    | "evals"
): string {
  switch (purpose) {
    case "fine-tune":
      return "Fine-tuning";
    case "assistants":
      return "Assistants";
    case "user_data":
      return "User Data";
    case "batch":
      return "Batch Processing";
    case "vision":
      return "Vision";
    case "evals":
      return "Evaluations";
    default:
      return purpose;
  }
}

/**
 * Get purpose description for UI help text
 */
export function getPurposeDescription(
  purpose:
    | "fine-tune"
    | "assistants"
    | "user_data"
    | "batch"
    | "vision"
    | "evals"
): string {
  switch (purpose) {
    case "fine-tune":
      return "For training and fine-tuning language models";
    case "assistants":
      return "For use with AI assistants and chat completions";
    case "user_data":
      return "General user data and documents";
    case "batch":
      return "For batch processing and bulk operations";
    case "vision":
      return "For computer vision and image processing tasks";
    case "evals":
      return "For model evaluation and testing";
    default:
      return "General purpose file";
  }
}

/**
 * Format timestamp to human readable date
 */
export function formatTimestamp(timestamp: number): string {
  return new Date(timestamp * 1000).toLocaleString();
}

/**
 * Check if file is text-based for preview
 */
export function isTextFile(mimeType: string): boolean {
  return (
    mimeType.startsWith("text/") ||
    mimeType === "application/json" ||
    mimeType === "text/markdown"
  );
}

/**
 * Create a download link for file content
 */
export function createDownloadUrl(content: string | Blob): string {
  let blob: Blob;

  if (typeof content === "string") {
    blob = new Blob([content], { type: "text/plain" });
  } else {
    blob = content;
  }

  return URL.createObjectURL(blob);
}

/**
 * Trigger file download
 */
export function downloadFile(url: string, filename: string): void {
  const link = document.createElement("a");
  link.href = url;
  link.download = filename;
  document.body.appendChild(link);
  link.click();
  document.body.removeChild(link);
  URL.revokeObjectURL(url);
}

/**
 * Truncate filename for display
 */
export function truncateFilename(
  filename: string,
  maxLength: number = 30
): string {
  if (filename.length <= maxLength) return filename;

  const extension = getFileExtension(filename);
  const nameWithoutExt = filename.substring(
    0,
    filename.length - extension.length
  );
  const truncatedName = nameWithoutExt.substring(
    0,
    maxLength - extension.length - 3
  );

  return `${truncatedName}...${extension}`;
}
