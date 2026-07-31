import {
  formatFileSize,
  getFileTypeIcon,
  formatPurpose,
  getPurposeDescription,
  formatTimestamp,
  truncateFilename,
  isTextFile,
  createDownloadUrl,
} from "./file-utils";

describe("file-utils", () => {
  describe("formatFileSize", () => {
    test("formats bytes correctly", () => {
      expect(formatFileSize(0)).toBe("0 B");
      expect(formatFileSize(500)).toBe("500 B");
      expect(formatFileSize(1023)).toBe("1023 B");
    });

    test("formats kilobytes correctly", () => {
      expect(formatFileSize(1024)).toBe("1.0 KB");
      expect(formatFileSize(1536)).toBe("1.5 KB");
      expect(formatFileSize(1024 * 1023)).toBe("1023.0 KB");
    });

    test("formats megabytes correctly", () => {
      expect(formatFileSize(1024 * 1024)).toBe("1.0 MB");
      expect(formatFileSize(1024 * 1024 * 2.5)).toBe("2.5 MB");
    });

    test("formats gigabytes correctly", () => {
      expect(formatFileSize(1024 * 1024 * 1024)).toBe("1.0 GB");
      expect(formatFileSize(1024 * 1024 * 1024 * 1.8)).toBe("1.8 GB");
    });
  });

  describe("getFileTypeIcon", () => {
    test("returns correct icons for common file types", () => {
      expect(getFileTypeIcon("pdf")).toBe("📕");
      expect(getFileTypeIcon(".pdf")).toBe("📕");
      expect(getFileTypeIcon("txt")).toBe("📄");
      expect(getFileTypeIcon("html")).toBe("🌐");
      expect(getFileTypeIcon("md")).toBe("📝");
      expect(getFileTypeIcon("csv")).toBe("📊");
      expect(getFileTypeIcon("json")).toBe("⚙️");
      expect(getFileTypeIcon("docx")).toBe("📘");
      expect(getFileTypeIcon("js")).toBe("⚡");
      expect(getFileTypeIcon("py")).toBe("🐍");
    });

    test("handles MIME types", () => {
      expect(getFileTypeIcon("application/pdf")).toBe("📕");
      expect(getFileTypeIcon("text/plain")).toBe("📄");
      expect(getFileTypeIcon("text/markdown")).toBe("📝");
      expect(getFileTypeIcon("application/json")).toBe("⚙️");
      expect(getFileTypeIcon("text/html")).toBe("🌐");
    });

    test("returns default icon for unknown types", () => {
      expect(getFileTypeIcon("unknown")).toBe("📄");
      expect(getFileTypeIcon("")).toBe("📄");
      expect(getFileTypeIcon(undefined)).toBe("📄");
    });

    test("handles complex MIME types", () => {
      expect(
        getFileTypeIcon(
          "application/vnd.openxmlformats-officedocument.wordprocessingml.document"
        )
      ).toBe("📘");
    });
  });

  describe("formatPurpose", () => {
    test("formats purpose labels correctly", () => {
      expect(formatPurpose("fine-tune")).toBe("Fine-tuning");
      expect(formatPurpose("assistants")).toBe("Assistants");
      expect(formatPurpose("user_data")).toBe("User Data");
      expect(formatPurpose("batch")).toBe("Batch Processing");
      expect(formatPurpose("vision")).toBe("Vision");
      expect(formatPurpose("evals")).toBe("Evaluations");
    });
  });

  describe("getPurposeDescription", () => {
    test("returns correct descriptions", () => {
      expect(getPurposeDescription("fine-tune")).toBe(
        "For training and fine-tuning language models"
      );
      expect(getPurposeDescription("assistants")).toBe(
        "For use with AI assistants and chat completions"
      );
      expect(getPurposeDescription("user_data")).toBe(
        "General user data and documents"
      );
      expect(getPurposeDescription("batch")).toBe(
        "For batch processing and bulk operations"
      );
      expect(getPurposeDescription("vision")).toBe(
        "For computer vision and image processing tasks"
      );
      expect(getPurposeDescription("evals")).toBe(
        "For model evaluation and testing"
      );
    });
  });

  describe("formatTimestamp", () => {
    test("formats Unix timestamp to readable date", () => {
      const timestamp = 1640995200; // Jan 1, 2022 00:00:00 UTC
      const result = formatTimestamp(timestamp);

      // Should return a valid date string
      expect(typeof result).toBe("string");
      expect(result.length).toBeGreaterThan(0);

      // Test that it's calling Date correctly by using a known timestamp
      const testDate = new Date(timestamp * 1000);
      expect(result).toBe(testDate.toLocaleString());
    });
  });

  describe("truncateFilename", () => {
    test("returns filename unchanged if under max length", () => {
      expect(truncateFilename("short.txt", 30)).toBe("short.txt");
    });

    test("truncates long filenames while preserving extension", () => {
      const longFilename =
        "this_is_a_very_long_filename_that_should_be_truncated.pdf";
      const result = truncateFilename(longFilename, 30);

      expect(result).toContain("...");
      expect(result).toContain(".pdf");
      expect(result.length).toBeLessThanOrEqual(30);
    });

    test("handles files without extension", () => {
      const longFilename = "this_is_a_very_long_filename_without_extension";
      const result = truncateFilename(longFilename, 20);

      expect(result).toContain("...");
      expect(result.length).toBeLessThanOrEqual(20);
    });

    test("uses default max length", () => {
      const longFilename = "a".repeat(50) + ".txt";
      const result = truncateFilename(longFilename);

      expect(result).toContain("...");
      expect(result).toContain(".txt");
      expect(result.length).toBeLessThanOrEqual(30); // Default max length
    });
  });

  describe("isTextFile", () => {
    test("identifies text files correctly", () => {
      expect(isTextFile("text/plain")).toBe(true);
      expect(isTextFile("text/html")).toBe(true);
      expect(isTextFile("text/markdown")).toBe(true);
      expect(isTextFile("application/json")).toBe(true);
    });

    test("identifies non-text files correctly", () => {
      expect(isTextFile("application/pdf")).toBe(false);
      expect(isTextFile("image/png")).toBe(false);
      expect(isTextFile("application/octet-stream")).toBe(false);
    });
  });

  describe("createDownloadUrl", () => {
    // Mock URL.createObjectURL and revokeObjectURL for Node environment
    const mockCreateObjectURL = jest.fn(() => "blob:mock-url");
    const mockRevokeObjectURL = jest.fn();

    beforeAll(() => {
      global.URL = {
        ...global.URL,
        createObjectURL: mockCreateObjectURL,
        revokeObjectURL: mockRevokeObjectURL,
      } as typeof URL;
    });

    beforeEach(() => {
      mockCreateObjectURL.mockClear();
      mockRevokeObjectURL.mockClear();
    });

    test("creates download URL from string content", () => {
      const content = "Hello, world!";
      const url = createDownloadUrl(content);

      expect(url).toBe("blob:mock-url");
      expect(mockCreateObjectURL).toHaveBeenCalledWith(expect.any(Blob));
    });

    test("creates download URL from Blob", () => {
      const blob = new Blob(["Hello, world!"], { type: "text/plain" });
      const url = createDownloadUrl(blob);

      expect(url).toBe("blob:mock-url");
      expect(mockCreateObjectURL).toHaveBeenCalledWith(blob);
    });
  });
});
