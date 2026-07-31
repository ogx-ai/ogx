import {
  validateFileForUpload,
  validateUploadParams,
  formatValidationErrors,
  formatValidationWarnings,
  detectPotentialCorruption,
} from "./file-validation";
import { MAX_FILE_SIZE } from "./types";

describe("file-validation", () => {
  describe("validateFileForUpload", () => {
    test("validates a correct file", () => {
      const file = new File(["content"], "test.pdf", {
        type: "application/pdf",
      });
      const result = validateFileForUpload(file);

      expect(result.isValid).toBe(true);
      expect(result.errors).toHaveLength(0);
    });

    test("rejects file that is too large", () => {
      // Create a mock file that appears large without actually creating huge content
      const file = new File(["small content"], "large.pdf", {
        type: "application/pdf",
      });
      // Override size property
      Object.defineProperty(file, "size", {
        value: MAX_FILE_SIZE + 1,
        writable: false,
      });

      const result = validateFileForUpload(file);

      expect(result.isValid).toBe(false);
      expect(result.errors).toEqual(
        expect.arrayContaining([
          expect.objectContaining({
            field: "size",
            message: expect.stringContaining("exceeds maximum limit"),
          }),
        ])
      );
    });

    test("rejects empty file", () => {
      const file = new File([], "empty.pdf", { type: "application/pdf" });
      const result = validateFileForUpload(file);

      expect(result.isValid).toBe(false);
      expect(result.errors).toEqual(
        expect.arrayContaining([
          expect.objectContaining({
            field: "size",
            message: "File is empty",
          }),
        ])
      );
    });

    test("rejects unsupported file type", () => {
      const file = new File(["content"], "test.exe", {
        type: "application/exe",
      });
      const result = validateFileForUpload(file);

      expect(result.isValid).toBe(false);
      expect(result.errors).toEqual(
        expect.arrayContaining([
          expect.objectContaining({
            field: "type",
            message: expect.stringContaining("Unsupported file type"),
          }),
        ])
      );
    });

    test("warns about special characters in filename", () => {
      const file = new File(["content"], "test<>file.pdf", {
        type: "application/pdf",
      });
      const result = validateFileForUpload(file);

      expect(result.isValid).toBe(true);
      expect(result.warnings).toEqual(
        expect.arrayContaining([expect.stringContaining("special characters")])
      );
    });

    test("rejects file with very long name", () => {
      const longName = "a".repeat(256) + ".pdf";
      const file = new File(["content"], longName, { type: "application/pdf" });
      const result = validateFileForUpload(file);

      expect(result.isValid).toBe(false);
      expect(result.errors).toEqual(
        expect.arrayContaining([
          expect.objectContaining({
            field: "name",
            message: expect.stringContaining("too long"),
          }),
        ])
      );
    });
  });

  describe("validateUploadParams", () => {
    const validFile = new File(["content"], "test.pdf", {
      type: "application/pdf",
    });

    test("validates correct upload parameters", () => {
      const result = validateUploadParams([validFile], "assistants", 3600);

      expect(result.isValid).toBe(true);
      expect(result.errors).toHaveLength(0);
    });

    test("rejects when no files provided", () => {
      const result = validateUploadParams([], "assistants", 3600);

      expect(result.isValid).toBe(false);
      expect(result.errors).toEqual(
        expect.arrayContaining([
          expect.objectContaining({
            field: "files",
            message: "At least one file is required",
          }),
        ])
      );
    });

    test("rejects too many files", () => {
      const files = Array.from(
        { length: 11 },
        (_, i) => new File(["content"], `file${i}.txt`, { type: "text/plain" })
      );
      const result = validateUploadParams(files, "assistants");

      expect(result.isValid).toBe(false);
      expect(result.errors).toEqual(
        expect.arrayContaining([
          expect.objectContaining({
            field: "files",
            message: expect.stringContaining("Maximum 10 files"),
          }),
        ])
      );
    });

    test("rejects invalid purpose", () => {
      const result = validateUploadParams(
        [validFile],
        "invalid" as
          | "fine-tune"
          | "assistants"
          | "user_data"
          | "batch"
          | "vision"
          | "evals"
      );

      expect(result.isValid).toBe(false);
      expect(result.errors).toEqual(
        expect.arrayContaining([
          expect.objectContaining({
            field: "purpose",
            message: expect.stringContaining("Purpose must be one of"),
          }),
        ])
      );
    });

    test("rejects expiration time too short", () => {
      const result = validateUploadParams([validFile], "assistants", 1800); // 30 minutes

      expect(result.isValid).toBe(false);
      expect(result.errors).toEqual(
        expect.arrayContaining([
          expect.objectContaining({
            field: "expiresAfter",
            message: expect.stringContaining(
              "Minimum expiration time is 1 hour"
            ),
          }),
        ])
      );
    });

    test("rejects expiration time too long", () => {
      const result = validateUploadParams([validFile], "assistants", 2592001); // > 30 days

      expect(result.isValid).toBe(false);
      expect(result.errors).toEqual(
        expect.arrayContaining([
          expect.objectContaining({
            field: "expiresAfter",
            message: expect.stringContaining(
              "Maximum expiration time is 30 days"
            ),
          }),
        ])
      );
    });

    test("warns about duplicate filenames", () => {
      const files = [
        new File(["content1"], "test.txt", { type: "text/plain" }),
        new File(["content2"], "test.txt", { type: "text/plain" }),
      ];
      const result = validateUploadParams(files, "assistants");

      expect(result.isValid).toBe(true);
      expect(result.warnings).toEqual(
        expect.arrayContaining([
          expect.stringContaining('Multiple files with name "test.txt"'),
        ])
      );
    });
  });

  describe("detectPotentialCorruption", () => {
    test("warns about suspiciously small PDF", () => {
      const file = new File(["x"], "small.pdf", { type: "application/pdf" });
      const warnings = detectPotentialCorruption(file);

      expect(warnings).toEqual(
        expect.arrayContaining([
          expect.stringContaining("PDF file appears unusually small"),
        ])
      );
    });

    test("warns about very large text file", () => {
      const file = new File(["content"], "huge.txt", { type: "text/plain" });
      // Override size to appear very large
      Object.defineProperty(file, "size", {
        value: 101 * 1024 * 1024,
        writable: false,
      });

      const warnings = detectPotentialCorruption(file);

      expect(warnings).toEqual(
        expect.arrayContaining([
          expect.stringContaining("Very large text file"),
        ])
      );
    });
  });

  describe("formatValidationErrors", () => {
    test("returns empty string for no errors", () => {
      expect(formatValidationErrors([])).toBe("");
    });

    test("returns single error message", () => {
      const errors = [{ field: "size", message: "File too large" }];
      expect(formatValidationErrors(errors)).toBe("File too large");
    });

    test("formats multiple errors", () => {
      const errors = [
        { field: "size", message: "File too large" },
        { field: "type", message: "Unsupported type" },
      ];
      const result = formatValidationErrors(errors);

      expect(result).toContain("Multiple issues found:");
      expect(result).toContain("• File too large");
      expect(result).toContain("• Unsupported type");
    });
  });

  describe("formatValidationWarnings", () => {
    test("returns empty string for no warnings", () => {
      expect(formatValidationWarnings([])).toBe("");
    });

    test("returns single warning", () => {
      expect(formatValidationWarnings(["Single warning"])).toBe(
        "Single warning"
      );
    });

    test("formats multiple warnings", () => {
      const warnings = ["Warning 1", "Warning 2"];
      const result = formatValidationWarnings(warnings);

      expect(result).toContain("2 warnings:");
      expect(result).toContain("• Warning 1");
      expect(result).toContain("• Warning 2");
    });
  });
});
