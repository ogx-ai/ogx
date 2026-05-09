import React from "react";
import { render, screen, fireEvent } from "@testing-library/react";
import "@testing-library/jest-dom";
import { FileUploadZone } from "./file-upload-zone";
import {
  validateFileForUpload,
  detectPotentialCorruption,
  formatValidationErrors,
} from "@/lib/file-validation";

// Mock file utils
jest.mock("@/lib/file-utils", () => ({
  formatFileSize: jest.fn(bytes => `${bytes} B`),
  getFileTypeIcon: jest.fn(() => "📄"),
}));

// Mock file validation
jest.mock("@/lib/file-validation", () => ({
  validateFileForUpload: jest.fn(),
  detectPotentialCorruption: jest.fn(),
  formatValidationErrors: jest.fn(),
}));

// Mock window.alert and confirm
const originalAlert = window.alert;
const originalConfirm = window.confirm;

describe("FileUploadZone", () => {
  const mockOnFilesSelected = jest.fn();
  const mockOnRemoveFile = jest.fn();
  const selectedFiles = [
    new File(["content"], "test.txt", { type: "text/plain" }),
  ];

  const defaultProps = {
    onFilesSelected: mockOnFilesSelected,
    selectedFiles: [],
    onRemoveFile: mockOnRemoveFile,
    maxFiles: 10,
  };

  beforeEach(() => {
    jest.clearAllMocks();
    window.alert = jest.fn();
    window.confirm = jest.fn(() => true);
    (validateFileForUpload as jest.Mock).mockReturnValue({
      isValid: true,
      errors: [],
      warnings: [],
    });
    (detectPotentialCorruption as jest.Mock).mockReturnValue([]);
    (formatValidationErrors as jest.Mock).mockReturnValue("");
  });

  afterEach(() => {
    window.alert = originalAlert;
    window.confirm = originalConfirm;
  });

  describe("Drag and Drop", () => {
    test("updates drag state on drag enter", () => {
      render(<FileUploadZone {...defaultProps} />);

      const dropzone = screen.getByText("Click to upload or drag and drop");
      const dragZoneDiv = dropzone.closest('[class*="border-2"]');

      fireEvent.dragEnter(dragZoneDiv!, {
        dataTransfer: { files: [] },
      });

      // Should show drag over state
      expect(dragZoneDiv).toHaveClass("border-blue-500");
    });

    test("handles file drop correctly", () => {
      const files = [
        new File(["content"], "test.pdf", { type: "application/pdf" }),
      ];

      render(<FileUploadZone {...defaultProps} />);

      const dropzone = screen.getByText("Click to upload or drag and drop");

      fireEvent.drop(dropzone, {
        dataTransfer: { files },
      });

      expect(mockOnFilesSelected).toHaveBeenCalledWith([files[0]]);
    });

    test("prevents drop when disabled", () => {
      render(<FileUploadZone {...defaultProps} disabled={true} />);

      const dropzone = screen.getByText("Click to upload or drag and drop");

      fireEvent.drop(dropzone, {
        dataTransfer: {
          files: [new File(["content"], "test.txt", { type: "text/plain" })],
        },
      });

      expect(mockOnFilesSelected).not.toHaveBeenCalled();
    });
  });

  describe("File Validation", () => {
    test("shows validation errors for invalid files", () => {
      (validateFileForUpload as jest.Mock).mockReturnValue({
        isValid: false,
        errors: [{ field: "size", message: "File too large" }],
        warnings: [],
      });
      (formatValidationErrors as jest.Mock).mockReturnValue("File too large");

      const file = new File(["content"], "large.pdf", {
        type: "application/pdf",
      });
      // Mock large size
      Object.defineProperty(file, "size", {
        value: 200 * 1024 * 1024,
        writable: false,
      });
      const files = [file];

      render(<FileUploadZone {...defaultProps} />);

      const dropzone = screen.getByText("Click to upload or drag and drop");

      fireEvent.drop(dropzone, {
        dataTransfer: { files },
      });

      expect(window.alert).toHaveBeenCalledWith(
        expect.stringContaining("File too large")
      );
      expect(mockOnFilesSelected).not.toHaveBeenCalled();
    });

    test("shows warnings but allows upload", () => {
      (validateFileForUpload as jest.Mock).mockReturnValue({
        isValid: true,
        errors: [],
        warnings: ["Large file warning"],
      });

      const files = [
        new File(["content"], "test.pdf", { type: "application/pdf" }),
      ];

      render(<FileUploadZone {...defaultProps} />);

      const dropzone = screen.getByText("Click to upload or drag and drop");

      fireEvent.drop(dropzone, {
        dataTransfer: { files },
      });

      expect(window.confirm).toHaveBeenCalled();
      expect(mockOnFilesSelected).toHaveBeenCalledWith([files[0]]);
    });

    test("prevents duplicate files", () => {
      const existingFile = new File(["content"], "test.txt", {
        type: "text/plain",
      });
      const duplicateFile = new File(["content"], "test.txt", {
        type: "text/plain",
      });

      render(
        <FileUploadZone {...defaultProps} selectedFiles={[existingFile]} />
      );

      const dropzone = screen.getByText("Click to upload or drag and drop");

      fireEvent.drop(dropzone, {
        dataTransfer: { files: [duplicateFile] },
      });

      expect(window.alert).toHaveBeenCalledWith(
        expect.stringContaining('"test.txt" is already selected')
      );
      expect(mockOnFilesSelected).not.toHaveBeenCalled();
    });

    test("enforces max files limit", () => {
      const existingFiles = Array.from(
        { length: 9 },
        (_, i) => new File(["content"], `file${i}.txt`, { type: "text/plain" })
      );

      const newFiles = [
        new File(["content"], "file10.txt", { type: "text/plain" }),
        new File(["content"], "file11.txt", { type: "text/plain" }),
      ];

      render(
        <FileUploadZone
          {...defaultProps}
          selectedFiles={existingFiles}
          maxFiles={10}
        />
      );

      const dropzone = screen.getByText("Click to upload or drag and drop");

      fireEvent.drop(dropzone, {
        dataTransfer: { files: newFiles },
      });

      expect(window.alert).toHaveBeenCalledWith(
        expect.stringContaining("You can only upload 1 more file(s)")
      );
      expect(mockOnFilesSelected).not.toHaveBeenCalled();
    });
  });

  describe("Selected Files Display", () => {
    test("displays selected files with remove buttons", () => {
      render(
        <FileUploadZone {...defaultProps} selectedFiles={selectedFiles} />
      );

      expect(screen.getByText("test.txt")).toBeInTheDocument();
      expect(screen.getByRole("button", { name: "" })).toBeInTheDocument();
    });

    test("calls onRemoveFile when remove button clicked", () => {
      render(
        <FileUploadZone {...defaultProps} selectedFiles={selectedFiles} />
      );

      const removeButton = screen.getByRole("button", { name: "" });
      fireEvent.click(removeButton);

      expect(mockOnRemoveFile).toHaveBeenCalledWith(0);
    });
  });

  describe("File Selection via Click", () => {
    test("creates file input when zone clicked", () => {
      // Mock document.createElement to track input creation
      const mockInput = {
        type: "",
        multiple: false,
        accept: "",
        onchange: null as ((event: Event) => void) | null,
        click: jest.fn(),
      };

      const originalCreateElement = document.createElement;
      document.createElement = jest.fn((tagName: string) => {
        if (tagName === "input") {
          return mockInput as unknown as HTMLInputElement;
        }
        return originalCreateElement.call(document, tagName);
      });

      render(<FileUploadZone {...defaultProps} />);

      const clickArea = screen.getByText("Click to upload or drag and drop");
      fireEvent.click(clickArea);

      expect(document.createElement).toHaveBeenCalledWith("input");
      expect(mockInput.type).toBe("file");
      expect(mockInput.click).toHaveBeenCalled();

      document.createElement = originalCreateElement;
    });
  });

  describe("Disabled State", () => {
    test("prevents interaction when disabled", () => {
      render(<FileUploadZone {...defaultProps} disabled={true} />);

      const clickArea = screen.getByText("Click to upload or drag and drop");
      const dragZoneDiv = clickArea.closest('[class*="border-2"]');

      // Should not be clickable when disabled
      expect(dragZoneDiv).toHaveClass("cursor-not-allowed");
      expect(dragZoneDiv).toHaveClass("opacity-50");

      fireEvent.click(clickArea);
      // File selection should not work when disabled - hard to test the preventDefault
      // but the component should be visually disabled
    });
  });
});
