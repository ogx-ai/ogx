import React from "react";
import { render, screen, fireEvent, waitFor } from "@testing-library/react";
import "@testing-library/jest-dom";
import { FilesManagement } from "./files-management";
import type { FileResource } from "@/lib/types";

// Mock the auth client
const mockFilesClient = {
  list: jest.fn(),
  delete: jest.fn(),
  content: jest.fn(),
};

jest.mock("@/hooks/use-auth-client", () => ({
  useAuthClient: () => ({
    files: mockFilesClient,
  }),
}));

// Mock router
const mockPush = jest.fn();
jest.mock("next/navigation", () => ({
  useRouter: () => ({
    push: mockPush,
  }),
}));

// Mock pagination hook
const mockLoadMore = jest.fn();
const mockRefetch = jest.fn();
jest.mock("@/hooks/use-pagination", () => ({
  usePagination: jest.fn(),
}));

import { usePagination } from "@/hooks/use-pagination";
const mockedUsePagination = usePagination as jest.MockedFunction<
  typeof usePagination
>;

// Mock file utils
jest.mock("@/lib/file-utils", () => ({
  formatFileSize: jest.fn(bytes => `${bytes} B`),
  getFileTypeIcon: jest.fn(() => "📄"),
  formatTimestamp: jest.fn(ts => `2024-01-01 ${ts}`),
  formatPurpose: jest.fn(purpose => purpose),
  truncateFilename: jest.fn(name => name),
  getPurposeDescription: jest.fn(purpose => `Description for ${purpose}`),
}));

// Mock window.confirm
const originalConfirm = window.confirm;

describe("FilesManagement", () => {
  const mockFiles: FileResource[] = [
    {
      id: "file_123",
      filename: "test.pdf",
      bytes: 1024,
      created_at: 1640995200,
      expires_at: 1640995200 + 86400,
      purpose: "assistants",
    },
    {
      id: "file_456",
      filename: "document.txt",
      bytes: 2048,
      created_at: 1640995200,
      expires_at: 0,
      purpose: "user_data",
    },
  ];

  beforeEach(() => {
    jest.clearAllMocks();
    mockPush.mockClear();
    mockLoadMore.mockClear();
    mockRefetch.mockClear();
    window.confirm = originalConfirm;
  });

  describe("Loading State", () => {
    test("renders loading skeleton when status is loading", () => {
      mockedUsePagination.mockReturnValue({
        data: [],
        status: "loading",
        hasMore: false,
        error: null,
        loadMore: mockLoadMore,
        refetch: mockRefetch,
      });

      const { container } = render(<FilesManagement />);
      const skeletons = container.querySelectorAll('[data-slot="skeleton"]');
      expect(skeletons.length).toBeGreaterThan(0);
    });
  });

  describe("Empty State", () => {
    test("renders empty state when no files", () => {
      mockedUsePagination.mockReturnValue({
        data: [],
        status: "idle",
        hasMore: false,
        error: null,
        loadMore: mockLoadMore,
        refetch: mockRefetch,
      });

      render(<FilesManagement />);
      expect(screen.getByText("No files found.")).toBeInTheDocument();
      expect(screen.getByText("Upload Your First File")).toBeInTheDocument();
    });
  });

  describe("Error State", () => {
    test("renders error message when API fails", () => {
      const error = new Error("API Error");
      mockedUsePagination.mockReturnValue({
        data: [],
        status: "error",
        hasMore: false,
        error,
        loadMore: mockLoadMore,
        refetch: mockRefetch,
      });

      render(<FilesManagement />);
      expect(screen.getByText(/Error: API Error/)).toBeInTheDocument();
    });
  });

  describe("Files List", () => {
    beforeEach(() => {
      mockedUsePagination.mockReturnValue({
        data: mockFiles,
        status: "idle",
        hasMore: false,
        error: null,
        loadMore: mockLoadMore,
        refetch: mockRefetch,
      });
    });

    test("renders files table with correct data", () => {
      render(<FilesManagement />);

      expect(screen.getByText("file_123")).toBeInTheDocument();
      expect(screen.getByText("test.pdf")).toBeInTheDocument();
      expect(screen.getByText("file_456")).toBeInTheDocument();
      expect(screen.getByText("document.txt")).toBeInTheDocument();
    });

    test("filters files by search term", () => {
      render(<FilesManagement />);

      const searchInput = screen.getByPlaceholderText("Search files...");
      fireEvent.change(searchInput, { target: { value: "test" } });

      expect(screen.getByText("test.pdf")).toBeInTheDocument();
      expect(screen.queryByText("document.txt")).not.toBeInTheDocument();
    });

    test("navigates to file detail on row click", () => {
      render(<FilesManagement />);

      const row = screen.getByText("test.pdf").closest("tr");
      fireEvent.click(row!);

      expect(mockPush).toHaveBeenCalledWith("/logs/files/file_123");
    });
  });

  describe("File Operations", () => {
    beforeEach(() => {
      mockedUsePagination.mockReturnValue({
        data: mockFiles,
        status: "idle",
        hasMore: false,
        error: null,
        loadMore: mockLoadMore,
        refetch: mockRefetch,
      });
    });

    test("deletes file with confirmation", async () => {
      window.confirm = jest.fn(() => true);
      mockFilesClient.delete.mockResolvedValue({ deleted: true });

      render(<FilesManagement />);

      const deleteButtons = screen.getAllByTitle("Delete file");
      fireEvent.click(deleteButtons[0]);

      expect(window.confirm).toHaveBeenCalledWith(
        "Are you sure you want to delete this file? This action cannot be undone."
      );

      await waitFor(() => {
        expect(mockFilesClient.delete).toHaveBeenCalledWith("file_123");
      });

      await waitFor(() => {
        expect(mockRefetch).toHaveBeenCalled();
      });
    });

    test("cancels delete when user declines confirmation", () => {
      window.confirm = jest.fn(() => false);

      render(<FilesManagement />);

      const deleteButtons = screen.getAllByTitle("Delete file");
      fireEvent.click(deleteButtons[0]);

      expect(window.confirm).toHaveBeenCalled();
      expect(mockFilesClient.delete).not.toHaveBeenCalled();
    });
  });

  describe("Upload Modal", () => {
    test("opens upload modal when upload button clicked", () => {
      mockedUsePagination.mockReturnValue({
        data: mockFiles,
        status: "idle",
        hasMore: false,
        error: null,
        loadMore: mockLoadMore,
        refetch: mockRefetch,
      });

      render(<FilesManagement />);

      const uploadButton = screen.getByText("Upload File");
      fireEvent.click(uploadButton);

      expect(
        screen.getByRole("heading", { name: "Upload File" })
      ).toBeInTheDocument();
      expect(screen.getByText("Select Files")).toBeInTheDocument();
    });

    test("closes modal on ESC key", () => {
      mockedUsePagination.mockReturnValue({
        data: [],
        status: "idle",
        hasMore: false,
        error: null,
        loadMore: mockLoadMore,
        refetch: mockRefetch,
      });

      render(<FilesManagement />);

      // Open modal first
      const uploadButton = screen.getByText("Upload Your First File");
      fireEvent.click(uploadButton);

      // Press ESC key
      fireEvent.keyDown(document, { key: "Escape" });

      // Modal should be closed - the upload form shouldn't be visible
      expect(screen.queryByText("Select Files")).not.toBeInTheDocument();
    });
  });
});
