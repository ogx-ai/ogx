import React from "react";
import {
  render,
  screen,
  fireEvent,
  waitFor,
  act,
} from "@testing-library/react";
import "@testing-library/jest-dom";
import ContentsListPage from "./page";
import { VectorStoreContentItem } from "@/lib/contents-api";
import type { VectorStore } from "llama-stack-client/resources/vector-stores/vector-stores";
import type { VectorStoreFile } from "llama-stack-client/resources/vector-stores/files";

const mockPush = jest.fn();
const mockParams = {
  id: "vs_123",
  fileId: "file_456",
};

jest.mock("next/navigation", () => ({
  useParams: () => mockParams,
  useRouter: () => ({
    push: mockPush,
  }),
}));

const mockClient = {
  vectorStores: {
    retrieve: jest.fn(),
    files: {
      retrieve: jest.fn(),
    },
  },
};

jest.mock("@/hooks/use-auth-client", () => ({
  useAuthClient: () => mockClient,
}));

const mockContentsAPI = {
  listContents: jest.fn(),
  deleteContent: jest.fn(),
};

jest.mock("@/lib/contents-api", () => ({
  ContentsAPI: jest.fn(() => mockContentsAPI),
}));

describe("ContentsListPage", () => {
  const mockStore: VectorStore = {
    id: "vs_123",
    name: "Test Vector Store",
    created_at: 1710000000,
    status: "ready",
    file_counts: { total: 5 },
    usage_bytes: 1024,
    metadata: {
      provider_id: "test_provider",
    },
  };

  const mockFile: VectorStoreFile = {
    id: "file_456",
    status: "completed",
    created_at: 1710001000,
    usage_bytes: 512,
    chunking_strategy: { type: "fixed_size" },
  };

  const mockContents: VectorStoreContentItem[] = [
    {
      id: "content_1",
      object: "vector_store.content",
      content: "First piece of content for testing.",
      embedding: [0.1, 0.2, 0.3, 0.4, 0.5],
      metadata: {
        chunk_window: "0-35",
        content_length: 35,
      },
      created_timestamp: 1710002000,
    },
    {
      id: "content_2",
      object: "vector_store.content",
      content:
        "Second piece of content with longer text for testing truncation and display.",
      embedding: [0.6, 0.7, 0.8],
      metadata: {
        chunk_window: "36-95",
        content_length: 85,
      },
      created_timestamp: 1710003000,
    },
    {
      id: "content_3",
      object: "vector_store.content",
      content: "Third content without embedding.",
      embedding: undefined,
      metadata: {
        content_length: 33,
      },
      created_timestamp: 1710004000,
    },
  ];

  beforeEach(() => {
    jest.clearAllMocks();

    mockClient.vectorStores.retrieve.mockResolvedValue(mockStore);
    mockClient.vectorStores.files.retrieve.mockResolvedValue(mockFile);
    mockContentsAPI.listContents.mockResolvedValue({
      data: mockContents,
    });
  });

  describe("Loading and Error States", () => {
    test("renders loading skeleton while fetching store data", async () => {
      mockClient.vectorStores.retrieve.mockImplementation(
        () => new Promise(() => {})
      );

      await act(async () => {
        render(<ContentsListPage />);
      });

      const skeletons = document.querySelectorAll('[data-slot="skeleton"]');
      expect(skeletons.length).toBeGreaterThan(0);
    });

    test("renders error message when store API call fails", async () => {
      const error = new Error("Failed to load store");
      mockClient.vectorStores.retrieve.mockRejectedValue(error);

      await act(async () => {
        render(<ContentsListPage />);
      });

      await waitFor(() => {
        expect(
          screen.getByText(/Error loading details for ID vs_123/)
        ).toBeInTheDocument();
        expect(screen.getByText(/Failed to load store/)).toBeInTheDocument();
      });
    });

    test("renders not found when store doesn't exist", async () => {
      mockClient.vectorStores.retrieve.mockResolvedValue(null);

      await act(async () => {
        render(<ContentsListPage />);
      });

      await waitFor(() => {
        expect(
          screen.getByText(/No details found for ID: vs_123/)
        ).toBeInTheDocument();
      });
    });

    test("renders contents loading skeleton", async () => {
      mockContentsAPI.listContents.mockImplementation(
        () => new Promise(() => {})
      );

      const { container } = render(<ContentsListPage />);

      await waitFor(() => {
        expect(
          screen.getByText("Contents in File: file_456")
        ).toBeInTheDocument();
      });

      const skeletons = container.querySelectorAll('[data-slot="skeleton"]');
      expect(skeletons.length).toBeGreaterThan(0);
    });

    test("renders contents error message", async () => {
      const error = new Error("Failed to load contents");
      mockContentsAPI.listContents.mockRejectedValue(error);

      render(<ContentsListPage />);

      await waitFor(() => {
        expect(
          screen.getByText("Error loading contents: Failed to load contents")
        ).toBeInTheDocument();
      });
    });
  });

  describe("Contents Table Display", () => {
    test("renders contents table with correct headers", async () => {
      render(<ContentsListPage />);

      await waitFor(() => {
        expect(screen.getByText("Content Chunks (3)")).toBeInTheDocument();
        expect(screen.getByText("Contents in this file")).toBeInTheDocument();
      });

      // Check table headers
      expect(screen.getByText("Content ID")).toBeInTheDocument();
      expect(screen.getByText("Content Preview")).toBeInTheDocument();
      expect(screen.getByText("Embedding")).toBeInTheDocument();
      expect(screen.getByText("Position")).toBeInTheDocument();
      expect(screen.getByText("Created")).toBeInTheDocument();
      expect(screen.getByText("Actions")).toBeInTheDocument();
    });

    test("renders content data correctly", async () => {
      render(<ContentsListPage />);

      await waitFor(() => {
        // Check first content row
        expect(screen.getByText("content_1...")).toBeInTheDocument();
        expect(
          screen.getByText("First piece of content for testing.")
        ).toBeInTheDocument();
        expect(
          screen.getByText("[0.100, 0.200, 0.300...] (5D)")
        ).toBeInTheDocument();
        expect(screen.getByText("0-35")).toBeInTheDocument();
        expect(
          screen.getByText(new Date(1710002000 * 1000).toLocaleString())
        ).toBeInTheDocument();

        expect(screen.getByText("content_2...")).toBeInTheDocument();
        expect(
          screen.getByText(/Second piece of content with longer text/)
        ).toBeInTheDocument();
        expect(
          screen.getByText("[0.600, 0.700, 0.800...] (3D)")
        ).toBeInTheDocument();
        expect(screen.getByText("36-95")).toBeInTheDocument();

        expect(screen.getByText("content_3...")).toBeInTheDocument();
        expect(
          screen.getByText("Third content without embedding.")
        ).toBeInTheDocument();
        expect(screen.getByText("No embedding")).toBeInTheDocument();
        expect(screen.getByText("33 chars")).toBeInTheDocument();
      });
    });

    test("handles empty contents list", async () => {
      mockContentsAPI.listContents.mockResolvedValue({
        data: [],
      });

      render(<ContentsListPage />);

      await waitFor(() => {
        expect(screen.getByText("Content Chunks (0)")).toBeInTheDocument();
        expect(
          screen.getByText("No contents found for this file.")
        ).toBeInTheDocument();
      });
    });

    test("truncates long content IDs", async () => {
      const longIdContent = {
        ...mockContents[0],
        id: "very_long_content_id_that_should_be_truncated_123456789",
      };

      mockContentsAPI.listContents.mockResolvedValue({
        data: [longIdContent],
      });

      render(<ContentsListPage />);

      await waitFor(() => {
        expect(screen.getByText("very_long_...")).toBeInTheDocument();
      });
    });
  });

  describe("Content Navigation", () => {
    test("navigates to content detail when content ID is clicked", async () => {
      render(<ContentsListPage />);

      await waitFor(() => {
        expect(screen.getByText("content_1...")).toBeInTheDocument();
      });

      const contentLink = screen.getByRole("button", { name: "content_1..." });
      fireEvent.click(contentLink);

      expect(mockPush).toHaveBeenCalledWith(
        "/logs/vector-stores/vs_123/files/file_456/contents/content_1"
      );
    });

    test("navigates to content detail when view button is clicked", async () => {
      render(<ContentsListPage />);

      await waitFor(() => {
        expect(screen.getByText("Content Chunks (3)")).toBeInTheDocument();
      });

      const viewButtons = screen.getAllByTitle("View content details");
      fireEvent.click(viewButtons[0]);

      expect(mockPush).toHaveBeenCalledWith(
        "/logs/vector-stores/vs_123/files/file_456/contents/content_1"
      );
    });

    test("navigates to content detail when edit button is clicked", async () => {
      render(<ContentsListPage />);

      await waitFor(() => {
        expect(screen.getByText("Content Chunks (3)")).toBeInTheDocument();
      });

      const editButtons = screen.getAllByTitle("Edit content");
      fireEvent.click(editButtons[0]);

      expect(mockPush).toHaveBeenCalledWith(
        "/logs/vector-stores/vs_123/files/file_456/contents/content_1"
      );
    });
  });

  describe("Content Deletion", () => {
    test("deletes content when delete button is clicked", async () => {
      mockContentsAPI.deleteContent.mockResolvedValue(undefined);

      render(<ContentsListPage />);

      await waitFor(() => {
        expect(screen.getByText("Content Chunks (3)")).toBeInTheDocument();
      });

      const deleteButtons = screen.getAllByTitle("Delete content");
      fireEvent.click(deleteButtons[0]);

      await waitFor(() => {
        expect(mockContentsAPI.deleteContent).toHaveBeenCalledWith(
          "vs_123",
          "file_456",
          "content_1"
        );
      });

      await waitFor(() => {
        expect(screen.getByText("Content Chunks (2)")).toBeInTheDocument();
      });

      expect(screen.queryByText("content_1...")).not.toBeInTheDocument();
    });

    test("handles delete error gracefully", async () => {
      const consoleError = jest
        .spyOn(console, "error")
        .mockImplementation(() => {});
      mockContentsAPI.deleteContent.mockRejectedValue(
        new Error("Delete failed")
      );

      render(<ContentsListPage />);

      await waitFor(() => {
        expect(screen.getByText("Content Chunks (3)")).toBeInTheDocument();
      });

      const deleteButtons = screen.getAllByTitle("Delete content");
      fireEvent.click(deleteButtons[0]);

      await waitFor(() => {
        expect(consoleError).toHaveBeenCalledWith(
          "Failed to delete content:",
          expect.any(Error)
        );
      });

      expect(screen.getByText("Content Chunks (3)")).toBeInTheDocument();
      expect(screen.getByText("content_1...")).toBeInTheDocument();

      consoleError.mockRestore();
    });
  });

  describe("Breadcrumb Navigation", () => {
    test("renders correct breadcrumb structure", async () => {
      render(<ContentsListPage />);

      await waitFor(() => {
        const vectorStoreTexts = screen.getAllByText("Vector Stores");
        expect(vectorStoreTexts.length).toBeGreaterThan(0);
        const storeNameTexts = screen.getAllByText("Test Vector Store");
        expect(storeNameTexts.length).toBeGreaterThan(0);
        const filesTexts = screen.getAllByText("Files");
        expect(filesTexts.length).toBeGreaterThan(0);
        const fileIdTexts = screen.getAllByText("file_456");
        expect(fileIdTexts.length).toBeGreaterThan(0);
        const contentsTexts = screen.getAllByText("Contents");
        expect(contentsTexts.length).toBeGreaterThan(0);
      });
    });
  });

  describe("Sidebar Properties", () => {
    test("renders file and store properties", async () => {
      render(<ContentsListPage />);

      await waitFor(() => {
        const fileIdTexts = screen.getAllByText("file_456");
        expect(fileIdTexts.length).toBeGreaterThan(0);
        const storeIdTexts = screen.getAllByText("vs_123");
        expect(storeIdTexts.length).toBeGreaterThan(0);
        const storeNameTexts = screen.getAllByText("Test Vector Store");
        expect(storeNameTexts.length).toBeGreaterThan(0);

        expect(screen.getByText("completed")).toBeInTheDocument();
        expect(screen.getByText("512")).toBeInTheDocument();
        expect(screen.getByText("fixed_size")).toBeInTheDocument();
        expect(screen.getByText("test_provider")).toBeInTheDocument();
      });
    });
  });

  describe("Content Text Utilities", () => {
    test("handles different content formats correctly", async () => {
      const contentWithObject = {
        ...mockContents[0],
        content: { type: "text", text: "Object format content" },
      };

      mockContentsAPI.listContents.mockResolvedValue({
        data: [contentWithObject],
      });

      render(<ContentsListPage />);

      await waitFor(() => {
        expect(screen.getByText("Object format content")).toBeInTheDocument();
      });
    });

    test("handles string content format", async () => {
      const contentWithString = {
        ...mockContents[0],
        content: "String format content",
      };

      mockContentsAPI.listContents.mockResolvedValue({
        data: [contentWithString],
      });

      render(<ContentsListPage />);

      await waitFor(() => {
        expect(screen.getByText("String format content")).toBeInTheDocument();
      });
    });

    test("handles unknown content format", async () => {
      const contentWithUnknown = {
        ...mockContents[0],
        content: { unknown: "format" },
      };

      mockContentsAPI.listContents.mockResolvedValue({
        data: [contentWithUnknown],
      });

      render(<ContentsListPage />);

      await waitFor(() => {
        expect(screen.getByText("Content Chunks (1)")).toBeInTheDocument();
      });

      const contentCells = screen.getAllByRole("cell");
      const contentPreviewCell = contentCells.find(cell =>
        cell.querySelector("p[title]")
      );
      expect(contentPreviewCell?.querySelector("p")?.textContent).toBe("");
    });
  });
});
