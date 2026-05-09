import React from "react";
import { render, screen, fireEvent } from "@testing-library/react";
import "@testing-library/jest-dom";
import { VectorStoreDetailView } from "./vector-store-detail";
import type { VectorStore } from "llama-stack-client/resources/vector-stores/vector-stores";
import type { VectorStoreFile } from "llama-stack-client/resources/vector-stores/files";

const mockPush = jest.fn();
jest.mock("next/navigation", () => ({
  useRouter: () => ({
    push: mockPush,
  }),
}));

// Mock NextAuth
jest.mock("next-auth/react", () => ({
  useSession: () => ({
    data: {
      accessToken: "mock-access-token",
      user: {
        id: "mock-user-id",
        email: "test@example.com",
      },
    },
    status: "authenticated",
  }),
}));

describe("VectorStoreDetailView", () => {
  const defaultProps = {
    store: null,
    files: [],
    isLoadingStore: false,
    isLoadingFiles: false,
    errorStore: null,
    errorFiles: null,
    id: "test_vector_store_id",
  };

  beforeEach(() => {
    mockPush.mockClear();
  });

  describe("Loading States", () => {
    test("renders loading skeleton when store is loading", () => {
      const { container } = render(
        <VectorStoreDetailView {...defaultProps} isLoadingStore={true} />
      );

      const skeletons = container.querySelectorAll('[data-slot="skeleton"]');
      expect(skeletons.length).toBeGreaterThan(0);
    });

    test("renders files loading skeleton when files are loading", () => {
      const mockStore: VectorStore = {
        id: "vs_123",
        name: "Test Vector Store",
        created_at: 1710000000,
        status: "ready",
        file_counts: { total: 5 },
        usage_bytes: 1024,
        metadata: {
          provider_id: "test_provider",
          provider_vector_db_id: "test_db_id",
        },
      };

      const { container } = render(
        <VectorStoreDetailView
          {...defaultProps}
          store={mockStore}
          isLoadingFiles={true}
        />
      );

      expect(screen.getByText("Vector Store Details")).toBeInTheDocument();
      expect(screen.getByText("Files")).toBeInTheDocument();
      const skeletons = container.querySelectorAll('[data-slot="skeleton"]');
      expect(skeletons.length).toBeGreaterThan(0);
    });
  });

  describe("Error States", () => {
    test("renders error message when store error occurs", () => {
      render(
        <VectorStoreDetailView
          {...defaultProps}
          errorStore={{ name: "Error", message: "Failed to load store" }}
        />
      );

      expect(screen.getByText("Vector Store Details")).toBeInTheDocument();
      expect(
        screen.getByText(/Error loading details for ID test_vector_store_id/)
      ).toBeInTheDocument();
      expect(screen.getByText(/Failed to load store/)).toBeInTheDocument();
    });

    test("renders files error when files fail to load", () => {
      const mockStore: VectorStore = {
        id: "vs_123",
        name: "Test Vector Store",
        created_at: 1710000000,
        status: "ready",
        file_counts: { total: 5 },
        usage_bytes: 1024,
        metadata: {
          provider_id: "test_provider",
          provider_vector_db_id: "test_db_id",
        },
      };

      render(
        <VectorStoreDetailView
          {...defaultProps}
          store={mockStore}
          errorFiles={{ name: "Error", message: "Failed to load files" }}
        />
      );

      expect(screen.getByText("Files")).toBeInTheDocument();
      expect(
        screen.getByText("Error loading files: Failed to load files")
      ).toBeInTheDocument();
    });
  });

  describe("Not Found State", () => {
    test("renders not found message when store is null", () => {
      render(<VectorStoreDetailView {...defaultProps} store={null} />);

      expect(screen.getByText("Vector Store Details")).toBeInTheDocument();
      expect(
        screen.getByText(/No details found for ID: test_vector_store_id/)
      ).toBeInTheDocument();
    });
  });

  describe("Store Data Rendering", () => {
    const mockStore: VectorStore = {
      id: "vs_123",
      name: "Test Vector Store",
      created_at: 1710000000,
      status: "ready",
      file_counts: { total: 3 },
      usage_bytes: 2048,
      metadata: {
        provider_id: "test_provider",
        provider_vector_db_id: "test_db_id",
      },
    };

    test("renders store properties correctly", () => {
      render(<VectorStoreDetailView {...defaultProps} store={mockStore} />);

      expect(screen.getByText("Vector Store Details")).toBeInTheDocument();
      expect(screen.getByText("vs_123")).toBeInTheDocument();
      expect(screen.getByText("Test Vector Store")).toBeInTheDocument();
      expect(
        screen.getByText(new Date(1710000000 * 1000).toLocaleString())
      ).toBeInTheDocument();
      expect(screen.getByText("ready")).toBeInTheDocument();
      expect(screen.getByText("3")).toBeInTheDocument();
      expect(screen.getByText("2048")).toBeInTheDocument();
      expect(screen.getByText("test_provider")).toBeInTheDocument();
      expect(screen.getByText("test_db_id")).toBeInTheDocument();
    });

    test("handles empty/missing optional fields", () => {
      const minimalStore: VectorStore = {
        id: "vs_minimal",
        name: "",
        created_at: 1710000000,
        status: "ready",
        file_counts: { total: 0 },
        usage_bytes: 0,
        metadata: {},
      };

      render(<VectorStoreDetailView {...defaultProps} store={minimalStore} />);

      expect(screen.getByText("vs_minimal")).toBeInTheDocument();
      expect(screen.getByText("ready")).toBeInTheDocument();
      const zeroTexts = screen.getAllByText("0");
      expect(zeroTexts.length).toBeGreaterThanOrEqual(2);
    });

    test("shows empty files message when no files", () => {
      render(
        <VectorStoreDetailView {...defaultProps} store={mockStore} files={[]} />
      );

      expect(screen.getByText("Files")).toBeInTheDocument();
      expect(
        screen.getByText("No files in this vector store.")
      ).toBeInTheDocument();
    });
  });

  describe("Files Table", () => {
    const mockStore: VectorStore = {
      id: "vs_123",
      name: "Test Vector Store",
      created_at: 1710000000,
      status: "ready",
      file_counts: { total: 2 },
      usage_bytes: 2048,
      metadata: {},
    };

    const mockFiles: VectorStoreFile[] = [
      {
        id: "file_123",
        status: "completed",
        created_at: 1710001000,
        usage_bytes: 1024,
      },
      {
        id: "file_456",
        status: "processing",
        created_at: 1710002000,
        usage_bytes: 512,
      },
    ];

    test("renders files table with correct data", () => {
      render(
        <VectorStoreDetailView
          {...defaultProps}
          store={mockStore}
          files={mockFiles}
        />
      );

      expect(screen.getByText("Files")).toBeInTheDocument();
      expect(
        screen.getByText("Files in this vector store")
      ).toBeInTheDocument();

      expect(screen.getByText("ID")).toBeInTheDocument();
      expect(screen.getByText("Status")).toBeInTheDocument();
      expect(screen.getByText("Created")).toBeInTheDocument();
      expect(screen.getByText("Usage Bytes")).toBeInTheDocument();

      expect(screen.getByText("file_123")).toBeInTheDocument();
      expect(screen.getByText("completed")).toBeInTheDocument();
      expect(
        screen.getByText(new Date(1710001000 * 1000).toLocaleString())
      ).toBeInTheDocument();
      expect(screen.getByText("1024")).toBeInTheDocument();

      expect(screen.getByText("file_456")).toBeInTheDocument();
      expect(screen.getByText("processing")).toBeInTheDocument();
      expect(
        screen.getByText(new Date(1710002000 * 1000).toLocaleString())
      ).toBeInTheDocument();
      expect(screen.getByText("512")).toBeInTheDocument();
    });

    test("file ID links are clickable and navigate correctly", () => {
      render(
        <VectorStoreDetailView
          {...defaultProps}
          store={mockStore}
          files={mockFiles}
          id="vs_123"
        />
      );

      const fileButton = screen.getByRole("button", { name: "file_123" });
      expect(fileButton).toBeInTheDocument();

      fireEvent.click(fileButton);
      expect(mockPush).toHaveBeenCalledWith(
        "/logs/vector-stores/vs_123/files/file_123"
      );
    });

    test("handles multiple file clicks correctly", () => {
      render(
        <VectorStoreDetailView
          {...defaultProps}
          store={mockStore}
          files={mockFiles}
          id="vs_123"
        />
      );

      const file1Button = screen.getByRole("button", { name: "file_123" });
      const file2Button = screen.getByRole("button", { name: "file_456" });

      fireEvent.click(file1Button);
      expect(mockPush).toHaveBeenCalledWith(
        "/logs/vector-stores/vs_123/files/file_123"
      );

      fireEvent.click(file2Button);
      expect(mockPush).toHaveBeenCalledWith(
        "/logs/vector-stores/vs_123/files/file_456"
      );

      expect(mockPush).toHaveBeenCalledTimes(2);
    });
  });

  describe("Layout Structure", () => {
    const mockStore: VectorStore = {
      id: "vs_layout_test",
      name: "Layout Test Store",
      created_at: 1710000000,
      status: "ready",
      file_counts: { total: 1 },
      usage_bytes: 1024,
      metadata: {},
    };

    test("renders main content and sidebar in correct layout", () => {
      render(<VectorStoreDetailView {...defaultProps} store={mockStore} />);

      expect(screen.getByText("Files")).toBeInTheDocument();

      expect(screen.getByText("vs_layout_test")).toBeInTheDocument();
      expect(screen.getByText("Layout Test Store")).toBeInTheDocument();
      expect(screen.getByText("ready")).toBeInTheDocument();
      expect(screen.getByText("1")).toBeInTheDocument();
      expect(screen.getByText("1024")).toBeInTheDocument();
    });
  });
});
