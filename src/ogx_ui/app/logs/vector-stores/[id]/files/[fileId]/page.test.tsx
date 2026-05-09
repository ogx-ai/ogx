import React from "react";
import {
  render,
  screen,
  fireEvent,
  waitFor,
  act,
} from "@testing-library/react";
import "@testing-library/jest-dom";
import FileDetailPage from "./page";
import type { VectorStore } from "llama-stack-client/resources/vector-stores/vector-stores";
import type {
  VectorStoreFile,
  FileContentResponse,
} from "llama-stack-client/resources/vector-stores/files";

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
      content: jest.fn(),
    },
  },
};

jest.mock("@/hooks/use-auth-client", () => ({
  useAuthClient: () => mockClient,
}));

describe("FileDetailPage", () => {
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
    usage_bytes: 2048,
    chunking_strategy: { type: "fixed_size" },
  };

  const mockFileContent: FileContentResponse = {
    content: [
      { text: "First chunk of file content." },
      {
        text: "Second chunk with more detailed information about the content.",
      },
      { text: "Third and final chunk of the file." },
    ],
  };

  beforeEach(() => {
    jest.clearAllMocks();

    mockClient.vectorStores.retrieve.mockResolvedValue(mockStore);
    mockClient.vectorStores.files.retrieve.mockResolvedValue(mockFile);
    mockClient.vectorStores.files.content.mockResolvedValue(mockFileContent);
  });

  describe("Loading and Error States", () => {
    test("renders loading skeleton while fetching store data", async () => {
      mockClient.vectorStores.retrieve.mockImplementation(
        () => new Promise(() => {})
      );

      await act(async () => {
        await act(async () => {
          render(<FileDetailPage />);
        });
      });

      const skeletons = document.querySelectorAll('[data-slot="skeleton"]');
      expect(skeletons.length).toBeGreaterThan(0);
    });

    test("renders error message when store API call fails", async () => {
      const error = new Error("Failed to load store");
      mockClient.vectorStores.retrieve.mockRejectedValue(error);

      await act(async () => {
        await act(async () => {
          render(<FileDetailPage />);
        });
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
        render(<FileDetailPage />);
      });

      await waitFor(() => {
        expect(
          screen.getByText(/No details found for ID: vs_123/)
        ).toBeInTheDocument();
      });
    });

    test("renders file loading skeleton", async () => {
      mockClient.vectorStores.files.retrieve.mockImplementation(
        () => new Promise(() => {})
      );

      const { container } = render(<FileDetailPage />);

      await waitFor(() => {
        expect(screen.getByText("File: file_456")).toBeInTheDocument();
      });

      const skeletons = container.querySelectorAll('[data-slot="skeleton"]');
      expect(skeletons.length).toBeGreaterThan(0);
    });

    test("renders file error message", async () => {
      const error = new Error("Failed to load file");
      mockClient.vectorStores.files.retrieve.mockRejectedValue(error);

      await act(async () => {
        render(<FileDetailPage />);
      });

      await waitFor(() => {
        expect(
          screen.getByText("Error loading file: Failed to load file")
        ).toBeInTheDocument();
      });
    });

    test("renders content error message", async () => {
      const error = new Error("Failed to load contents");
      mockClient.vectorStores.files.content.mockRejectedValue(error);

      await act(async () => {
        render(<FileDetailPage />);
      });

      await waitFor(() => {
        expect(
          screen.getByText(
            "Error loading content summary: Failed to load contents"
          )
        ).toBeInTheDocument();
      });
    });
  });

  describe("File Information Display", () => {
    test("renders file details correctly", async () => {
      await act(async () => {
        await act(async () => {
          render(<FileDetailPage />);
        });
      });

      await waitFor(() => {
        expect(screen.getByText("File: file_456")).toBeInTheDocument();
        expect(screen.getByText("File Information")).toBeInTheDocument();
        expect(screen.getByText("File Details")).toBeInTheDocument();
      });

      const statusTexts = screen.getAllByText("Status:");
      expect(statusTexts.length).toBeGreaterThan(0);
      const completedTexts = screen.getAllByText("completed");
      expect(completedTexts.length).toBeGreaterThan(0);
      expect(screen.getByText("Size:")).toBeInTheDocument();
      expect(screen.getByText("2048 bytes")).toBeInTheDocument();
      const createdTexts = screen.getAllByText("Created:");
      expect(createdTexts.length).toBeGreaterThan(0);
      const dateTexts = screen.getAllByText(
        new Date(1710001000 * 1000).toLocaleString()
      );
      expect(dateTexts.length).toBeGreaterThan(0);
      const strategyTexts = screen.getAllByText("Content Strategy:");
      expect(strategyTexts.length).toBeGreaterThan(0);
      const fixedSizeTexts = screen.getAllByText("fixed_size");
      expect(fixedSizeTexts.length).toBeGreaterThan(0);
    });

    test("handles missing file data", async () => {
      mockClient.vectorStores.files.retrieve.mockResolvedValue(null);

      await act(async () => {
        render(<FileDetailPage />);
      });

      await waitFor(() => {
        expect(screen.getByText("File not found.")).toBeInTheDocument();
      });
    });
  });

  describe("Content Summary Display", () => {
    test("renders content summary correctly", async () => {
      await act(async () => {
        render(<FileDetailPage />);
      });

      await waitFor(() => {
        expect(screen.getByText("Content Summary")).toBeInTheDocument();
        expect(screen.getByText("Content Items:")).toBeInTheDocument();
        expect(screen.getByText("3")).toBeInTheDocument();
        expect(screen.getByText("Total Characters:")).toBeInTheDocument();

        const totalChars = mockFileContent.content.reduce(
          (total, item) => total + item.text.length,
          0
        );
        expect(screen.getByText(totalChars.toString())).toBeInTheDocument();

        expect(screen.getByText("Preview:")).toBeInTheDocument();
        expect(
          screen.getByText(/First chunk of file content\./)
        ).toBeInTheDocument();
      });
    });

    test("handles empty content", async () => {
      mockClient.vectorStores.files.content.mockResolvedValue({
        content: [],
      });

      await act(async () => {
        render(<FileDetailPage />);
      });

      await waitFor(() => {
        expect(
          screen.getByText("No contents found for this file.")
        ).toBeInTheDocument();
      });
    });

    test("truncates long content preview", async () => {
      const longContent = {
        content: [
          {
            text: "This is a very long piece of content that should be truncated after 200 characters to ensure the preview doesn't take up too much space in the UI and remains readable and manageable for users viewing the file details page.",
          },
        ],
      };

      mockClient.vectorStores.files.content.mockResolvedValue(longContent);

      await act(async () => {
        render(<FileDetailPage />);
      });

      await waitFor(() => {
        expect(
          screen.getByText(/This is a very long piece of content/)
        ).toBeInTheDocument();
        expect(screen.getByText(/\.\.\.$/)).toBeInTheDocument();
      });
    });
  });

  describe("Navigation and Actions", () => {
    test("navigates to contents list when View Contents button is clicked", async () => {
      await act(async () => {
        render(<FileDetailPage />);
      });

      await waitFor(() => {
        expect(screen.getByText("Actions")).toBeInTheDocument();
      });

      const viewContentsButton = screen.getByRole("button", {
        name: /View Contents/,
      });
      fireEvent.click(viewContentsButton);

      expect(mockPush).toHaveBeenCalledWith(
        "/logs/vector-stores/vs_123/files/file_456/contents"
      );
    });

    test("View Contents button is styled correctly", async () => {
      await act(async () => {
        render(<FileDetailPage />);
      });

      await waitFor(() => {
        const button = screen.getByRole("button", { name: /View Contents/ });
        expect(button).toHaveClass("flex", "items-center", "gap-2");
      });
    });
  });

  describe("Breadcrumb Navigation", () => {
    test("renders correct breadcrumb structure", async () => {
      await act(async () => {
        render(<FileDetailPage />);
      });

      await waitFor(() => {
        const vectorStoresTexts = screen.getAllByText("Vector Stores");
        expect(vectorStoresTexts.length).toBeGreaterThan(0);
        const storeNameTexts = screen.getAllByText("Test Vector Store");
        expect(storeNameTexts.length).toBeGreaterThan(0);
        const filesTexts = screen.getAllByText("Files");
        expect(filesTexts.length).toBeGreaterThan(0);
        const fileIdTexts = screen.getAllByText("file_456");
        expect(fileIdTexts.length).toBeGreaterThan(0);
      });
    });

    test("uses store ID when store name is not available", async () => {
      const storeWithoutName = { ...mockStore, name: "" };
      mockClient.vectorStores.retrieve.mockResolvedValue(storeWithoutName);

      await act(async () => {
        render(<FileDetailPage />);
      });

      await waitFor(() => {
        const storeIdTexts = screen.getAllByText("vs_123");
        expect(storeIdTexts.length).toBeGreaterThan(0);
      });
    });
  });

  describe("Sidebar Properties", () => {
    test.skip("renders file and store properties correctly", async () => {
      await act(async () => {
        render(<FileDetailPage />);
      });

      await waitFor(() => {
        expect(screen.getByText("File ID")).toBeInTheDocument();
        const fileIdTexts = screen.getAllByText("file_456");
        expect(fileIdTexts.length).toBeGreaterThan(0);
        expect(screen.getByText("Vector Store ID")).toBeInTheDocument();
        const storeIdTexts = screen.getAllByText("vs_123");
        expect(storeIdTexts.length).toBeGreaterThan(0);
        expect(screen.getByText("Status")).toBeInTheDocument();
        const completedTexts = screen.getAllByText("completed");
        expect(completedTexts.length).toBeGreaterThan(0);
        expect(screen.getByText("Usage Bytes")).toBeInTheDocument();
        const usageTexts = screen.getAllByText("2048");
        expect(usageTexts.length).toBeGreaterThan(0);
        expect(screen.getByText("Content Strategy")).toBeInTheDocument();
        const fixedSizeTexts = screen.getAllByText("fixed_size");
        expect(fixedSizeTexts.length).toBeGreaterThan(0);

        expect(screen.getByText("Store Name")).toBeInTheDocument();
        const storeNameTexts = screen.getAllByText("Test Vector Store");
        expect(storeNameTexts.length).toBeGreaterThan(0);
        expect(screen.getByText("Provider ID")).toBeInTheDocument();
        expect(screen.getByText("test_provider")).toBeInTheDocument();
      });
    });

    test("handles missing optional properties", async () => {
      const minimalFile = {
        id: "file_456",
        status: "completed",
        created_at: 1710001000,
        usage_bytes: 2048,
        chunking_strategy: { type: "fixed_size" },
      };

      const minimalStore = {
        ...mockStore,
        name: "",
        metadata: {},
      };

      mockClient.vectorStores.files.retrieve.mockResolvedValue(minimalFile);
      mockClient.vectorStores.retrieve.mockResolvedValue(minimalStore);

      await act(async () => {
        render(<FileDetailPage />);
      });

      await waitFor(() => {
        const fileIdTexts = screen.getAllByText("file_456");
        expect(fileIdTexts.length).toBeGreaterThan(0);
        const storeIdTexts = screen.getAllByText("vs_123");
        expect(storeIdTexts.length).toBeGreaterThan(0);
      });

      expect(screen.getByText("File: file_456")).toBeInTheDocument();
    });
  });

  describe("Loading States for Individual Sections", () => {
    test("shows loading skeleton for content while file loads", async () => {
      mockClient.vectorStores.files.content.mockImplementation(
        () => new Promise(() => {})
      );

      const { container } = render(<FileDetailPage />);

      await waitFor(() => {
        expect(screen.getByText("Content Summary")).toBeInTheDocument();
      });

      const skeletons = container.querySelectorAll('[data-slot="skeleton"]');
      expect(skeletons.length).toBeGreaterThan(0);
    });
  });

  describe("Error Handling", () => {
    test("handles multiple simultaneous errors gracefully", async () => {
      mockClient.vectorStores.files.retrieve.mockRejectedValue(
        new Error("File error")
      );
      mockClient.vectorStores.files.content.mockRejectedValue(
        new Error("Content error")
      );

      await act(async () => {
        render(<FileDetailPage />);
      });

      await waitFor(() => {
        expect(
          screen.getByText("Error loading file: File error")
        ).toBeInTheDocument();
        expect(
          screen.getByText("Error loading content summary: Content error")
        ).toBeInTheDocument();
      });
    });
  });
});
