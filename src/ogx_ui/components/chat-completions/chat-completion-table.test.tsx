import React from "react";
import { render, screen, fireEvent } from "@testing-library/react";
import "@testing-library/jest-dom";
import { ChatCompletionsTable } from "./chat-completions-table";
import { ChatCompletion } from "@/lib/types";

// Mock next/navigation
const mockPush = jest.fn();
jest.mock("next/navigation", () => ({
  useRouter: () => ({
    push: mockPush,
  }),
}));

// Mock next-auth
jest.mock("next-auth/react", () => ({
  useSession: () => ({
    status: "authenticated",
    data: { accessToken: "mock-token" },
  }),
}));

// Mock helper functions
jest.mock("@/lib/truncate-text");
jest.mock("@/lib/format-message-content");

// Mock the auth client hook
const mockClient = {
  chat: {
    completions: {
      list: jest.fn(),
    },
  },
};

jest.mock("@/hooks/use-auth-client", () => ({
  useAuthClient: () => mockClient,
}));

// Mock the usePagination hook
const mockLoadMore = jest.fn();
jest.mock("@/hooks/use-pagination", () => ({
  usePagination: jest.fn(() => ({
    data: [],
    status: "idle",
    hasMore: false,
    error: null,
    loadMore: mockLoadMore,
  })),
}));

// Import the mocked functions to set up default or specific implementations
import { truncateText as originalTruncateText } from "@/lib/truncate-text";
import {
  extractTextFromContentPart as originalExtractTextFromContentPart,
  extractDisplayableText as originalExtractDisplayableText,
} from "@/lib/format-message-content";

// Import the mocked hook
import { usePagination } from "@/hooks/use-pagination";
const mockedUsePagination = usePagination as jest.MockedFunction<
  typeof usePagination
>;

// Cast to jest.Mock for typings
const truncateText = originalTruncateText as jest.Mock;
const extractTextFromContentPart =
  originalExtractTextFromContentPart as jest.Mock;
const extractDisplayableText = originalExtractDisplayableText as jest.Mock;

describe("ChatCompletionsTable", () => {
  const defaultProps = {};

  beforeEach(() => {
    // Reset all mocks before each test
    mockPush.mockClear();
    truncateText.mockClear();
    extractTextFromContentPart.mockClear();
    extractDisplayableText.mockClear();
    mockLoadMore.mockClear();
    jest.clearAllMocks();

    // Default pass-through implementations
    truncateText.mockImplementation((text: string | undefined) => text);
    extractTextFromContentPart.mockImplementation((content: unknown) =>
      typeof content === "string" ? content : "extracted text"
    );
    extractDisplayableText.mockImplementation((message: unknown) => {
      const msg = message as { content?: string };
      return msg?.content || "extracted output";
    });

    // Default hook return value
    mockedUsePagination.mockReturnValue({
      data: [],
      status: "idle",
      hasMore: false,
      error: null,
      loadMore: mockLoadMore,
    });
  });

  test("renders without crashing with default props", () => {
    render(<ChatCompletionsTable {...defaultProps} />);
    expect(screen.getByText("No chat completions found.")).toBeInTheDocument();
  });

  test("click on a row navigates to the correct URL", () => {
    const mockData: ChatCompletion[] = [
      {
        id: "completion_123",
        choices: [
          {
            message: { role: "assistant", content: "Test response" },
            finish_reason: "stop",
            index: 0,
          },
        ],
        object: "chat.completion",
        created: 1234567890,
        model: "test-model",
        input_messages: [{ role: "user", content: "Test prompt" }],
      },
    ];

    // Configure the mock to return our test data
    mockedUsePagination.mockReturnValue({
      data: mockData,
      status: "idle",
      hasMore: false,
      error: null,
      loadMore: mockLoadMore,
    });

    render(<ChatCompletionsTable {...defaultProps} />);

    const row = screen.getByText("Test prompt").closest("tr");
    if (row) {
      fireEvent.click(row);
      expect(mockPush).toHaveBeenCalledWith(
        "/logs/chat-completions/completion_123"
      );
    } else {
      throw new Error('Row with "Test prompt" not found for router mock test.');
    }
  });

  describe("Loading State", () => {
    test("renders skeleton UI when isLoading is true", () => {
      mockedUsePagination.mockReturnValue({
        data: [],
        status: "loading",
        hasMore: false,
        error: null,
        loadMore: mockLoadMore,
      });

      const { container } = render(<ChatCompletionsTable {...defaultProps} />);

      // Check for skeleton in the table caption
      const tableCaption = container.querySelector("caption");
      expect(tableCaption).toBeInTheDocument();
      if (tableCaption) {
        const captionSkeleton = tableCaption.querySelector(
          '[data-slot="skeleton"]'
        );
        expect(captionSkeleton).toBeInTheDocument();
      }

      // Check for skeletons in the table body cells
      const tableBody = container.querySelector("tbody");
      expect(tableBody).toBeInTheDocument();
      if (tableBody) {
        const bodySkeletons = tableBody.querySelectorAll(
          '[data-slot="skeleton"]'
        );
        expect(bodySkeletons.length).toBeGreaterThan(0);
      }
    });
  });

  describe("Error State", () => {
    test("renders error message when error prop is provided", () => {
      const errorMessage = "Network Error";
      mockedUsePagination.mockReturnValue({
        data: [],
        status: "error",
        hasMore: false,
        error: { name: "Error", message: errorMessage } as Error,
        loadMore: mockLoadMore,
      });

      render(<ChatCompletionsTable {...defaultProps} />);
      expect(
        screen.getByText("Unable to load chat completions")
      ).toBeInTheDocument();
      expect(screen.getByText(errorMessage)).toBeInTheDocument();
    });

    test.each([{ name: "Error", message: "" }, {}])(
      "renders default error message when error has no message",
      errorObject => {
        mockedUsePagination.mockReturnValue({
          data: [],
          status: "error",
          hasMore: false,
          error: errorObject as Error,
          loadMore: mockLoadMore,
        });

        render(<ChatCompletionsTable {...defaultProps} />);
        expect(
          screen.getByText("Unable to load chat completions")
        ).toBeInTheDocument();
        expect(
          screen.getByText(
            "An unexpected error occurred while loading the data."
          )
        ).toBeInTheDocument();
      }
    );
  });

  describe("Empty State", () => {
    test('renders "No chat completions found." and no table when data array is empty', () => {
      render(<ChatCompletionsTable {...defaultProps} />);
      expect(
        screen.getByText("No chat completions found.")
      ).toBeInTheDocument();

      // Ensure that the table structure is NOT rendered in the empty state
      const table = screen.queryByRole("table");
      expect(table).not.toBeInTheDocument();
    });
  });

  describe("Data Rendering", () => {
    test("renders table caption, headers, and completion data correctly", () => {
      const mockCompletions: ChatCompletion[] = [
        {
          id: "comp_1",
          object: "chat.completion",
          created: 1710000000,
          model: "llama-test-model",
          choices: [
            {
              index: 0,
              message: { role: "assistant", content: "Test output" },
              finish_reason: "stop",
            },
          ],
          input_messages: [{ role: "user", content: "Test input" }],
        },
        {
          id: "comp_2",
          object: "chat.completion",
          created: 1710001000,
          model: "llama-another-model",
          choices: [
            {
              index: 0,
              message: { role: "assistant", content: "Another output" },
              finish_reason: "stop",
            },
          ],
          input_messages: [{ role: "user", content: "Another input" }],
        },
      ];

      // Set up mocks to return expected values
      extractTextFromContentPart.mockImplementation((content: unknown) => {
        if (content === "Test input") return "Test input";
        if (content === "Another input") return "Another input";
        return "extracted text";
      });
      extractDisplayableText.mockImplementation((message: unknown) => {
        const msg = message as { content?: string };
        if (msg?.content === "Test output") return "Test output";
        if (msg?.content === "Another output") return "Another output";
        return "extracted output";
      });

      mockedUsePagination.mockReturnValue({
        data: mockCompletions,
        status: "idle",
        hasMore: false,
        error: null,
        loadMore: mockLoadMore,
      });

      render(<ChatCompletionsTable {...defaultProps} />);

      // Table caption
      expect(
        screen.getByText("A list of your recent chat completions.")
      ).toBeInTheDocument();

      // Table headers
      expect(screen.getByText("Input")).toBeInTheDocument();
      expect(screen.getByText("Output")).toBeInTheDocument();
      expect(screen.getByText("Model")).toBeInTheDocument();
      expect(screen.getByText("Created")).toBeInTheDocument();

      // Data rows
      expect(screen.getByText("Test input")).toBeInTheDocument();
      expect(screen.getByText("Test output")).toBeInTheDocument();
      expect(screen.getByText("llama-test-model")).toBeInTheDocument();
      expect(
        screen.getByText(new Date(1710000000 * 1000).toLocaleString())
      ).toBeInTheDocument();

      expect(screen.getByText("Another input")).toBeInTheDocument();
      expect(screen.getByText("Another output")).toBeInTheDocument();
      expect(screen.getByText("llama-another-model")).toBeInTheDocument();
      expect(
        screen.getByText(new Date(1710001000 * 1000).toLocaleString())
      ).toBeInTheDocument();
    });
  });

  describe("Text Truncation and Content Extraction", () => {
    test("truncates long input and output text", () => {
      // Specific mock implementation for this test
      truncateText.mockImplementation(
        (text: string | undefined, maxLength?: number) => {
          const defaultTestMaxLength = 10;
          const effectiveMaxLength = maxLength ?? defaultTestMaxLength;
          return typeof text === "string" && text.length > effectiveMaxLength
            ? text.slice(0, effectiveMaxLength) + "..."
            : text;
        }
      );

      const longInput =
        "This is a very long input message that should be truncated.";
      const longOutput =
        "This is a very long output message that should also be truncated.";

      extractTextFromContentPart.mockReturnValue(longInput);
      extractDisplayableText.mockReturnValue(longOutput);

      const mockCompletions: ChatCompletion[] = [
        {
          id: "comp_trunc",
          object: "chat.completion",
          created: 1710002000,
          model: "llama-trunc-model",
          choices: [
            {
              index: 0,
              message: { role: "assistant", content: longOutput },
              finish_reason: "stop",
            },
          ],
          input_messages: [{ role: "user", content: longInput }],
        },
      ];

      mockedUsePagination.mockReturnValue({
        data: mockCompletions,
        status: "idle",
        hasMore: false,
        error: null,
        loadMore: mockLoadMore,
      });

      render(<ChatCompletionsTable {...defaultProps} />);

      // The truncated text should be present for both input and output
      const truncatedTexts = screen.getAllByText(
        longInput.slice(0, 10) + "..."
      );
      expect(truncatedTexts.length).toBe(2); // one for input, one for output
    });

    test("uses content extraction functions correctly", () => {
      const complexMessage = [
        { type: "text", text: "Extracted input" },
        { type: "image", url: "http://example.com/image.png" },
      ];
      const assistantMessage = {
        role: "assistant",
        content: "Extracted output from assistant",
      };

      const mockCompletions: ChatCompletion[] = [
        {
          id: "comp_extract",
          object: "chat.completion",
          created: 1710003000,
          model: "llama-extract-model",
          choices: [
            {
              index: 0,
              message: assistantMessage,
              finish_reason: "stop",
            },
          ],
          input_messages: [{ role: "user", content: complexMessage }],
        },
      ];

      extractTextFromContentPart.mockReturnValue("Extracted input");
      extractDisplayableText.mockReturnValue("Extracted output from assistant");

      mockedUsePagination.mockReturnValue({
        data: mockCompletions,
        status: "idle",
        hasMore: false,
        error: null,
        loadMore: mockLoadMore,
      });

      render(<ChatCompletionsTable {...defaultProps} />);

      // Verify the extraction functions were called
      expect(extractTextFromContentPart).toHaveBeenCalledWith(complexMessage);
      expect(extractDisplayableText).toHaveBeenCalledWith(assistantMessage);

      // Verify the extracted text appears in the table
      expect(screen.getByText("Extracted input")).toBeInTheDocument();
      expect(
        screen.getByText("Extracted output from assistant")
      ).toBeInTheDocument();
    });
  });
});
