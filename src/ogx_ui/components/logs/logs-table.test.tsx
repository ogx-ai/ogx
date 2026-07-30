import React from "react";
import { render, screen, fireEvent } from "@testing-library/react";
import "@testing-library/jest-dom";
import { LogsTable, LogTableRow } from "./logs-table";
import { PaginationStatus } from "@/lib/types";

// Mock next/navigation
const mockPush = jest.fn();
jest.mock("next/navigation", () => ({
  useRouter: () => ({
    push: mockPush,
  }),
}));

// Mock helper functions
jest.mock("@/lib/truncate-text");

// Import the mocked functions
import { truncateText as originalTruncateText } from "@/lib/truncate-text";

// Cast to jest.Mock for typings
const truncateText = originalTruncateText as jest.Mock;

describe("LogsTable", () => {
  const defaultProps = {
    data: [] as LogTableRow[],
    status: "idle" as PaginationStatus,
    error: null,
    caption: "Test table caption",
    emptyMessage: "No data found",
  };

  beforeEach(() => {
    // Reset all mocks before each test
    mockPush.mockClear();
    truncateText.mockClear();

    // Default pass-through implementation
    truncateText.mockImplementation((text: string | undefined) => text);
  });

  test("renders without crashing with default props", () => {
    render(<LogsTable {...defaultProps} />);
    expect(screen.getByText("No data found")).toBeInTheDocument();
  });

  test("click on a row navigates to the correct URL", () => {
    const mockData: LogTableRow[] = [
      {
        id: "row_123",
        input: "Test input",
        output: "Test output",
        model: "test-model",
        createdTime: "2024-01-01 12:00:00",
        detailPath: "/test/path/row_123",
      },
    ];

    render(<LogsTable {...defaultProps} data={mockData} />);

    const row = screen.getByText("Test input").closest("tr");
    if (row) {
      fireEvent.click(row);
      expect(mockPush).toHaveBeenCalledWith("/test/path/row_123");
    } else {
      throw new Error('Row with "Test input" not found for router mock test.');
    }
  });

  describe("Loading State", () => {
    test("renders skeleton UI when isLoading is true", () => {
      const { container } = render(
        <LogsTable {...defaultProps} status="loading" />
      );

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

      // Check that table headers are still rendered
      expect(screen.getByText("Input")).toBeInTheDocument();
      expect(screen.getByText("Output")).toBeInTheDocument();
      expect(screen.getByText("Model")).toBeInTheDocument();
      expect(screen.getByText("Created")).toBeInTheDocument();
    });

    test("renders correct number of skeleton rows", () => {
      const { container } = render(
        <LogsTable {...defaultProps} status="loading" />
      );

      const skeletonRows = container.querySelectorAll("tbody tr");
      expect(skeletonRows.length).toBe(3); // Should render 3 skeleton rows
    });
  });

  describe("Error State", () => {
    test("renders error message when error prop is provided", () => {
      const errorMessage = "Network Error";
      render(
        <LogsTable
          {...defaultProps}
          status="error"
          error={{ name: "Error", message: errorMessage } as Error}
        />
      );
      expect(
        screen.getByText("Unable to load chat completions")
      ).toBeInTheDocument();
      expect(screen.getByText(errorMessage)).toBeInTheDocument();
    });

    test("renders default error message when error.message is not available", () => {
      render(
        <LogsTable
          {...defaultProps}
          status="error"
          error={{ name: "Error", message: "" } as Error}
        />
      );
      expect(
        screen.getByText("Unable to load chat completions")
      ).toBeInTheDocument();
      expect(
        screen.getByText("An unexpected error occurred while loading the data.")
      ).toBeInTheDocument();
    });

    test("renders default error message when error prop is an object without message", () => {
      render(
        <LogsTable {...defaultProps} status="error" error={{} as Error} />
      );
      expect(
        screen.getByText("Unable to load chat completions")
      ).toBeInTheDocument();
      expect(
        screen.getByText("An unexpected error occurred while loading the data.")
      ).toBeInTheDocument();
    });

    test("does not render table when in error state", () => {
      render(
        <LogsTable
          {...defaultProps}
          status="error"
          error={{ name: "Error", message: "Test error" } as Error}
        />
      );
      const table = screen.queryByRole("table");
      expect(table).not.toBeInTheDocument();
    });
  });

  describe("Empty State", () => {
    test("renders custom empty message when data array is empty", () => {
      render(
        <LogsTable
          {...defaultProps}
          data={[]}
          emptyMessage="Custom empty message"
        />
      );
      expect(screen.getByText("Custom empty message")).toBeInTheDocument();

      // Ensure that the table structure is NOT rendered in the empty state
      const table = screen.queryByRole("table");
      expect(table).not.toBeInTheDocument();
    });
  });

  describe("Data Rendering", () => {
    test("renders table caption, headers, and data correctly", () => {
      const mockData: LogTableRow[] = [
        {
          id: "row_1",
          input: "First input",
          output: "First output",
          model: "model-1",
          createdTime: "2024-01-01 12:00:00",
          detailPath: "/path/1",
        },
        {
          id: "row_2",
          input: "Second input",
          output: "Second output",
          model: "model-2",
          createdTime: "2024-01-02 13:00:00",
          detailPath: "/path/2",
        },
      ];

      render(
        <LogsTable
          {...defaultProps}
          data={mockData}
          caption="Custom table caption"
        />
      );

      // Table caption
      expect(screen.getByText("Custom table caption")).toBeInTheDocument();

      // Table headers
      expect(screen.getByText("Input")).toBeInTheDocument();
      expect(screen.getByText("Output")).toBeInTheDocument();
      expect(screen.getByText("Model")).toBeInTheDocument();
      expect(screen.getByText("Created")).toBeInTheDocument();

      // Data rows
      expect(screen.getByText("First input")).toBeInTheDocument();
      expect(screen.getByText("First output")).toBeInTheDocument();
      expect(screen.getByText("model-1")).toBeInTheDocument();
      expect(screen.getByText("2024-01-01 12:00:00")).toBeInTheDocument();

      expect(screen.getByText("Second input")).toBeInTheDocument();
      expect(screen.getByText("Second output")).toBeInTheDocument();
      expect(screen.getByText("model-2")).toBeInTheDocument();
      expect(screen.getByText("2024-01-02 13:00:00")).toBeInTheDocument();
    });

    test("applies correct CSS classes to table rows", () => {
      const mockData: LogTableRow[] = [
        {
          id: "row_1",
          input: "Test input",
          output: "Test output",
          model: "test-model",
          createdTime: "2024-01-01 12:00:00",
          detailPath: "/test/path",
        },
      ];

      render(<LogsTable {...defaultProps} data={mockData} />);

      const row = screen.getByText("Test input").closest("tr");
      expect(row).toHaveClass("cursor-pointer");
      expect(row).toHaveClass("hover:bg-muted/50");
    });

    test("applies correct alignment to Created column", () => {
      const mockData: LogTableRow[] = [
        {
          id: "row_1",
          input: "Test input",
          output: "Test output",
          model: "test-model",
          createdTime: "2024-01-01 12:00:00",
          detailPath: "/test/path",
        },
      ];

      render(<LogsTable {...defaultProps} data={mockData} />);

      const createdCell = screen.getByText("2024-01-01 12:00:00").closest("td");
      expect(createdCell).toHaveClass("text-right");
    });
  });

  describe("Text Truncation", () => {
    test("truncates input and output text using truncateText function", () => {
      // Mock truncateText to return truncated versions
      truncateText.mockImplementation((text: string | undefined) => {
        if (typeof text === "string" && text.length > 10) {
          return text.slice(0, 10) + "...";
        }
        return text;
      });

      const longInput =
        "This is a very long input text that should be truncated";
      const longOutput =
        "This is a very long output text that should be truncated";

      const mockData: LogTableRow[] = [
        {
          id: "row_1",
          input: longInput,
          output: longOutput,
          model: "test-model",
          createdTime: "2024-01-01 12:00:00",
          detailPath: "/test/path",
        },
      ];

      render(<LogsTable {...defaultProps} data={mockData} />);

      // Verify truncateText was called
      expect(truncateText).toHaveBeenCalledWith(longInput);
      expect(truncateText).toHaveBeenCalledWith(longOutput);

      // Verify truncated text is displayed
      const truncatedTexts = screen.getAllByText("This is a ...");
      expect(truncatedTexts).toHaveLength(2); // one for input, one for output
      truncatedTexts.forEach(textElement =>
        expect(textElement).toBeInTheDocument()
      );
    });

    test("does not truncate model names", () => {
      const mockData: LogTableRow[] = [
        {
          id: "row_1",
          input: "Test input",
          output: "Test output",
          model: "very-long-model-name-that-should-not-be-truncated",
          createdTime: "2024-01-01 12:00:00",
          detailPath: "/test/path",
        },
      ];

      render(<LogsTable {...defaultProps} data={mockData} />);

      // Model name should not be passed to truncateText
      expect(truncateText).not.toHaveBeenCalledWith(
        "very-long-model-name-that-should-not-be-truncated"
      );

      // Full model name should be displayed
      expect(
        screen.getByText("very-long-model-name-that-should-not-be-truncated")
      ).toBeInTheDocument();
    });
  });

  describe("Accessibility", () => {
    test("table has proper role and structure", () => {
      const mockData: LogTableRow[] = [
        {
          id: "row_1",
          input: "Test input",
          output: "Test output",
          model: "test-model",
          createdTime: "2024-01-01 12:00:00",
          detailPath: "/test/path",
        },
      ];

      render(<LogsTable {...defaultProps} data={mockData} />);

      const tables = screen.getAllByRole("table");
      expect(tables).toHaveLength(2); // Fixed header table + body table

      const columnHeaders = screen.getAllByRole("columnheader");
      expect(columnHeaders).toHaveLength(4);

      const rows = screen.getAllByRole("row");
      expect(rows).toHaveLength(3); // 1 header row + 1 data row + 1 "no more items" row

      expect(screen.getByText("Input")).toBeInTheDocument();
      expect(screen.getByText("Output")).toBeInTheDocument();
      expect(screen.getByText("Model")).toBeInTheDocument();
      expect(screen.getByText("Created")).toBeInTheDocument();
    });
  });
});
