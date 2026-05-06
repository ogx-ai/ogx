import React from "react";
import { render, waitFor } from "@testing-library/react";
import "@testing-library/jest-dom";
import { LogsTable, LogTableRow } from "./logs-table";
import { PaginationStatus } from "@/lib/types";

// Mock next/navigation
jest.mock("next/navigation", () => ({
  useRouter: () => ({
    push: jest.fn(),
  }),
}));

// Mock the useInfiniteScroll hook
jest.mock("@/hooks/use-infinite-scroll", () => ({
  useInfiniteScroll: jest.fn((onLoadMore, options) => {
    const ref = React.useRef(null);

    React.useEffect(() => {
      // Simulate the observer behavior
      if (options?.enabled && onLoadMore) {
        // Trigger load after a delay to simulate intersection
        const timeout = setTimeout(() => {
          onLoadMore();
        }, 100);

        return () => clearTimeout(timeout);
      }
    }, [options?.enabled, onLoadMore]);

    return ref;
  }),
}));

// IntersectionObserver mock is already in jest.setup.ts

describe("LogsTable Viewport Loading", () => {
  const mockData: LogTableRow[] = Array.from({ length: 10 }, (_, i) => ({
    id: `row_${i}`,
    input: `Input ${i}`,
    output: `Output ${i}`,
    model: "test-model",
    createdTime: new Date().toISOString(),
    detailPath: `/logs/test/${i}`,
  }));

  const defaultProps = {
    data: mockData,
    status: "idle" as PaginationStatus,
    hasMore: true,
    error: null,
    caption: "Test table",
    emptyMessage: "No data",
  };

  beforeEach(() => {
    jest.clearAllMocks();
  });

  test("should trigger loadMore when sentinel is visible", async () => {
    const mockLoadMore = jest.fn();

    render(<LogsTable {...defaultProps} onLoadMore={mockLoadMore} />);

    // Wait for the intersection observer to trigger
    await waitFor(
      () => {
        expect(mockLoadMore).toHaveBeenCalled();
      },
      { timeout: 300 }
    );

    expect(mockLoadMore).toHaveBeenCalledTimes(1);
  });

  test("should not trigger loadMore when already loading", async () => {
    const mockLoadMore = jest.fn();

    render(
      <LogsTable
        {...defaultProps}
        status="loading-more"
        onLoadMore={mockLoadMore}
      />
    );

    // Wait for possible triggers
    await new Promise(resolve => setTimeout(resolve, 300));

    expect(mockLoadMore).not.toHaveBeenCalled();
  });

  test("should not trigger loadMore when status is loading", async () => {
    const mockLoadMore = jest.fn();

    render(
      <LogsTable {...defaultProps} status="loading" onLoadMore={mockLoadMore} />
    );

    // Wait for possible triggers
    await new Promise(resolve => setTimeout(resolve, 300));

    expect(mockLoadMore).not.toHaveBeenCalled();
  });

  test("should not trigger loadMore when hasMore is false", async () => {
    const mockLoadMore = jest.fn();

    render(
      <LogsTable {...defaultProps} hasMore={false} onLoadMore={mockLoadMore} />
    );

    // Wait for possible triggers
    await new Promise(resolve => setTimeout(resolve, 300));

    expect(mockLoadMore).not.toHaveBeenCalled();
  });

  test("sentinel element should not be rendered when loading", () => {
    const { container } = render(
      <LogsTable {...defaultProps} status="loading-more" />
    );

    // Check that no sentinel row with height: 1 exists
    const sentinelRow = container.querySelector('tr[style*="height: 1"]');
    expect(sentinelRow).not.toBeInTheDocument();
  });

  test("sentinel element should be rendered when not loading and hasMore", () => {
    const { container } = render(
      <LogsTable {...defaultProps} hasMore={true} status="idle" />
    );

    // Check that sentinel row exists
    const sentinelRow = container.querySelector('tr[style*="height: 1"]');
    expect(sentinelRow).toBeInTheDocument();
  });
});
