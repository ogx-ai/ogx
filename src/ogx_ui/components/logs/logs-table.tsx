"use client";

import { useRouter } from "next/navigation";
import { useRef } from "react";
import { truncateText } from "@/lib/truncate-text";
import { PaginationStatus } from "@/lib/types";
import { useInfiniteScroll } from "@/hooks/use-infinite-scroll";
import {
  Table,
  TableBody,
  TableCaption,
  TableCell,
  TableHead,
  TableHeader,
  TableRow,
} from "@/components/ui/table";
import { Skeleton } from "@/components/ui/skeleton";

// Generic table row data interface
export interface LogTableRow {
  id: string;
  input: string;
  output: string;
  model: string;
  createdTime: string;
  detailPath: string;
}

interface LogsTableProps {
  /** Array of log table row data to display */
  data: LogTableRow[];
  /** Current loading/error status */
  status: PaginationStatus;
  /** Whether more data is available to load */
  hasMore?: boolean;
  /** Error state, null if no error */
  error: Error | null;
  /** Table caption for accessibility */
  caption: string;
  /** Message to show when no data is available */
  emptyMessage: string;
  /** Callback function to load more data */
  onLoadMore?: () => void;
}

export function LogsTable({
  data,
  status,
  hasMore = false,
  error,
  caption,
  emptyMessage,
  onLoadMore,
}: LogsTableProps) {
  const router = useRouter();
  const tableContainerRef = useRef<HTMLDivElement>(null);

  // Use Intersection Observer for infinite scroll
  const sentinelRef = useInfiniteScroll(onLoadMore, {
    enabled: hasMore && status === "idle",
    rootMargin: "100px",
    threshold: 0.1,
  });

  // Fixed header component
  const FixedHeader = () => (
    <div className="bg-background border-b border-border">
      <Table>
        <TableHeader>
          <TableRow>
            <TableHead className="w-1/4">Input</TableHead>
            <TableHead className="w-1/4">Output</TableHead>
            <TableHead className="w-1/4">Model</TableHead>
            <TableHead className="w-1/4 text-right">Created</TableHead>
          </TableRow>
        </TableHeader>
      </Table>
    </div>
  );

  if (status === "loading") {
    return (
      <div className="h-full flex flex-col">
        <FixedHeader />
        <div ref={tableContainerRef} className="overflow-auto flex-1 min-h-0">
          <Table>
            <TableCaption>
              <Skeleton className="h-4 w-[250px] mx-auto" />
            </TableCaption>
            <TableBody>
              {[...Array(3)].map((_, i) => (
                <TableRow key={`skeleton-${i}`}>
                  <TableCell className="w-1/4">
                    <Skeleton className="h-4 w-full" />
                  </TableCell>
                  <TableCell className="w-1/4">
                    <Skeleton className="h-4 w-full" />
                  </TableCell>
                  <TableCell className="w-1/4">
                    <Skeleton className="h-4 w-3/4" />
                  </TableCell>
                  <TableCell className="w-1/4 text-right">
                    <Skeleton className="h-4 w-1/2 ml-auto" />
                  </TableCell>
                </TableRow>
              ))}
            </TableBody>
          </Table>
        </div>
      </div>
    );
  }

  if (status === "error") {
    return (
      <div className="flex flex-col items-center justify-center p-8 space-y-4">
        <div className="text-destructive font-medium">
          Unable to load chat completions
        </div>
        <div className="text-sm text-muted-foreground text-center max-w-md">
          {error?.message ||
            "An unexpected error occurred while loading the data."}
        </div>
        <button
          onClick={() => window.location.reload()}
          className="px-4 py-2 bg-primary text-primary-foreground rounded-md hover:bg-primary/90 transition-colors"
        >
          Retry
        </button>
      </div>
    );
  }

  if (data.length === 0) {
    return <p>{emptyMessage}</p>;
  }

  return (
    <div className="h-full flex flex-col">
      <FixedHeader />
      <div ref={tableContainerRef} className="overflow-auto flex-1 min-h-0">
        <Table>
          <TableCaption className="sr-only">{caption}</TableCaption>
          <TableBody>
            {data.map(row => (
              <TableRow
                key={row.id}
                onClick={() => router.push(row.detailPath)}
                className="cursor-pointer hover:bg-muted/50"
              >
                <TableCell className="w-1/4">
                  {truncateText(row.input)}
                </TableCell>
                <TableCell className="w-1/4">
                  {truncateText(row.output)}
                </TableCell>
                <TableCell className="w-1/4">{row.model}</TableCell>
                <TableCell className="w-1/4 text-right">
                  {row.createdTime}
                </TableCell>
              </TableRow>
            ))}
            {/* Sentinel element for infinite scroll */}
            {hasMore && status === "idle" && (
              <TableRow ref={sentinelRef} style={{ height: 1 }}>
                <TableCell colSpan={4} style={{ padding: 0, border: 0 }} />
              </TableRow>
            )}
            {status === "loading-more" && (
              <TableRow>
                <TableCell colSpan={4} className="text-center py-4">
                  <div className="flex items-center justify-center space-x-2">
                    <Skeleton className="h-4 w-4 rounded-full" />
                    <span className="text-sm text-muted-foreground">
                      Loading more...
                    </span>
                  </div>
                </TableCell>
              </TableRow>
            )}
            {!hasMore && data.length > 0 && (
              <TableRow>
                <TableCell colSpan={4} className="text-center py-4">
                  <span className="text-sm text-muted-foreground">
                    No more items to load
                  </span>
                </TableCell>
              </TableRow>
            )}
          </TableBody>
        </Table>
      </div>
    </div>
  );
}
