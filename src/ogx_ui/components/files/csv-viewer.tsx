"use client";

import React, { useMemo, useState } from "react";
import Papa from "papaparse";
import {
  TableBody,
  TableCell,
  TableHead,
  TableHeader,
  TableRow,
} from "@/components/ui/table";
import { Input } from "@/components/ui/input";
import { Search, ChevronUp, ChevronDown } from "lucide-react";

interface CSVViewerProps {
  content: string;
  filename?: string;
}

interface ParsedCSV {
  headers: string[];
  rows: string[][];
  errors: Papa.ParseError[];
}

// Constants for content size management
const MAX_CSV_SIZE = 10 * 1024 * 1024; // 10MB
const WARN_CSV_SIZE = 5 * 1024 * 1024; // 5MB

export function CSVViewer({ content }: CSVViewerProps) {
  const [searchTerm, setSearchTerm] = useState("");
  const [sortColumn, setSortColumn] = useState<number | null>(null);
  const [sortDirection, setSortDirection] = useState<"asc" | "desc">("asc");

  // Check content size
  const contentSize = content.length;
  const isLargeFile = contentSize > WARN_CSV_SIZE;
  const isOversized = contentSize > MAX_CSV_SIZE;

  const parsedData: ParsedCSV = useMemo(() => {
    if (!content) {
      return { headers: [], rows: [], errors: [] };
    }

    const result = Papa.parse(content.trim(), {
      header: false,
      skipEmptyLines: true,
      delimiter: "", // Auto-detect delimiter
      quoteChar: '"',
      escapeChar: '"',
    });

    if (result.errors.length > 0) {
      console.warn("CSV parsing errors:", result.errors);
    }

    const data = result.data as string[][];
    if (data.length === 0) {
      return { headers: [], rows: [], errors: result.errors };
    }

    // First row as headers, rest as data
    const headers = data[0] || [];
    const rows = data.slice(1);

    return {
      headers,
      rows,
      errors: result.errors,
    };
  }, [content]);

  const filteredAndSortedRows = useMemo(() => {
    let filtered = parsedData.rows;

    // Apply search filter
    if (searchTerm) {
      filtered = filtered.filter(row =>
        row.some(cell => cell?.toLowerCase().includes(searchTerm.toLowerCase()))
      );
    }

    // Apply sorting
    if (sortColumn !== null) {
      filtered = [...filtered].sort((a, b) => {
        const aVal = a[sortColumn] || "";
        const bVal = b[sortColumn] || "";

        // Try to parse as numbers for numeric sorting
        const aNum = parseFloat(aVal);
        const bNum = parseFloat(bVal);

        if (!isNaN(aNum) && !isNaN(bNum)) {
          return sortDirection === "asc" ? aNum - bNum : bNum - aNum;
        }

        // String sorting
        const comparison = aVal.localeCompare(bVal);
        return sortDirection === "asc" ? comparison : -comparison;
      });
    }

    return filtered;
  }, [parsedData.rows, searchTerm, sortColumn, sortDirection]);

  // Handle oversized files after hooks
  if (isOversized) {
    return (
      <div className="p-4 border border-red-200 rounded-lg bg-red-50 dark:bg-red-900/20 dark:border-red-800">
        <h3 className="font-semibold text-red-800 dark:text-red-300 mb-2">
          File Too Large
        </h3>
        <p className="text-sm text-red-600 dark:text-red-400">
          CSV file is too large to display (
          {(contentSize / (1024 * 1024)).toFixed(2)}MB). Maximum supported size
          is {MAX_CSV_SIZE / (1024 * 1024)}MB.
        </p>
        <p className="text-xs text-red-500 dark:text-red-400 mt-2">
          Please download the file to view its contents.
        </p>
      </div>
    );
  }

  const handleSort = (columnIndex: number) => {
    if (sortColumn === columnIndex) {
      setSortDirection(sortDirection === "asc" ? "desc" : "asc");
    } else {
      setSortColumn(columnIndex);
      setSortDirection("asc");
    }
  };

  if (parsedData.errors.length > 0 && parsedData.headers.length === 0) {
    return (
      <div className="p-4 border border-red-200 rounded-lg bg-red-50 dark:bg-red-900/20 dark:border-red-800">
        <h3 className="font-semibold text-red-800 dark:text-red-300 mb-2">
          CSV Parsing Error
        </h3>
        <p className="text-sm text-red-600 dark:text-red-400">
          Failed to parse CSV file. Please check the file format.
        </p>
        {parsedData.errors.slice(0, 3).map((error, index) => (
          <p
            key={index}
            className="text-xs text-red-500 dark:text-red-400 mt-1"
          >
            Line {error.row}: {error.message}
          </p>
        ))}
      </div>
    );
  }

  if (parsedData.headers.length === 0) {
    return (
      <div className="p-4 border rounded-lg bg-muted/50">
        <p className="text-sm text-muted-foreground">
          No data found in CSV file.
        </p>
      </div>
    );
  }

  return (
    <div className="space-y-4 w-full overflow-x-auto">
      {/* CSV Info & Search */}
      <div className="flex items-center justify-between gap-4">
        <div className="text-sm text-muted-foreground">
          {parsedData.rows.length} rows × {parsedData.headers.length} columns
          {filteredAndSortedRows.length !== parsedData.rows.length && (
            <span className="ml-2">
              (showing {filteredAndSortedRows.length} filtered)
            </span>
          )}
        </div>
        <div className="relative max-w-sm">
          <Search className="absolute left-2 top-2.5 h-4 w-4 text-muted-foreground" />
          <Input
            placeholder="Search CSV data..."
            value={searchTerm}
            onChange={e => setSearchTerm(e.target.value)}
            className="pl-8"
          />
        </div>
      </div>

      {/* Large File Warning */}
      {isLargeFile && !isOversized && (
        <div className="p-3 border border-orange-200 rounded-lg bg-orange-50 dark:bg-orange-900/20 dark:border-orange-800">
          <p className="text-sm text-orange-800 dark:text-orange-300">
            ⚠️ Large file detected ({(contentSize / (1024 * 1024)).toFixed(2)}
            MB). Performance may be slower than usual.
          </p>
        </div>
      )}

      {/* Parsing Warnings */}
      {parsedData.errors.length > 0 && (
        <div className="p-3 border border-yellow-200 rounded-lg bg-yellow-50 dark:bg-yellow-900/20 dark:border-yellow-800">
          <p className="text-sm text-yellow-800 dark:text-yellow-300">
            ⚠️ {parsedData.errors.length} parsing warning(s) - data may be
            incomplete
          </p>
        </div>
      )}

      {/* CSV Table */}
      <div className="border rounded-lg">
        <div className="max-h-[500px] overflow-y-auto scrollbar-thin scrollbar-track-muted scrollbar-thumb-muted-foreground">
          <table className="w-full caption-bottom text-sm table-auto">
            <TableHeader>
              <TableRow>
                {parsedData.headers.map((header, index) => (
                  <TableHead
                    key={index}
                    className="cursor-pointer hover:bg-muted/50 select-none px-3 whitespace-nowrap"
                    onClick={() => handleSort(index)}
                  >
                    <div className="flex items-center justify-between">
                      <div className="font-semibold">
                        {header || `Column ${index + 1}`}
                      </div>
                      {sortColumn === index && (
                        <div className="ml-1 flex-shrink-0">
                          {sortDirection === "asc" ? (
                            <ChevronUp className="h-3 w-3" />
                          ) : (
                            <ChevronDown className="h-3 w-3" />
                          )}
                        </div>
                      )}
                    </div>
                  </TableHead>
                ))}
              </TableRow>
            </TableHeader>
            <TableBody>
              {filteredAndSortedRows.map((row, rowIndex) => (
                <TableRow key={rowIndex}>
                  {row.map((cell, cellIndex) => (
                    <TableCell
                      key={cellIndex}
                      className="font-mono text-sm overflow-hidden px-3 max-w-xs"
                    >
                      <div className="truncate" title={cell || ""}>
                        {cell || ""}
                      </div>
                    </TableCell>
                  ))}
                  {/* Fill empty cells if row is shorter than headers */}
                  {Array.from({
                    length: Math.max(0, parsedData.headers.length - row.length),
                  }).map((_, emptyIndex) => (
                    <TableCell
                      key={`empty-${emptyIndex}`}
                      className="text-muted-foreground px-3 max-w-xs"
                    >
                      —
                    </TableCell>
                  ))}
                </TableRow>
              ))}
            </TableBody>
          </table>
        </div>
      </div>

      {/* Table Stats */}
      <div className="text-xs text-muted-foreground space-y-1">
        {filteredAndSortedRows.length === 0 && searchTerm && (
          <p>No rows match your search criteria.</p>
        )}
        {sortColumn !== null && (
          <p>
            Sorted by column &quot;
            {parsedData.headers[sortColumn] || `Column ${sortColumn + 1}`}&quot;
            ({sortDirection}ending)
          </p>
        )}
        {parsedData.headers.length > 4 && (
          <p className="italic">
            💡 Scroll horizontally to view all {parsedData.headers.length}{" "}
            columns
          </p>
        )}
      </div>
    </div>
  );
}
