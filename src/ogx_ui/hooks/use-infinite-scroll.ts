"use client";

import { useRef, useEffect } from "react";

interface UseInfiniteScrollOptions {
  /** Whether the feature is enabled (e.g., hasMore data) */
  enabled?: boolean;
  /** Threshold for intersection (0-1, how much of sentinel must be visible) */
  threshold?: number;
  /** Margin around root to trigger earlier (e.g., "100px" to load 100px before visible) */
  rootMargin?: string;
}

/**
 * Custom hook for infinite scroll using Intersection Observer
 *
 * @param onLoadMore - Callback to load more data
 * @param options - Configuration options
 * @returns ref to attach to sentinel element
 */
export function useInfiniteScroll(
  onLoadMore: (() => void) | undefined,
  options: UseInfiniteScrollOptions = {}
) {
  const { enabled = true, threshold = 0.1, rootMargin = "100px" } = options;
  const sentinelRef = useRef<HTMLTableRowElement>(null);

  useEffect(() => {
    if (!onLoadMore || !enabled) return;

    const observer = new IntersectionObserver(
      entries => {
        const [entry] = entries;
        if (entry.isIntersecting) {
          onLoadMore();
        }
      },
      {
        threshold,
        rootMargin,
      }
    );

    const sentinel = sentinelRef.current;
    if (sentinel) {
      observer.observe(sentinel);
    }

    return () => {
      observer.disconnect();
    };
  }, [onLoadMore, enabled, threshold, rootMargin]);

  return sentinelRef;
}
