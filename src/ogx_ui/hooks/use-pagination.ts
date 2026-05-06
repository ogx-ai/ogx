"use client";

import { useState, useCallback, useEffect, useRef } from "react";
import { PaginationStatus, UsePaginationOptions } from "@/lib/types";
import { useSession } from "next-auth/react";
import { useAuthClient } from "@/hooks/use-auth-client";
import { useRouter } from "next/navigation";

interface PaginationState<T> {
  data: T[];
  status: PaginationStatus;
  hasMore: boolean;
  error: Error | null;
  lastId: string | null;
}

interface PaginationResponse<T> {
  data: T[];
  has_more: boolean;
  last_id: string;
  first_id: string;
  object: "list";
}

export interface PaginationReturn<T> {
  data: T[];
  status: PaginationStatus;
  hasMore: boolean;
  error: Error | null;
  loadMore: () => void;
  refetch: () => void;
}

interface UsePaginationParams<T> extends UsePaginationOptions {
  fetchFunction: (
    client: ReturnType<typeof useAuthClient>,
    params: {
      after?: string;
      limit: number;
      model?: string;
      order?: string;
    }
  ) => Promise<PaginationResponse<T>>;
  errorMessagePrefix: string;
  enabled?: boolean;
  useAuth?: boolean;
}

export function usePagination<T>({
  limit = 20,
  model,
  order = "desc",
  fetchFunction,
  errorMessagePrefix,
  enabled = true,
  useAuth = true,
}: UsePaginationParams<T>): PaginationReturn<T> {
  const { status: sessionStatus } = useSession();
  const client = useAuthClient();
  const router = useRouter();
  const [state, setState] = useState<PaginationState<T>>({
    data: [],
    status: "loading",
    hasMore: true,
    error: null,
    lastId: null,
  });

  // Use refs to avoid stale closures
  const stateRef = useRef(state);
  stateRef.current = state;

  // Track if initial data has been fetched
  const hasFetchedInitialData = useRef(false);

  /**
   * Fetches data from the API with cursor-based pagination
   */
  const fetchData = useCallback(
    async (after?: string, targetRows?: number) => {
      const isInitialLoad = !after;
      const fetchLimit = targetRows || limit;

      try {
        setState(prev => ({
          ...prev,
          status: isInitialLoad ? "loading" : "loading-more",
          error: null,
        }));

        const response = await fetchFunction(client, {
          after: after || undefined,
          limit: fetchLimit,
          ...(model && { model }),
          ...(order && { order }),
        });

        setState(prev => ({
          ...prev,
          data: isInitialLoad
            ? response.data
            : [...prev.data, ...response.data],
          hasMore: response.has_more,
          lastId: response.last_id || null,
          status: "idle",
        }));
      } catch (err) {
        // Check if it's a 401 unauthorized error
        if (
          err &&
          typeof err === "object" &&
          "status" in err &&
          err.status === 401
        ) {
          router.push("/auth/signin");
          return;
        }

        const errorMessage = isInitialLoad
          ? `Failed to load ${errorMessagePrefix}. Please try refreshing the page.`
          : `Failed to load more ${errorMessagePrefix}. Please try again.`;

        const error =
          err instanceof Error
            ? new Error(`${errorMessage} ${err.message}`)
            : new Error(errorMessage);

        setState(prev => ({
          ...prev,
          error,
          status: "error",
        }));
      }
    },
    [limit, model, order, fetchFunction, errorMessagePrefix, client, router]
  );

  /**
   * Loads more data for infinite scroll
   */
  const loadMore = useCallback(() => {
    const currentState = stateRef.current;
    if (currentState.hasMore && currentState.status === "idle") {
      fetchData(currentState.lastId || undefined);
    }
  }, [fetchData]);

  /**
   * Refetches data from the beginning (resets pagination)
   */
  const refetch = useCallback(() => {
    fetchData();
  }, [fetchData]);

  // Auto-load initial data on mount when enabled
  useEffect(() => {
    // If using auth, wait for session to load
    const isAuthReady = !useAuth || sessionStatus !== "loading";
    const shouldFetch = enabled && isAuthReady;

    if (shouldFetch && !hasFetchedInitialData.current) {
      hasFetchedInitialData.current = true;
      fetchData();
    } else if (!shouldFetch) {
      // Reset the flag when disabled so it can fetch when re-enabled
      hasFetchedInitialData.current = false;
    }
  }, [fetchData, enabled, useAuth, sessionStatus]);

  // Override status if we're waiting for auth
  const effectiveStatus =
    useAuth && sessionStatus === "loading" ? "loading" : state.status;

  return {
    data: state.data,
    status: effectiveStatus,
    hasMore: state.hasMore,
    error: state.error,
    loadMore,
    refetch,
  };
}
