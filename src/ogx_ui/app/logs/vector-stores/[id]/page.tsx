"use client";

import { useEffect, useState } from "react";
import { useParams } from "next/navigation";
import { useAuthClient } from "@/hooks/use-auth-client";
import type { VectorStore } from "llama-stack-client/resources/vector-stores/vector-stores";
import type { VectorStoreFile } from "llama-stack-client/resources/vector-stores/files";
import { VectorStoreDetailView } from "@/components/vector-stores/vector-store-detail";

export default function VectorStoreDetailPage() {
  const params = useParams();
  const id = params.id as string;
  const client = useAuthClient();

  const [store, setStore] = useState<VectorStore | null>(null);
  const [files, setFiles] = useState<VectorStoreFile[]>([]);
  const [isLoadingStore, setIsLoadingStore] = useState(true);
  const [isLoadingFiles, setIsLoadingFiles] = useState(true);
  const [errorStore, setErrorStore] = useState<Error | null>(null);
  const [errorFiles, setErrorFiles] = useState<Error | null>(null);

  useEffect(() => {
    if (!id) {
      setErrorStore(new Error("Vector Store ID is missing."));
      setIsLoadingStore(false);
      return;
    }
    const fetchStore = async () => {
      setIsLoadingStore(true);
      setErrorStore(null);
      try {
        const response = await client.vectorStores.retrieve(id);
        setStore(response as VectorStore);
      } catch (err) {
        setErrorStore(
          err instanceof Error ? err : new Error("Failed to load vector store.")
        );
      } finally {
        setIsLoadingStore(false);
      }
    };
    fetchStore();
  }, [id, client]);

  useEffect(() => {
    if (!id) {
      setErrorFiles(new Error("Vector Store ID is missing."));
      setIsLoadingFiles(false);
      return;
    }
    const fetchFiles = async () => {
      setIsLoadingFiles(true);
      setErrorFiles(null);
      try {
        const result = await client.vectorStores.files.list(id);
        setFiles((result as { data: VectorStoreFile[] }).data);
      } catch (err) {
        setErrorFiles(
          err instanceof Error ? err : new Error("Failed to load files.")
        );
      } finally {
        setIsLoadingFiles(false);
      }
    };
    fetchFiles();
  }, [id, client.vectorStores.files]);

  return (
    <VectorStoreDetailView
      store={store}
      files={files}
      isLoadingStore={isLoadingStore}
      isLoadingFiles={isLoadingFiles}
      errorStore={errorStore}
      errorFiles={errorFiles}
      id={id}
    />
  );
}
