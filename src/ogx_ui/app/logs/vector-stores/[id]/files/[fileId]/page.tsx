"use client";

import { useEffect, useState } from "react";
import { useParams, useRouter } from "next/navigation";
import { useAuthClient } from "@/hooks/use-auth-client";
import type { VectorStore } from "llama-stack-client/resources/vector-stores/vector-stores";
import type {
  VectorStoreFile,
  FileContentResponse,
} from "llama-stack-client/resources/vector-stores/files";
import { Card, CardContent, CardHeader, CardTitle } from "@/components/ui/card";
import { Skeleton } from "@/components/ui/skeleton";
import { Button } from "@/components/ui/button";
import { List } from "lucide-react";
import {
  DetailLoadingView,
  DetailErrorView,
  DetailNotFoundView,
  DetailLayout,
  PropertiesCard,
  PropertyItem,
} from "@/components/layout/detail-layout";
import {
  PageBreadcrumb,
  BreadcrumbSegment,
} from "@/components/layout/page-breadcrumb";

export default function FileDetailPage() {
  const params = useParams();
  const router = useRouter();
  const vectorStoreId = params.id as string;
  const fileId = params.fileId as string;
  const client = useAuthClient();

  const [store, setStore] = useState<VectorStore | null>(null);
  const [file, setFile] = useState<VectorStoreFile | null>(null);
  const [contents, setContents] = useState<FileContentResponse | null>(null);
  const [isLoadingStore, setIsLoadingStore] = useState(true);
  const [isLoadingFile, setIsLoadingFile] = useState(true);
  const [isLoadingContents, setIsLoadingContents] = useState(true);
  const [errorStore, setErrorStore] = useState<Error | null>(null);
  const [errorFile, setErrorFile] = useState<Error | null>(null);
  const [errorContents, setErrorContents] = useState<Error | null>(null);

  useEffect(() => {
    if (!vectorStoreId) return;

    const fetchStore = async () => {
      setIsLoadingStore(true);
      setErrorStore(null);
      try {
        const response = await client.vectorStores.retrieve(vectorStoreId);
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
  }, [vectorStoreId, client]);

  useEffect(() => {
    if (!vectorStoreId || !fileId) return;

    const fetchFile = async () => {
      setIsLoadingFile(true);
      setErrorFile(null);
      try {
        const response = await client.vectorStores.files.retrieve(
          vectorStoreId,
          fileId
        );
        setFile(response as VectorStoreFile);
      } catch (err) {
        setErrorFile(
          err instanceof Error ? err : new Error("Failed to load file.")
        );
      } finally {
        setIsLoadingFile(false);
      }
    };
    fetchFile();
  }, [vectorStoreId, fileId, client]);

  useEffect(() => {
    if (!vectorStoreId || !fileId) return;

    const fetchContents = async () => {
      setIsLoadingContents(true);
      setErrorContents(null);
      try {
        const response = await client.vectorStores.files.content(
          vectorStoreId,
          fileId
        );
        setContents(response);
      } catch (err) {
        setErrorContents(
          err instanceof Error ? err : new Error("Failed to load contents.")
        );
      } finally {
        setIsLoadingContents(false);
      }
    };
    fetchContents();
  }, [vectorStoreId, fileId, client]);

  const handleViewContents = () => {
    router.push(
      `/logs/vector-stores/${vectorStoreId}/files/${fileId}/contents`
    );
  };

  const title = `File: ${fileId}`;

  const breadcrumbSegments: BreadcrumbSegment[] = [
    { label: "Vector Stores", href: "/logs/vector-stores" },
    {
      label: store?.name || vectorStoreId,
      href: `/logs/vector-stores/${vectorStoreId}`,
    },
    { label: "Files", href: `/logs/vector-stores/${vectorStoreId}` },
    { label: fileId },
  ];

  if (errorStore) {
    return (
      <DetailErrorView title={title} id={vectorStoreId} error={errorStore} />
    );
  }
  if (isLoadingStore) {
    return <DetailLoadingView title={title} />;
  }
  if (!store) {
    return <DetailNotFoundView title={title} id={vectorStoreId} />;
  }

  const mainContent = (
    <>
      <Card>
        <CardHeader>
          <CardTitle>File Information</CardTitle>
        </CardHeader>
        <CardContent>
          {isLoadingFile ? (
            <div className="space-y-2">
              <Skeleton className="h-4 w-full" />
              <Skeleton className="h-4 w-3/4" />
              <Skeleton className="h-4 w-1/2" />
            </div>
          ) : errorFile ? (
            <div className="text-destructive text-sm">
              Error loading file: {errorFile.message}
            </div>
          ) : file ? (
            <div className="space-y-4">
              <div>
                <h3 className="text-lg font-medium mb-2">File Details</h3>
                <div className="grid grid-cols-2 gap-4 text-sm">
                  <div>
                    <span className="font-medium text-gray-600 dark:text-gray-400">
                      Status:
                    </span>
                    <span className="ml-2">{file.status}</span>
                  </div>
                  <div>
                    <span className="font-medium text-gray-600 dark:text-gray-400">
                      Size:
                    </span>
                    <span className="ml-2">{file.usage_bytes} bytes</span>
                  </div>
                  <div>
                    <span className="font-medium text-gray-600 dark:text-gray-400">
                      Created:
                    </span>
                    <span className="ml-2">
                      {new Date(file.created_at * 1000).toLocaleString()}
                    </span>
                  </div>
                  <div>
                    <span className="font-medium text-gray-600 dark:text-gray-400">
                      Content Strategy:
                    </span>
                    <span className="ml-2">{file.chunking_strategy.type}</span>
                  </div>
                </div>
              </div>

              <div className="border-t pt-4">
                <h3 className="text-lg font-medium mb-3">Actions</h3>
                <Button
                  onClick={handleViewContents}
                  className="flex items-center gap-2 hover:bg-primary/90 dark:hover:bg-primary/80 hover:scale-105 transition-all duration-200"
                >
                  <List className="h-4 w-4" />
                  View Contents
                </Button>
              </div>
            </div>
          ) : (
            <p className="text-gray-500 italic text-sm">File not found.</p>
          )}
        </CardContent>
      </Card>

      <Card>
        <CardHeader>
          <CardTitle>Content Summary</CardTitle>
        </CardHeader>
        <CardContent>
          {isLoadingContents ? (
            <div className="space-y-2">
              <Skeleton className="h-4 w-full" />
              <Skeleton className="h-4 w-3/4" />
              <Skeleton className="h-4 w-1/2" />
            </div>
          ) : errorContents ? (
            <div className="text-destructive text-sm">
              Error loading content summary: {errorContents.message}
            </div>
          ) : contents && contents.content.length > 0 ? (
            <div className="space-y-3">
              <div className="grid grid-cols-2 gap-4 text-sm">
                <div>
                  <span className="font-medium text-gray-600 dark:text-gray-400">
                    Content Items:
                  </span>
                  <span className="ml-2">{contents.content.length}</span>
                </div>
                <div>
                  <span className="font-medium text-gray-600 dark:text-gray-400">
                    Total Characters:
                  </span>
                  <span className="ml-2">
                    {contents.content.reduce(
                      (total, item) => total + item.text.length,
                      0
                    )}
                  </span>
                </div>
              </div>
              <div className="pt-2">
                <span className="text-sm font-medium text-gray-600 dark:text-gray-400">
                  Preview:
                </span>
                <div className="mt-1 bg-gray-50 dark:bg-gray-800 rounded-md p-3">
                  <p className="text-sm text-gray-900 dark:text-gray-100 line-clamp-3">
                    {contents.content[0]?.text.substring(0, 200)}...
                  </p>
                </div>
              </div>
            </div>
          ) : (
            <p className="text-gray-500 italic text-sm">
              No contents found for this file.
            </p>
          )}
        </CardContent>
      </Card>
    </>
  );

  const sidebar = (
    <PropertiesCard>
      <PropertyItem label="File ID" value={fileId} />
      <PropertyItem label="Vector Store ID" value={vectorStoreId} />
      {file && (
        <>
          <PropertyItem label="Status" value={file.status} />
          <PropertyItem
            label="Created"
            value={new Date(file.created_at * 1000).toLocaleString()}
          />
          <PropertyItem label="Usage Bytes" value={file.usage_bytes} />
          <PropertyItem
            label="Content Strategy"
            value={file.chunking_strategy.type}
          />
        </>
      )}
      {store && (
        <>
          <PropertyItem label="Store Name" value={store.name || ""} />
          <PropertyItem
            label="Provider ID"
            value={(store.metadata.provider_id as string) || ""}
          />
        </>
      )}
    </PropertiesCard>
  );

  return (
    <>
      <PageBreadcrumb segments={breadcrumbSegments} />
      <DetailLayout title={title} mainContent={mainContent} sidebar={sidebar} />
    </>
  );
}
