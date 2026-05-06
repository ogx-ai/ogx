"use client";

import { useEffect, useState } from "react";
import { useParams, useRouter } from "next/navigation";
import { useAuthClient } from "@/hooks/use-auth-client";
import { ContentsAPI, VectorStoreContentItem } from "@/lib/contents-api";
import type { VectorStore } from "llama-stack-client/resources/vector-stores/vector-stores";
import type { VectorStoreFile } from "llama-stack-client/resources/vector-stores/files";
import { Card, CardContent, CardHeader, CardTitle } from "@/components/ui/card";
import { Skeleton } from "@/components/ui/skeleton";
import { Button } from "@/components/ui/button";
import { Edit, Trash2, Eye } from "lucide-react";
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
import {
  Table,
  TableBody,
  TableCaption,
  TableCell,
  TableHead,
  TableHeader,
  TableRow,
} from "@/components/ui/table";

export default function ContentsListPage() {
  const params = useParams();
  const router = useRouter();
  const vectorStoreId = params.id as string;
  const fileId = params.fileId as string;
  const client = useAuthClient();

  const getTextFromContent = (content: unknown): string => {
    if (typeof content === "string") {
      return content;
    } else if (content && content.type === "text") {
      return content.text;
    }
    return "";
  };

  const [store, setStore] = useState<VectorStore | null>(null);
  const [file, setFile] = useState<VectorStoreFile | null>(null);
  const [contents, setContents] = useState<VectorStoreContentItem[]>([]);
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
        const contentsAPI = new ContentsAPI(client);
        const contentsResponse = await contentsAPI.listContents(
          vectorStoreId,
          fileId,
          { limit: 100 }
        );
        setContents(contentsResponse.data);
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

  const handleDeleteContent = async (contentId: string) => {
    try {
      const contentsAPI = new ContentsAPI(client);
      await contentsAPI.deleteContent(vectorStoreId, fileId, contentId);
      setContents(contents.filter(content => content.id !== contentId));
    } catch (err) {
      console.error("Failed to delete content:", err);
    }
  };

  const handleViewContent = (contentId: string) => {
    router.push(
      `/logs/vector-stores/${vectorStoreId}/files/${fileId}/contents/${contentId}`
    );
  };

  const title = `Contents in File: ${fileId}`;

  const breadcrumbSegments: BreadcrumbSegment[] = [
    { label: "Vector Stores", href: "/logs/vector-stores" },
    {
      label: store?.name || vectorStoreId,
      href: `/logs/vector-stores/${vectorStoreId}`,
    },
    { label: "Files", href: `/logs/vector-stores/${vectorStoreId}` },
    {
      label: fileId,
      href: `/logs/vector-stores/${vectorStoreId}/files/${fileId}`,
    },
    { label: "Contents" },
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
          <CardTitle>Content Chunks ({contents.length})</CardTitle>
        </CardHeader>
        <CardContent>
          {isLoadingFile ? (
            <Skeleton className="h-4 w-full" />
          ) : errorFile ? (
            <div className="text-destructive text-sm">
              Error loading file: {errorFile.message}
            </div>
          ) : isLoadingContents ? (
            <div className="space-y-2">
              <Skeleton className="h-4 w-full" />
              <Skeleton className="h-4 w-3/4" />
              <Skeleton className="h-4 w-1/2" />
            </div>
          ) : errorContents ? (
            <div className="text-destructive text-sm">
              Error loading contents: {errorContents.message}
            </div>
          ) : contents.length > 0 ? (
            <Table>
              <TableCaption>Contents in this file</TableCaption>
              <TableHeader>
                <TableRow>
                  <TableHead>Content ID</TableHead>
                  <TableHead>Content Preview</TableHead>
                  <TableHead>Embedding</TableHead>
                  <TableHead>Position</TableHead>
                  <TableHead>Created</TableHead>
                  <TableHead>Actions</TableHead>
                </TableRow>
              </TableHeader>
              <TableBody>
                {contents.map(content => (
                  <TableRow key={content.id}>
                    <TableCell className="font-mono text-xs">
                      <Button
                        variant="link"
                        className="p-0 h-auto font-mono text-xs text-blue-600 hover:text-blue-800 dark:text-blue-400 dark:hover:text-blue-300"
                        onClick={() => handleViewContent(content.id)}
                        title={content.id}
                      >
                        {content.id.substring(0, 10)}...
                      </Button>
                    </TableCell>
                    <TableCell>
                      <div className="max-w-md">
                        <p
                          className="text-sm truncate"
                          title={getTextFromContent(content.content)}
                        >
                          {getTextFromContent(content.content)}
                        </p>
                      </div>
                    </TableCell>
                    <TableCell className="text-xs text-gray-500">
                      {content.embedding && content.embedding.length > 0 ? (
                        <div className="max-w-xs">
                          <span
                            className="font-mono text-xs bg-gray-100 dark:bg-gray-800 rounded px-1 py-0.5"
                            title={`${content.embedding.length}D vector: [${content.embedding
                              .slice(0, 3)
                              .map(v => v.toFixed(3))
                              .join(", ")}...]`}
                          >
                            [
                            {content.embedding
                              .slice(0, 3)
                              .map(v => v.toFixed(3))
                              .join(", ")}
                            ...] ({content.embedding.length}D)
                          </span>
                        </div>
                      ) : (
                        <span className="text-gray-400 dark:text-gray-500 italic">
                          No embedding
                        </span>
                      )}
                    </TableCell>
                    <TableCell className="text-xs text-gray-500">
                      {content.metadata.chunk_window
                        ? content.metadata.chunk_window
                        : `${content.metadata.content_length || 0} chars`}
                    </TableCell>
                    <TableCell className="text-xs">
                      {new Date(
                        content.created_timestamp * 1000
                      ).toLocaleString()}
                    </TableCell>
                    <TableCell>
                      <div className="flex gap-1">
                        <Button
                          variant="ghost"
                          size="sm"
                          className="h-6 w-6 p-0"
                          title="View content details"
                          onClick={() => handleViewContent(content.id)}
                        >
                          <Eye className="h-3 w-3" />
                        </Button>
                        <Button
                          variant="ghost"
                          size="sm"
                          className="h-6 w-6 p-0"
                          title="Edit content"
                          onClick={() => handleViewContent(content.id)}
                        >
                          <Edit className="h-3 w-3" />
                        </Button>
                        <Button
                          variant="ghost"
                          size="sm"
                          className="h-6 w-6 p-0 text-destructive hover:text-destructive"
                          title="Delete content"
                          onClick={() => handleDeleteContent(content.id)}
                        >
                          <Trash2 className="h-3 w-3" />
                        </Button>
                      </div>
                    </TableCell>
                  </TableRow>
                ))}
              </TableBody>
            </Table>
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
            label="Chunking Strategy"
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
