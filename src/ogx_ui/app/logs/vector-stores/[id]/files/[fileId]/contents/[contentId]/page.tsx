"use client";

import { useEffect, useState } from "react";
import { useParams, useRouter } from "next/navigation";
import { useAuthClient } from "@/hooks/use-auth-client";
import { ContentsAPI, VectorStoreContentItem } from "@/lib/contents-api";
import type { VectorStore } from "llama-stack-client/resources/vector-stores/vector-stores";
import type { VectorStoreFile } from "llama-stack-client/resources/vector-stores/files";
import { Card, CardContent, CardHeader, CardTitle } from "@/components/ui/card";
import { Button } from "@/components/ui/button";
import { Input } from "@/components/ui/input";
import { Edit, Save, X, Trash2 } from "lucide-react";
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

export default function ContentDetailPage() {
  const params = useParams();
  const router = useRouter();
  const vectorStoreId = params.id as string;
  const fileId = params.fileId as string;
  const contentId = params.contentId as string;
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
  const [content, setContent] = useState<VectorStoreContentItem | null>(null);
  const [isLoading, setIsLoading] = useState(true);
  const [error, setError] = useState<Error | null>(null);
  const [isEditing, setIsEditing] = useState(false);
  const [editedContent, setEditedContent] = useState("");
  const [editedMetadata, setEditedMetadata] = useState<Record<string, unknown>>(
    {}
  );
  const [isEditingEmbedding, setIsEditingEmbedding] = useState(false);
  const [editedEmbedding, setEditedEmbedding] = useState<number[]>([]);

  useEffect(() => {
    if (!vectorStoreId || !fileId || !contentId) return;

    const fetchData = async () => {
      setIsLoading(true);
      setError(null);
      try {
        const [storeResponse, fileResponse] = await Promise.all([
          client.vectorStores.retrieve(vectorStoreId),
          client.vectorStores.files.retrieve(vectorStoreId, fileId),
        ]);

        setStore(storeResponse as VectorStore);
        setFile(fileResponse as VectorStoreFile);

        const contentsAPI = new ContentsAPI(client);
        const contentsResponse = await contentsAPI.listContents(
          vectorStoreId,
          fileId
        );
        const targetContent = contentsResponse.data.find(
          c => c.id === contentId
        );

        if (targetContent) {
          setContent(targetContent);
          setEditedContent(getTextFromContent(targetContent.content));
          setEditedMetadata({ ...targetContent.metadata });
          setEditedEmbedding(targetContent.embedding || []);
        } else {
          throw new Error(`Content ${contentId} not found`);
        }
      } catch (err) {
        setError(
          err instanceof Error ? err : new Error("Failed to load content.")
        );
      } finally {
        setIsLoading(false);
      }
    };
    fetchData();
  }, [vectorStoreId, fileId, contentId, client]);

  const handleSave = async () => {
    if (!content) return;

    try {
      const updates: { content?: string; metadata?: Record<string, unknown> } =
        {};

      if (editedContent !== getTextFromContent(content.content)) {
        updates.content = editedContent;
      }

      if (JSON.stringify(editedMetadata) !== JSON.stringify(content.metadata)) {
        updates.metadata = editedMetadata;
      }

      if (Object.keys(updates).length > 0) {
        const contentsAPI = new ContentsAPI(client);
        const updatedContent = await contentsAPI.updateContent(
          vectorStoreId,
          fileId,
          contentId,
          updates
        );
        setContent(updatedContent);
      }

      setIsEditing(false);
    } catch (err) {
      console.error("Failed to update content:", err);
    }
  };

  const handleDelete = async () => {
    if (!confirm("Are you sure you want to delete this content?")) return;

    try {
      const contentsAPI = new ContentsAPI(client);
      await contentsAPI.deleteContent(vectorStoreId, fileId, contentId);
      router.push(
        `/logs/vector-stores/${vectorStoreId}/files/${fileId}/contents`
      );
    } catch (err) {
      console.error("Failed to delete content:", err);
    }
  };

  const handleCancel = () => {
    setEditedContent(content ? getTextFromContent(content.content) : "");
    setEditedMetadata({ ...content?.metadata });
    setEditedEmbedding(content?.embedding || []);
    setIsEditing(false);
    setIsEditingEmbedding(false);
  };

  const title = `Content: ${contentId}`;

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
    {
      label: "Contents",
      href: `/logs/vector-stores/${vectorStoreId}/files/${fileId}/contents`,
    },
    { label: contentId },
  ];

  if (error) {
    return <DetailErrorView title={title} id={contentId} error={error} />;
  }
  if (isLoading) {
    return <DetailLoadingView title={title} />;
  }
  if (!content) {
    return <DetailNotFoundView title={title} id={contentId} />;
  }

  const mainContent = (
    <>
      <Card>
        <CardHeader className="flex flex-row items-center justify-between">
          <CardTitle>Content</CardTitle>
          <div className="flex gap-2">
            {isEditing ? (
              <>
                <Button size="sm" onClick={handleSave}>
                  <Save className="h-4 w-4 mr-1" />
                  Save
                </Button>
                <Button size="sm" variant="outline" onClick={handleCancel}>
                  <X className="h-4 w-4 mr-1" />
                  Cancel
                </Button>
              </>
            ) : (
              <>
                <Button size="sm" onClick={() => setIsEditing(true)}>
                  <Edit className="h-4 w-4 mr-1" />
                  Edit
                </Button>
                <Button size="sm" variant="destructive" onClick={handleDelete}>
                  <Trash2 className="h-4 w-4 mr-1" />
                  Delete
                </Button>
              </>
            )}
          </div>
        </CardHeader>
        <CardContent>
          {isEditing ? (
            <textarea
              value={editedContent}
              onChange={e => setEditedContent(e.target.value)}
              className="w-full h-64 p-3 border rounded-md resize-none font-mono text-sm"
              placeholder="Enter content..."
            />
          ) : (
            <div className="p-3 bg-gray-50 dark:bg-gray-800 rounded-md">
              <pre className="whitespace-pre-wrap font-mono text-sm text-gray-900 dark:text-gray-100">
                {getTextFromContent(content.content)}
              </pre>
            </div>
          )}
        </CardContent>
      </Card>

      <Card>
        <CardHeader className="flex flex-row items-center justify-between">
          <CardTitle>Content Embedding</CardTitle>
          <div className="flex gap-2">
            {isEditingEmbedding ? (
              <>
                <Button
                  size="sm"
                  onClick={() => {
                    setIsEditingEmbedding(false);
                  }}
                >
                  <Save className="h-4 w-4 mr-1" />
                  Save
                </Button>
                <Button
                  size="sm"
                  variant="outline"
                  onClick={() => {
                    setEditedEmbedding(content?.embedding || []);
                    setIsEditingEmbedding(false);
                  }}
                >
                  <X className="h-4 w-4 mr-1" />
                  Cancel
                </Button>
              </>
            ) : (
              <Button size="sm" onClick={() => setIsEditingEmbedding(true)}>
                <Edit className="h-4 w-4 mr-1" />
                Edit
              </Button>
            )}
          </div>
        </CardHeader>
        <CardContent>
          {content?.embedding && content.embedding.length > 0 ? (
            isEditingEmbedding ? (
              <div className="space-y-2">
                <p className="text-sm text-gray-600 dark:text-gray-400">
                  Embedding ({editedEmbedding.length}D vector):
                </p>
                <textarea
                  value={JSON.stringify(editedEmbedding, null, 2)}
                  onChange={e => {
                    try {
                      const parsed = JSON.parse(e.target.value);
                      if (
                        Array.isArray(parsed) &&
                        parsed.every(v => typeof v === "number")
                      ) {
                        setEditedEmbedding(parsed);
                      }
                    } catch {}
                  }}
                  className="w-full h-32 p-3 border rounded-md resize-none font-mono text-xs"
                  placeholder="Enter embedding as JSON array..."
                />
              </div>
            ) : (
              <div className="space-y-2">
                <div className="flex items-center gap-2">
                  <span className="font-mono text-xs bg-gray-100 dark:bg-gray-800 rounded px-2 py-1">
                    {content.embedding.length}D vector
                  </span>
                </div>
                <div className="p-3 bg-gray-50 dark:bg-gray-800 rounded-md max-h-32 overflow-y-auto">
                  <pre className="whitespace-pre-wrap font-mono text-xs text-gray-900 dark:text-gray-100">
                    [
                    {content.embedding
                      .slice(0, 20)
                      .map(v => v.toFixed(6))
                      .join(", ")}
                    {content.embedding.length > 20
                      ? `\n... and ${content.embedding.length - 20} more values`
                      : ""}
                    ]
                  </pre>
                </div>
              </div>
            )
          ) : (
            <p className="text-gray-500 italic text-sm">
              No embedding available for this content.
            </p>
          )}
        </CardContent>
      </Card>

      <Card>
        <CardHeader>
          <CardTitle>Metadata</CardTitle>
        </CardHeader>
        <CardContent>
          {isEditing ? (
            <div className="space-y-2">
              {Object.entries(editedMetadata).map(([key, value]) => (
                <div key={key} className="flex gap-2">
                  <Input
                    value={key}
                    onChange={e => {
                      const newMetadata = { ...editedMetadata };
                      delete newMetadata[key];
                      newMetadata[e.target.value] = value;
                      setEditedMetadata(newMetadata);
                    }}
                    placeholder="Key"
                    className="flex-1"
                  />
                  <Input
                    value={
                      typeof value === "string" ? value : JSON.stringify(value)
                    }
                    onChange={e => {
                      setEditedMetadata({
                        ...editedMetadata,
                        [key]: e.target.value,
                      });
                    }}
                    placeholder="Value"
                    className="flex-1"
                  />
                </div>
              ))}
              <Button
                size="sm"
                variant="outline"
                onClick={() => {
                  setEditedMetadata({
                    ...editedMetadata,
                    [""]: "",
                  });
                }}
              >
                Add Field
              </Button>
            </div>
          ) : (
            <div className="space-y-2">
              {Object.entries(content.metadata).map(([key, value]) => (
                <div key={key} className="flex justify-between py-1">
                  <span className="font-medium text-gray-600">{key}:</span>
                  <span className="font-mono text-sm">
                    {typeof value === "string" ? value : JSON.stringify(value)}
                  </span>
                </div>
              ))}
            </div>
          )}
        </CardContent>
      </Card>
    </>
  );

  const sidebar = (
    <PropertiesCard>
      <PropertyItem label="Content ID" value={contentId} />
      <PropertyItem label="File ID" value={fileId} />
      <PropertyItem label="Vector Store ID" value={vectorStoreId} />
      <PropertyItem label="Object Type" value={content.object} />
      <PropertyItem
        label="Created"
        value={new Date(content.created_timestamp * 1000).toLocaleString()}
      />
      <PropertyItem
        label="Content Length"
        value={`${getTextFromContent(content.content).length} chars`}
      />
      {content.metadata.chunk_window && (
        <PropertyItem label="Position" value={content.metadata.chunk_window} />
      )}
      {file && (
        <>
          <PropertyItem label="File Status" value={file.status} />
          <PropertyItem
            label="File Usage"
            value={`${file.usage_bytes} bytes`}
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
