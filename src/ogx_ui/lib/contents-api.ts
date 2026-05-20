import type { FileContentResponse } from "ogx-client/resources/vector-stores/files";
import type { OgxClient } from "ogx-client";

export type VectorStoreContent = FileContentResponse.Content;
export type VectorStoreContentsResponse = FileContentResponse;

export interface VectorStoreContentItem {
  id: string;
  object: string;
  created_timestamp: number;
  vector_store_id: string;
  file_id: string;
  content: VectorStoreContent;
  metadata: Record<string, unknown>;
  embedding?: number[];
}

export interface VectorStoreContentDeleteResponse {
  id: string;
  object: string;
  deleted: boolean;
}

export interface VectorStoreListContentsResponse {
  object: string;
  data: VectorStoreContentItem[];
  first_id?: string;
  last_id?: string;
  has_more: boolean;
}

export class ContentsAPI {
  constructor(private client: OgxClient) {}

  async getFileContents(
    vectorStoreId: string,
    fileId: string,
    includeEmbeddings: boolean = true,
    includeMetadata: boolean = true
  ): Promise<VectorStoreContentsResponse> {
    try {
      // Use query parameters to pass embeddings and metadata flags (OpenAI-compatible pattern)
      const extraQuery: Record<string, boolean> = {};
      if (includeEmbeddings) {
        extraQuery.include_embeddings = true;
      }
      if (includeMetadata) {
        extraQuery.include_metadata = true;
      }

      const result = await this.client.vectorStores.files.content(
        vectorStoreId,
        fileId,
        {
          query: {
            include_embeddings: includeEmbeddings,
            include_metadata: includeMetadata,
          },
        }
      );
      return result;
    } catch (error) {
      console.error("ContentsAPI.getFileContents error:", error);
      throw error;
    }
  }

  async getContent(
    vectorStoreId: string,
    fileId: string,
    contentId: string
  ): Promise<VectorStoreContentItem> {
    const contentsResponse = await this.listContents(vectorStoreId, fileId);
    const targetContent = contentsResponse.data.find(c => c.id === contentId);

    if (!targetContent) {
      throw new Error(`Content ${contentId} not found`);
    }

    return targetContent;
  }

  async updateContent(): Promise<VectorStoreContentItem> {
    throw new Error("Individual content updates not yet implemented in API");
  }

  async deleteContent(): Promise<VectorStoreContentDeleteResponse> {
    throw new Error("Individual content deletion not yet implemented in API");
  }

  async listContents(
    vectorStoreId: string,
    fileId: string,
    options?: {
      limit?: number;
      order?: string;
      after?: string;
      before?: string;
      includeEmbeddings?: boolean;
      includeMetadata?: boolean;
    }
  ): Promise<VectorStoreListContentsResponse> {
    const fileContents = await this.getFileContents(
      vectorStoreId,
      fileId,
      options?.includeEmbeddings ?? true,
      options?.includeMetadata ?? true
    );
    const contentItems: VectorStoreContentItem[] = [];

    (fileContents.data ?? []).forEach((item, contentIndex) => {
      const raw = item as Record<string, unknown>;
      const chunkMeta = (raw.chunk_metadata ?? {}) as Record<string, unknown>;
      const contentId =
        chunkMeta.chunk_id || raw.id || `content_${fileId}_${contentIndex}`;
      contentItems.push({
        id: contentId as string,
        object: (raw.object as string) || "vector_store.file.content",
        created_timestamp:
          (raw.created_timestamp as number) ||
          (raw.created_at as number) ||
          Date.now() / 1000,
        vector_store_id: vectorStoreId,
        file_id: fileId,
        content: item,
        embedding: raw.embedding as number[] | undefined,
        metadata: {
          ...chunkMeta,
          content_length: item.type === "text" ? item.text.length : 0,
        },
      });
    });

    // apply pagination if needed
    let filteredItems = contentItems;
    if (options?.limit) {
      filteredItems = filteredItems.slice(0, options.limit);
    }

    return {
      object: "list",
      data: filteredItems,
      has_more: contentItems.length > (options?.limit || contentItems.length),
    };
  }
}
