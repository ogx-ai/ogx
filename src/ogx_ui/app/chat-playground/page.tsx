"use client";

import { Suspense, useState, useEffect, useCallback, useRef } from "react";
import { useSearchParams } from "next/navigation";
import { flushSync } from "react-dom";
import { Button } from "@/components/ui/button";
import {
  Select,
  SelectContent,
  SelectItem,
  SelectTrigger,
  SelectValue,
} from "@/components/ui/select";
import { Input } from "@/components/ui/input";
import { Chat } from "@/components/chat-playground/chat";
import {
  type Message,
  type FileCitation,
} from "@/components/chat-playground/chat-message";
import { useAuthClient } from "@/hooks/use-auth-client";
import type { Model } from "llama-stack-client/resources/models";
import type { ResponseCreateParamsStreaming } from "llama-stack-client/resources/responses/responses";
import { addConversation, getConversation } from "@/lib/conversation-history";

type ModelWithMeta = Model & {
  custom_metadata?: Record<string, unknown>;
};

type VectorStoreInfo = {
  id: string;
  name: string;
};

type ToolsConfig = {
  webSearch: boolean;
  fileSearch: {
    enabled: boolean;
    vectorStoreIds: string[];
  };
  mcp: {
    enabled: boolean;
    serverLabel: string;
    serverUrl: string;
  };
};

type SimpleSession = {
  id: string;
  name: string;
  messages: Message[];
  selectedModel: string;
  systemInstructions: string;
  conversationId?: string;
  createdAt: number;
  updatedAt: number;
};

export default function ChatPlaygroundPage() {
  return (
    <Suspense
      fallback={
        <div className="flex items-center justify-center h-full">
          <p className="text-muted-foreground">Loading...</p>
        </div>
      }
    >
      <ChatPlaygroundContent />
    </Suspense>
  );
}

function ChatPlaygroundContent() {
  const searchParams = useSearchParams();
  const conversationParam = searchParams.get("conversation");

  const [currentSession, setCurrentSession] = useState<SimpleSession | null>(
    null
  );
  const [input, setInput] = useState("");
  const [isGenerating, setIsGenerating] = useState(false);
  const [error, setError] = useState<string | null>(null);
  const [loadingConversation, setLoadingConversation] = useState(false);

  const [models, setModels] = useState<Model[]>([]);
  const [selectedModel, setSelectedModel] = useState<string>("");
  const [modelsLoading, setModelsLoading] = useState(true);
  const [modelsError, setModelsError] = useState<string | null>(null);

  const [systemInstructions, setSystemInstructions] = useState<string>(
    "You are a helpful assistant."
  );

  // Tools configuration
  const [toolsConfig, setToolsConfig] = useState<ToolsConfig>({
    webSearch: false,
    fileSearch: { enabled: false, vectorStoreIds: [] },
    mcp: { enabled: false, serverLabel: "", serverUrl: "" },
  });

  // Available vector stores for file search
  const [vectorStores, setVectorStores] = useState<VectorStoreInfo[]>([]);
  const [selectedVectorStore, setSelectedVectorStore] = useState<string>("");

  const client = useAuthClient();
  const abortControllerRef = useRef<AbortController | null>(null);

  const createNewSession = useCallback(() => {
    const newSession: SimpleSession = {
      id: Date.now().toString(),
      name: "New Conversation",
      messages: [],
      selectedModel,
      systemInstructions,
      createdAt: Date.now(),
      updatedAt: Date.now(),
    };
    setCurrentSession(newSession);
  }, [selectedModel, systemInstructions]);

  // Load models
  useEffect(() => {
    const fetchModels = async () => {
      try {
        setModelsLoading(true);
        const result = await client.models.list();
        const allModels: Model[] = Array.isArray(result)
          ? result
          : (result as { data: Model[] }).data || [];

        const llmModels = allModels.filter(
          (m: ModelWithMeta) =>
            m.custom_metadata?.model_type === "llm" ||
            !m.custom_metadata?.model_type
        );

        llmModels.sort((a, b) => a.id.localeCompare(b.id));
        setModels(llmModels);
      } catch (err) {
        console.error("Error fetching models:", err);
        setModelsError("Failed to load models");
      } finally {
        setModelsLoading(false);
      }
    };

    fetchModels();
  }, [client]);

  // Load vector stores for file search tool
  useEffect(() => {
    const fetchVectorStores = async () => {
      try {
        const result = await client.vectorStores.list({
          limit: 100,
          order: "desc",
        });
        const stores =
          (result as { data?: Record<string, unknown>[] }).data || [];
        setVectorStores(
          stores.map((s: Record<string, unknown>) => ({
            id: s.id as string,
            name: (s.name as string) || (s.id as string),
          }))
        );
      } catch {
        // Vector stores may not be available — that's fine
      }
    };

    fetchVectorStores();
  }, [client]);

  useEffect(() => {
    if (selectedModel && !currentSession && !conversationParam) {
      createNewSession();
    }
  }, [selectedModel, currentSession, createNewSession, conversationParam]);

  // Load an existing conversation from URL param
  useEffect(() => {
    if (
      !conversationParam ||
      currentSession?.conversationId === conversationParam
    )
      return;

    const loadConversation = async () => {
      setLoadingConversation(true);
      setError(null);

      try {
        // Restore saved config from localStorage
        const saved = getConversation(conversationParam);
        if (saved?.toolsConfig) {
          setToolsConfig(saved.toolsConfig);
        }
        if (saved?.systemInstructions) {
          setSystemInstructions(saved.systemInstructions);
        }
        if (saved?.model) {
          setSelectedModel(saved.model);
        }

        // Fetch conversation items from API
        const result = await client.conversations.items.list(conversationParam);
        const itemList = Array.isArray(result)
          ? result
          : (result as { data?: Record<string, unknown>[] }).data || [];

        // Convert items to messages
        const messages: Message[] = [];
        for (const item of itemList) {
          const itemObj = item as Record<string, unknown>;
          const role = itemObj.role as string;
          if (role !== "user" && role !== "assistant") continue;

          let text = "";
          const content = itemObj.content;
          if (typeof content === "string") {
            text = content;
          } else if (Array.isArray(content)) {
            for (const part of content as Record<string, unknown>[]) {
              if (part.text) text += part.text as string;
            }
          }
          if (!text) continue;

          messages.push({
            id: (itemObj.id as string) || `${Date.now()}-${messages.length}`,
            role,
            content: text,
            createdAt: new Date(),
          });
        }

        // Items come back newest-first — reverse to chronological order
        messages.reverse();

        setCurrentSession({
          id: conversationParam,
          name: "Loaded Conversation",
          messages,
          selectedModel: saved?.model || selectedModel || "",
          systemInstructions: saved?.systemInstructions || systemInstructions,
          conversationId: conversationParam,
          createdAt: Date.now(),
          updatedAt: Date.now(),
        });
      } catch (err) {
        console.error("Failed to load conversation:", err);
        setError(
          "Failed to load conversation. It may have been deleted. Starting a new chat."
        );
        // Fall back to a new session
        createNewSession();
      } finally {
        setLoadingConversation(false);
      }
    };

    loadConversation();
  }, [
    conversationParam,
    client,
    selectedModel,
    systemInstructions,
    currentSession?.conversationId,
  ]);

  const handleModelChange = (modelId: string) => {
    setSelectedModel(modelId);
    setCurrentSession(null);
    setError(null);
  };

  const clearChat = () => {
    setCurrentSession(null);
    setError(null);
  };

  // Build tools array from config
  const buildTools = (): ResponseCreateParamsStreaming["tools"] => {
    const tools: NonNullable<ResponseCreateParamsStreaming["tools"]> = [];

    if (toolsConfig.webSearch) {
      tools.push({ type: "web_search" });
    }

    if (
      toolsConfig.fileSearch.enabled &&
      toolsConfig.fileSearch.vectorStoreIds.length > 0
    ) {
      tools.push({
        type: "file_search",
        vector_store_ids: toolsConfig.fileSearch.vectorStoreIds,
      });
    }

    if (
      toolsConfig.mcp.enabled &&
      toolsConfig.mcp.serverLabel &&
      toolsConfig.mcp.serverUrl
    ) {
      tools.push({
        type: "mcp",
        server_label: toolsConfig.mcp.serverLabel,
        server_url: toolsConfig.mcp.serverUrl,
      });
    }

    return tools.length > 0 ? tools : undefined;
  };

  const handleSubmit = async (e?: { preventDefault?: () => void }) => {
    e?.preventDefault?.();
    if (!input.trim()) return;
    if (!selectedModel) {
      setError("Please select a model first.");
      return;
    }

    const userMessage: Message = {
      id: Date.now().toString(),
      role: "user",
      content: input.trim(),
      createdAt: new Date(),
    };

    setCurrentSession(prev => {
      if (!prev) return null;
      return {
        ...prev,
        messages: [...prev.messages, userMessage],
        updatedAt: Date.now(),
      };
    });
    setInput("");

    await handleSubmitWithContent(userMessage.content);
  };

  const currentSessionRef = useRef(currentSession);
  currentSessionRef.current = currentSession;
  const selectedModelRef = useRef(selectedModel);
  selectedModelRef.current = selectedModel;
  const toolsConfigRef = useRef(toolsConfig);
  toolsConfigRef.current = toolsConfig;

  const handleSubmitWithContent = async (content: string) => {
    const session = currentSessionRef.current;
    const model = selectedModelRef.current;
    if (!session || !model) return;

    setIsGenerating(true);
    setError(null);

    if (abortControllerRef.current) {
      abortControllerRef.current.abort();
    }

    const abortController = new AbortController();
    abortControllerRef.current = abortController;

    try {
      // Create a conversation on first message if we don't have one
      let conversationId = session.conversationId;
      if (!conversationId) {
        const conv = await client.conversations.create({});
        conversationId = conv.id;
        setCurrentSession(prev => (prev ? { ...prev, conversationId } : prev));
        // Save to localStorage for sidebar history (including config)
        addConversation({
          id: conversationId,
          createdAt: Date.now(),
          firstMessage: content,
          model,
          systemInstructions: session.systemInstructions,
          toolsConfig: toolsConfigRef.current,
        });
      }

      const tools = buildTools();

      const requestParams: ResponseCreateParamsStreaming = {
        model: model,
        input: content,
        stream: true,
        conversation: conversationId,
        instructions: session.systemInstructions,
        ...(tools && { tools }),
      };

      const response = await client.responses.create(requestParams, {
        signal: abortController.signal,
        timeout: 300000,
      } as { signal: AbortSignal; timeout: number });

      let fullContent = "";
      let currentResponseId: string | null = null;
      let assistantMessageAdded = false;
      const collectedAnnotations: FileCitation[] = [];

      const updateAssistantMessage = (
        content: string,
        annotations?: FileCitation[]
      ) => {
        setCurrentSession(prev => {
          if (!prev) return null;
          const newMessages = [...prev.messages];
          const last = newMessages[newMessages.length - 1];
          if (last.role === "assistant") {
            last.content = content;
            if (annotations && annotations.length > 0) {
              last.annotations = annotations;
            }
          }
          return {
            ...prev,
            messages: newMessages,
            updatedAt: Date.now(),
          };
        });
      };

      if (Symbol.asyncIterator in response) {
        for await (const chunk of response) {
          if (abortController.signal.aborted) break;

          const chunkObj = chunk as Record<string, unknown>;

          if (chunkObj.type === "response.created" && chunkObj.response) {
            currentResponseId = (chunkObj.response as { id: string }).id;
          }

          if (chunkObj.type === "response.failed") {
            const errResponse = chunkObj.response as
              | {
                  error?: { message?: string };
                }
              | undefined;
            const failMsg =
              errResponse?.error?.message || "Response generation failed";
            setError(failMsg);
            // Don't clear conversationId on failure — the conversation is still valid
            break;
          }

          // Capture annotation events
          if (chunkObj.type === "response.output_text.annotation.added") {
            const ann = chunkObj.annotation as Record<string, unknown>;
            if (ann?.type === "file_citation") {
              collectedAnnotations.push({
                type: "file_citation",
                file_id: (ann.file_id as string) || "",
                filename: (ann.filename as string) || "",
                index: (ann.index as number) || 0,
              });
            }
          }

          // Capture annotations from content_part.done
          if (chunkObj.type === "response.content_part.done") {
            const part = chunkObj.part as Record<string, unknown>;
            const partAnnotations = part?.annotations as
              | Array<Record<string, unknown>>
              | undefined;
            if (partAnnotations) {
              for (const ann of partAnnotations) {
                if (ann.type === "file_citation") {
                  collectedAnnotations.push({
                    type: "file_citation",
                    file_id: (ann.file_id as string) || "",
                    filename: (ann.filename as string) || "",
                    index: (ann.index as number) || 0,
                  });
                }
              }
            }
          }

          const deltaText =
            chunkObj.type === "response.output_text.delta"
              ? (chunkObj.delta as string) || (chunkObj.text as string)
              : null;

          if (deltaText) {
            fullContent += deltaText;

            if (!assistantMessageAdded) {
              assistantMessageAdded = true;
              setCurrentSession(prev => {
                if (!prev) return null;
                return {
                  ...prev,
                  messages: [
                    ...prev.messages,
                    {
                      id: (Date.now() + 1).toString(),
                      role: "assistant" as const,
                      content: fullContent,
                      createdAt: new Date(),
                    },
                  ],
                  updatedAt: Date.now(),
                };
              });
            } else {
              flushSync(() => {
                updateAssistantMessage(fullContent);
              });
            }
          }

          // On completion, attach annotations to the message
          if (
            chunkObj.type === "response.completed" &&
            collectedAnnotations.length > 0
          ) {
            updateAssistantMessage(fullContent, collectedAnnotations);
          }
        }
      }

      // Response ID tracked for debugging; conversation manages context
    } catch (err) {
      if (err instanceof Error && err.name === "AbortError") {
        return;
      }

      console.error("Error sending message:", err);
      const errMsg =
        err instanceof Error ? err.message : "Unknown error occurred";
      setError(`Failed to send message: ${errMsg}`);
      setCurrentSession(prev =>
        prev
          ? {
              ...prev,
              messages: prev.messages.slice(0, -1),
              updatedAt: Date.now(),
            }
          : prev
      );
    } finally {
      setIsGenerating(false);
      abortControllerRef.current = null;
    }
  };

  const suggestions = [
    "Write a Python function that prints 'Hello, World!'",
    "Explain step-by-step how to solve this math problem: If x² + 6x + 9 = 25, what is x?",
    "Design a simple algorithm to find the longest palindrome in a string.",
  ];

  const append = (message: { role: "user"; content: string }) => {
    const newMessage: Message = {
      id: Date.now().toString(),
      role: "user",
      content: message.content,
      createdAt: new Date(),
    };

    setCurrentSession(prev => {
      if (!prev) return null;
      return {
        ...prev,
        messages: [...prev.messages, newMessage],
        updatedAt: Date.now(),
      };
    });
    handleSubmitWithContent(message.content);
  };

  const isModelsLoading = modelsLoading ?? true;

  const addVectorStore = () => {
    if (
      selectedVectorStore &&
      !toolsConfig.fileSearch.vectorStoreIds.includes(selectedVectorStore)
    ) {
      setToolsConfig(prev => ({
        ...prev,
        fileSearch: {
          ...prev.fileSearch,
          vectorStoreIds: [
            ...prev.fileSearch.vectorStoreIds,
            selectedVectorStore,
          ],
        },
      }));
      setSelectedVectorStore("");
    }
  };

  const removeVectorStore = (id: string) => {
    setToolsConfig(prev => ({
      ...prev,
      fileSearch: {
        ...prev.fileSearch,
        vectorStoreIds: prev.fileSearch.vectorStoreIds.filter(v => v !== id),
      },
    }));
  };

  return (
    <div className="flex flex-col h-[calc(100vh-4rem)]">
      {/* Header */}
      <div className="flex-none p-4 border-b">
        <div className="flex items-center gap-4">
          <h1 className="text-2xl font-bold">Chat Playground</h1>
          <Button variant="outline" onClick={clearChat} disabled={isGenerating}>
            Clear Chat
          </Button>
        </div>
      </div>

      {/* Main Two-Column Layout */}
      <div className="flex flex-1 gap-6 min-h-0 flex-col lg:flex-row p-4">
        {/* Left Column - Settings Panel */}
        <div className="w-full lg:w-80 lg:flex-shrink-0 space-y-6 p-4 border border-border rounded-lg bg-muted/30 overflow-y-auto">
          <h2 className="text-lg font-semibold border-b pb-2 text-left">
            Settings
          </h2>

          {/* Model Configuration */}
          <div className="space-y-4 text-left">
            <h3 className="text-lg font-semibold border-b pb-2 text-left">
              Model Configuration
            </h3>
            <div className="space-y-3">
              <div>
                <label className="text-sm font-medium block mb-2">Model</label>
                <Select
                  value={selectedModel}
                  onValueChange={handleModelChange}
                  disabled={isModelsLoading || isGenerating}
                >
                  <SelectTrigger className="w-full">
                    <SelectValue
                      placeholder={
                        isModelsLoading ? "Loading..." : "Select Model"
                      }
                    />
                  </SelectTrigger>
                  <SelectContent>
                    {models.map(model => (
                      <SelectItem key={model.id} value={model.id}>
                        {model.id}
                      </SelectItem>
                    ))}
                  </SelectContent>
                </Select>
                {modelsError && (
                  <p className="text-destructive text-xs mt-1">{modelsError}</p>
                )}
              </div>

              <div>
                <label className="text-sm font-medium block mb-2">
                  System Instructions
                </label>
                <textarea
                  className="w-full h-24 px-3 py-2 text-sm border border-input rounded-md bg-background resize-none focus:outline-none focus:ring-2 focus:ring-ring"
                  value={systemInstructions}
                  onChange={e => setSystemInstructions(e.target.value)}
                  placeholder="Enter system instructions..."
                  disabled={isGenerating}
                />
                <p className="text-xs text-muted-foreground mt-1">
                  Instructions sent as the system message for new conversations.
                </p>
              </div>
            </div>
          </div>

          {/* Tools Configuration */}
          <div className="space-y-4 text-left">
            <h3 className="text-lg font-semibold border-b pb-2 text-left">
              Tools
            </h3>
            <div className="space-y-4">
              {/* Web Search */}
              <label className="flex items-center gap-3 cursor-pointer">
                <input
                  type="checkbox"
                  checked={toolsConfig.webSearch}
                  onChange={e =>
                    setToolsConfig(prev => ({
                      ...prev,
                      webSearch: e.target.checked,
                    }))
                  }
                  disabled={isGenerating}
                  className="h-4 w-4 rounded border-input"
                />
                <div>
                  <span className="text-sm font-medium">Web Search</span>
                  <p className="text-xs text-muted-foreground">
                    Search the web for current information
                  </p>
                </div>
              </label>

              {/* File Search */}
              <div className="space-y-2">
                <label className="flex items-center gap-3 cursor-pointer">
                  <input
                    type="checkbox"
                    checked={toolsConfig.fileSearch.enabled}
                    onChange={e =>
                      setToolsConfig(prev => ({
                        ...prev,
                        fileSearch: {
                          ...prev.fileSearch,
                          enabled: e.target.checked,
                        },
                      }))
                    }
                    disabled={isGenerating}
                    className="h-4 w-4 rounded border-input"
                  />
                  <div>
                    <span className="text-sm font-medium">File Search</span>
                    <p className="text-xs text-muted-foreground">
                      Search across vector stores
                    </p>
                  </div>
                </label>

                {toolsConfig.fileSearch.enabled && (
                  <div className="ml-7 space-y-2">
                    <div className="flex gap-2">
                      <Select
                        value={selectedVectorStore}
                        onValueChange={setSelectedVectorStore}
                        disabled={isGenerating}
                      >
                        <SelectTrigger className="flex-1">
                          <SelectValue
                            placeholder={
                              vectorStores.length === 0
                                ? "No vector stores"
                                : "Select vector store"
                            }
                          />
                        </SelectTrigger>
                        <SelectContent>
                          {vectorStores.map(vs => (
                            <SelectItem key={vs.id} value={vs.id}>
                              {vs.name}
                            </SelectItem>
                          ))}
                        </SelectContent>
                      </Select>
                      <Button
                        variant="outline"
                        size="sm"
                        onClick={addVectorStore}
                        disabled={!selectedVectorStore || isGenerating}
                      >
                        Add
                      </Button>
                    </div>
                    {toolsConfig.fileSearch.vectorStoreIds.length > 0 && (
                      <div className="space-y-1">
                        {toolsConfig.fileSearch.vectorStoreIds.map(id => (
                          <div
                            key={id}
                            className="flex items-center justify-between px-2 py-1 bg-muted rounded text-xs"
                          >
                            <span className="truncate font-mono">{id}</span>
                            <button
                              onClick={() => removeVectorStore(id)}
                              className="text-muted-foreground hover:text-destructive ml-2"
                              disabled={isGenerating}
                            >
                              ×
                            </button>
                          </div>
                        ))}
                      </div>
                    )}
                    {toolsConfig.fileSearch.enabled &&
                      toolsConfig.fileSearch.vectorStoreIds.length === 0 && (
                        <p className="text-xs text-amber-600 dark:text-amber-400">
                          Add at least one vector store to use file search.
                        </p>
                      )}
                  </div>
                )}
              </div>

              {/* MCP */}
              <div className="space-y-2">
                <label className="flex items-center gap-3 cursor-pointer">
                  <input
                    type="checkbox"
                    checked={toolsConfig.mcp.enabled}
                    onChange={e =>
                      setToolsConfig(prev => ({
                        ...prev,
                        mcp: { ...prev.mcp, enabled: e.target.checked },
                      }))
                    }
                    disabled={isGenerating}
                    className="h-4 w-4 rounded border-input"
                  />
                  <div>
                    <span className="text-sm font-medium">MCP Server</span>
                    <p className="text-xs text-muted-foreground">
                      Connect to a Model Context Protocol server
                    </p>
                  </div>
                </label>

                {toolsConfig.mcp.enabled && (
                  <div className="ml-7 space-y-2">
                    <Input
                      placeholder="Server label (e.g. my-tools)"
                      value={toolsConfig.mcp.serverLabel}
                      onChange={e =>
                        setToolsConfig(prev => ({
                          ...prev,
                          mcp: { ...prev.mcp, serverLabel: e.target.value },
                        }))
                      }
                      disabled={isGenerating}
                      className="text-sm"
                    />
                    <Input
                      placeholder="Server URL (e.g. http://localhost:3001/sse)"
                      value={toolsConfig.mcp.serverUrl}
                      onChange={e =>
                        setToolsConfig(prev => ({
                          ...prev,
                          mcp: { ...prev.mcp, serverUrl: e.target.value },
                        }))
                      }
                      disabled={isGenerating}
                      className="text-sm"
                    />
                  </div>
                )}
              </div>
            </div>
          </div>
        </div>

        {/* Right Column - Chat Interface */}
        <div className="flex-1 flex flex-col min-h-0 p-4 border border-border rounded-lg bg-background">
          {error && (
            <div className="mb-4 p-3 bg-destructive/10 border border-destructive/20 rounded-md">
              <p className="text-destructive text-sm">{error}</p>
            </div>
          )}

          {loadingConversation ? (
            <div className="flex items-center justify-center h-full">
              <p className="text-muted-foreground">Loading conversation...</p>
            </div>
          ) : currentSession ? (
            <Chat
              messages={currentSession.messages}
              input={input}
              handleInputChange={e => setInput(e.target.value)}
              handleSubmit={handleSubmit}
              isGenerating={isGenerating}
              suggestions={suggestions}
              append={append}
            />
          ) : (
            <div className="flex items-center justify-center h-full">
              <p className="text-muted-foreground">
                Select a model to start chatting
              </p>
            </div>
          )}
        </div>
      </div>
    </div>
  );
}
