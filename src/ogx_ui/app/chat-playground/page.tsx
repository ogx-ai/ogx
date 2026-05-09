"use client";

import { useState, useEffect, useCallback, useRef } from "react";
import { flushSync } from "react-dom";
import { Button } from "@/components/ui/button";
import {
  Select,
  SelectContent,
  SelectItem,
  SelectTrigger,
  SelectValue,
} from "@/components/ui/select";
import { Card } from "@/components/ui/card";
import { Input } from "@/components/ui/input";
import { Trash2 } from "lucide-react";
import { Chat } from "@/components/chat-playground/chat";
import { type Message } from "@/components/chat-playground/chat-message";
import { VectorDBCreator } from "@/components/chat-playground/vector-db-creator";
import { useAuthClient } from "@/hooks/use-auth-client";
import type { Model } from "llama-stack-client/resources/models";
import type { TurnCreateParams } from "llama-stack-client/resources/agents/turn";

// Extended Model type to include properties from API response
type ModelWithMetadata = Model & {
  id: string;
  custom_metadata?: {
    model_type?: string;
    [key: string]: unknown;
  };
};
import {
  SessionUtils,
  type ChatSession,
} from "@/components/chat-playground/conversations";
import {
  cleanMessageContent,
  extractCleanText,
} from "@/lib/message-content-utils";
export default function ChatPlaygroundPage() {
  const [currentSession, setCurrentSession] = useState<ChatSession | null>(
    null
  );
  const [input, setInput] = useState("");
  const [isGenerating, setIsGenerating] = useState(false);
  const [error, setError] = useState<string | null>(null);
  const [models, setModels] = useState<Model[]>([]);
  const [selectedModel, setSelectedModel] = useState<string>("");
  const [modelsLoading, setModelsLoading] = useState(true);
  const [modelsError, setModelsError] = useState<string | null>(null);
  const [agents, setAgents] = useState<
    Array<{
      agent_id: string;
      agent_config?: {
        agent_name?: string;
        name?: string;
        instructions?: string;
      };
      [key: string]: unknown;
    }>
  >([]);
  const [selectedAgentConfig, setSelectedAgentConfig] = useState<{
    toolgroups?: Array<
      string | { name: string; args: Record<string, unknown> }
    >;
  } | null>(null);
  const [selectedAgentId, setSelectedAgentId] = useState<string>("");
  const [agentsLoading, setAgentsLoading] = useState(true);
  const [showCreateAgent, setShowCreateAgent] = useState(false);
  const [newAgentName, setNewAgentName] = useState("");
  const [newAgentInstructions, setNewAgentInstructions] = useState(
    "You are a helpful assistant."
  );
  const [selectedToolgroups, setSelectedToolgroups] = useState<string[]>([]);
  const [availableToolgroups, setAvailableToolgroups] = useState<
    Array<{
      identifier: string;
      provider_id: string;
      type: string;
      provider_resource_id?: string;
    }>
  >([]);
  const [showCreateVectorDB, setShowCreateVectorDB] = useState(false);
  const [availableVectorDBs, setAvailableVectorDBs] = useState<
    Array<{
      identifier: string;
      vector_db_name?: string;
      embedding_model: string;
    }>
  >([]);
  const [uploadNotification, setUploadNotification] = useState<{
    show: boolean;
    message: string;
    type: "success" | "error" | "loading";
  }>({ show: false, message: "", type: "success" });
  const [selectedVectorDBs, setSelectedVectorDBs] = useState<string[]>([]);
  const client = useAuthClient();
  const abortControllerRef = useRef<AbortController | null>(null);
  const lastResponseIdRef = useRef<string | null>(null);

  const isModelsLoading = modelsLoading ?? true;

  const loadAgentConfig = useCallback(
    async (agentId: string) => {
      try {
        // try to load from cache first
        const cachedConfig = SessionUtils.loadAgentConfig(agentId);
        if (cachedConfig) {
          setSelectedAgentConfig({
            toolgroups: cachedConfig.toolgroups,
          });
          return;
        }

        const agentDetails = await client.agents.retrieve(agentId);

        // cache config
        SessionUtils.saveAgentConfig(agentId, {
          ...agentDetails.agent_config,
          toolgroups: agentDetails.agent_config?.toolgroups,
        });

        setSelectedAgentConfig({
          toolgroups: agentDetails.agent_config?.toolgroups,
        });
      } catch (error) {
        console.error("Error loading agent config:", error);
        setSelectedAgentConfig(null);
      }
    },
    [client]
  );

  const createDefaultSession = useCallback(
    async (agentId: string) => {
      try {
        const response = await client.agents.session.create(agentId, {
          session_name: "Default Session",
        });

        const defaultSession: ChatSession = {
          id: response.session_id,
          name: "Default Session",
          messages: [],
          selectedModel: selectedModel, // use current selected model
          systemMessage: "You are a helpful assistant.",
          agentId,
          createdAt: Date.now(),
          updatedAt: Date.now(),
        };

        setCurrentSession(defaultSession);
        SessionUtils.saveCurrentSessionId(defaultSession.id, agentId);
        // cache entire session data
        SessionUtils.saveSessionData(agentId, defaultSession);
      } catch (error) {
        console.error("Error creating default session:", error);
      }
    },
    [client, selectedModel]
  );

  const loadSessionMessages = useCallback(
    async (agentId: string, sessionId: string): Promise<Message[]> => {
      try {
        const session = await client.agents.session.retrieve(
          agentId,
          sessionId
        );

        if (!session || !session.turns || !Array.isArray(session.turns)) {
          return [];
        }

        const messages: Message[] = [];
        for (const turn of session.turns) {
          if (turn.input_messages && Array.isArray(turn.input_messages)) {
            for (const input of turn.input_messages) {
              if (input.role === "user" && input.content) {
                messages.push({
                  id: `${turn.turn_id}-user-${messages.length}`,
                  role: "user",
                  content:
                    typeof input.content === "string"
                      ? input.content
                      : JSON.stringify(input.content),
                  createdAt: new Date(turn.started_at || Date.now()),
                });
              }
            }
          }

          if (turn.output_message && turn.output_message.content) {
            console.log("Raw message content:", turn.output_message.content);
            console.log("Content type:", typeof turn.output_message.content);

            const cleanContent = cleanMessageContent(
              turn.output_message.content
            );

            messages.push({
              id: `${turn.turn_id}-assistant-${messages.length}`,
              role: "assistant",
              content: cleanContent,
              createdAt: new Date(
                turn.completed_at || turn.started_at || Date.now()
              ),
            });
          }
        }

        return messages;
      } catch (error) {
        console.error("Error loading session messages:", error);
        return [];
      }
    },
    [client]
  );

  const loadAgentSessions = useCallback(
    async (agentId: string) => {
      try {
        const response = await client.agents.session.list(agentId);

        if (
          response.data &&
          Array.isArray(response.data) &&
          response.data.length > 0
        ) {
          // check for saved session ID for this agent
          const savedSessionId = SessionUtils.loadCurrentSessionId(agentId);
          // try to load cached agent session data first
          if (savedSessionId) {
            const cachedSession = SessionUtils.loadSessionData(
              agentId,
              savedSessionId
            );
            if (cachedSession) {
              setCurrentSession(cachedSession);
              SessionUtils.saveCurrentSessionId(cachedSession.id, agentId);
              return;
            }
            console.log("📡 Cache miss, fetching session from API...");
          }

          let sessionToLoad = response.data[0] as {
            session_id: string;
            session_name?: string;
            started_at?: string;
          };
          console.log(
            "Default session to load (first in list):",
            sessionToLoad.session_id
          );

          // try to find saved session id in available sessions
          if (savedSessionId) {
            const foundSession = response.data.find(
              (s: { [key: string]: unknown }) =>
                (s as { session_id: string }).session_id === savedSessionId
            );
            console.log("Found saved session in list:", foundSession);
            if (foundSession) {
              sessionToLoad = foundSession as {
                session_id: string;
                session_name?: string;
                started_at?: string;
              };
              console.log(
                "✅ Restored previously selected session:",
                savedSessionId
              );
            } else {
              console.log(
                "❌ Previously selected session not found, using latest session"
              );
            }
          } else {
            console.log("❌ No saved session ID found, using latest session");
          }

          const messages = await loadSessionMessages(
            agentId,
            sessionToLoad.session_id
          );

          const session: ChatSession = {
            id: sessionToLoad.session_id,
            name: sessionToLoad.session_name || "Session",
            messages,
            selectedModel: selectedModel || "",
            systemMessage: "You are a helpful assistant.",
            agentId,
            createdAt: sessionToLoad.started_at
              ? new Date(sessionToLoad.started_at).getTime()
              : Date.now(),
            updatedAt: Date.now(),
          };

          setCurrentSession(session);
          console.log(`💾 Saving session ID for agent ${agentId}:`, session.id);
          SessionUtils.saveCurrentSessionId(session.id, agentId);
          // cache session data
          SessionUtils.saveSessionData(agentId, session);
        } else {
          // no sessions, create a new one
          await createDefaultSession(agentId);
        }
      } catch (error) {
        console.error("Error loading agent sessions:", error);
        // fallback to creating a new session
        await createDefaultSession(agentId);
      }
    },
    [client, loadSessionMessages, createDefaultSession, selectedModel]
  );

  useEffect(() => {
    const fetchAgents = async () => {
      try {
        setAgentsLoading(true);
        const agentList = await client.agents.list();
        setAgents(
          (agentList.data as Array<{
            agent_id: string;
            agent_config?: {
              agent_name?: string;
              name?: string;
              instructions?: string;
            };
            [key: string]: unknown;
          }>) || []
        );

        if (agentList.data && agentList.data.length > 0) {
          // check if there's a previously selected agent
          const savedAgentId = SessionUtils.loadCurrentAgentId();

          let agentToSelect = agentList.data[0] as {
            agent_id: string;
            agent_config?: {
              agent_name?: string;
              name?: string;
              instructions?: string;
            };
            [key: string]: unknown;
          };

          // if we have a saved agent ID, find it in the available agents
          if (savedAgentId) {
            const foundAgent = agentList.data.find(
              (a: { [key: string]: unknown }) =>
                (a as { agent_id: string }).agent_id === savedAgentId
            );
            if (foundAgent) {
              agentToSelect = foundAgent as typeof agentToSelect;
            } else {
              console.log("Previously slelected agent not found:");
            }
          }
          setSelectedAgentId(agentToSelect.agent_id);
          SessionUtils.saveCurrentAgentId(agentToSelect.agent_id);
          // load agent config immediately
          await loadAgentConfig(agentToSelect.agent_id);
          // Note: loadAgentSessions will be called after models are loaded
        }
      } catch (error) {
        console.error("Error fetching agents (may be unavailable):", error);
        // Agents API not available - load persisted local agents
        const savedAgents = SessionUtils.loadAgentsList();
        if (savedAgents.length > 0) {
          setAgents(savedAgents);
          const savedAgentId = SessionUtils.loadCurrentAgentId();
          const agentToSelect =
            savedAgentId && savedAgents.find(a => a.agent_id === savedAgentId)
              ? savedAgentId
              : savedAgents[0].agent_id;
          setSelectedAgentId(agentToSelect);
          // Load the saved session
          const savedSessionId =
            SessionUtils.loadActiveSessionId(agentToSelect);
          if (savedSessionId) {
            const savedSession = SessionUtils.loadSessionData(
              agentToSelect,
              savedSessionId
            );
            if (savedSession) {
              setCurrentSession(savedSession);
              lastResponseIdRef.current = null;
              // Load instructions
              const agent = savedAgents.find(a => a.agent_id === agentToSelect);
              if (agent?.agent_config?.instructions) {
                setNewAgentInstructions(agent.agent_config.instructions);
              }
            }
          }
          if (!currentSession) {
            // No saved session found, create a fresh one
            const sessionId = `session-${Date.now()}`;
            const localSession: ChatSession = {
              id: sessionId,
              agentId: agentToSelect,
              name:
                savedAgents.find(a => a.agent_id === agentToSelect)
                  ?.agent_config?.name || "Chat",
              messages: [],
              createdAt: Date.now(),
              updatedAt: Date.now(),
            };
            setCurrentSession(localSession);
            SessionUtils.saveActiveSessionId(agentToSelect, sessionId);
            lastResponseIdRef.current = null;
          }
        } else {
          // No saved agents - create a default one
          const defaultAgent = {
            agent_id: `chat-${Date.now()}`,
            agent_config: {
              name: "New Chat",
              instructions: "You are a helpful assistant.",
            },
          };
          setAgents([defaultAgent]);
          SessionUtils.saveAgentsList([defaultAgent]);
          setSelectedAgentId(defaultAgent.agent_id);
          SessionUtils.saveCurrentAgentId(defaultAgent.agent_id);
          const sessionId = `session-${Date.now()}`;
          const localSession: ChatSession = {
            id: sessionId,
            agentId: defaultAgent.agent_id,
            name: "New Chat",
            messages: [],
            createdAt: Date.now(),
            updatedAt: Date.now(),
          };
          setCurrentSession(localSession);
          SessionUtils.saveActiveSessionId(defaultAgent.agent_id, sessionId);
          lastResponseIdRef.current = null;
        }
      } finally {
        setAgentsLoading(false);
      }
    };

    fetchAgents();

    const fetchToolgroups = async () => {
      try {
        const toolgroups = await client.toolgroups.list();

        const toolGroupsArray = Array.isArray(toolgroups)
          ? toolgroups
          : toolgroups &&
              typeof toolgroups === "object" &&
              "data" in toolgroups &&
              Array.isArray((toolgroups as { data: unknown }).data)
            ? (
                toolgroups as {
                  data: Array<{
                    identifier: string;
                    provider_id: string;
                    type: string;
                    provider_resource_id?: string;
                  }>;
                }
              ).data
            : [];

        if (toolGroupsArray && Array.isArray(toolGroupsArray)) {
          setAvailableToolgroups(toolGroupsArray);
        } else {
          console.error("Invalid toolgroups data format:", toolgroups);
        }
      } catch (error) {
        console.error(
          "Error fetching toolgroups, falling back to /v1/tools:",
          error
        );
        // Fallback: fetch individual tools and present them as toolgroups
        try {
          const res = await fetch("/api/v1/tools");
          if (res.ok) {
            const data = await res.json();
            const tools = Array.isArray(data) ? data : (data.data ?? []);
            const toolsAsToolgroups = tools.map(
              (t: Record<string, unknown>) => ({
                identifier: (t.identifier ??
                  t.tool_name ??
                  t.name ??
                  "") as string,
                provider_id: (t.provider_id ?? "") as string,
                type: "tool",
                provider_resource_id: (t.provider_resource_id ?? "") as string,
              })
            );
            setAvailableToolgroups(toolsAsToolgroups);
          }
        } catch {
          // both failed, leave empty
        }
      }
    };

    fetchToolgroups();

    const fetchVectorDBs = async () => {
      try {
        const vectorDBs = await client.vectorDBs.list();

        const vectorDBsArray = Array.isArray(vectorDBs) ? vectorDBs : [];

        if (vectorDBsArray && Array.isArray(vectorDBsArray)) {
          setAvailableVectorDBs(vectorDBsArray);
        } else {
          console.error("Invalid vector DBs data format:", vectorDBs);
        }
      } catch (error) {
        console.error("Error fetching vector DBs:", error);
      }
    };

    fetchVectorDBs();
  }, [client, loadAgentSessions, loadAgentConfig]);

  const createNewAgent = useCallback(
    async (
      name: string,
      instructions: string,
      model: string,
      _toolgroups: string[] = [],
      _vectorDBs: string[] = []
    ) => {
      // Create a local responses-mode chat session
      const agentId = `chat-${Date.now()}`;
      const sessionId = `session-${Date.now()}`;

      if (model) {
        setSelectedModel(model);
      }
      setNewAgentInstructions(instructions);

      const newAgent = {
        agent_id: agentId,
        agent_config: {
          name: name || "New Chat",
          instructions,
        },
      };

      setAgents(prev => {
        const updated = [...prev, newAgent];
        SessionUtils.saveAgentsList(updated);
        return updated;
      });

      const localSession: ChatSession = {
        id: sessionId,
        agentId,
        name: name || "New Chat",
        messages: [],
        createdAt: Date.now(),
        updatedAt: Date.now(),
      };

      setSelectedAgentId(agentId);
      setCurrentSession(localSession);
      lastResponseIdRef.current = null;
      SessionUtils.saveCurrentAgentId(agentId);
      SessionUtils.saveActiveSessionId(agentId, sessionId);
      SessionUtils.saveSessionData(agentId, localSession);

      return agentId;
    },
    []
  );

  const handleVectorDBCreated = useCallback(
    // eslint-disable-next-line @typescript-eslint/no-unused-vars
    async (_vectorDbId: string) => {
      setShowCreateVectorDB(false);

      try {
        const vectorDBs = await client.vectorDBs.list();
        const vectorDBsArray = Array.isArray(vectorDBs) ? vectorDBs : [];

        if (vectorDBsArray && Array.isArray(vectorDBsArray)) {
          setAvailableVectorDBs(vectorDBsArray);
        }
      } catch (error) {
        console.error("Error refreshing vector DBs:", error);
      }
    },
    [client]
  );

  const deleteAgent = useCallback(
    async (agentId: string) => {
      if (
        confirm(
          "Are you sure you want to delete this agent? This action cannot be undone and will delete the agent and all its sessions."
        )
      ) {
        try {
          // there's a known error where the delete API returns 500 even on success
          try {
            await client.agents.delete(agentId);
            console.log("Agent deleted successfully");
          } catch (deleteError) {
            // log the error but don't re-throw - we know deletion succeeded
            console.log(
              "Agent delete API returned error (but deletion likely succeeded):",
              deleteError
            );
          }

          SessionUtils.clearAgentCache(agentId);

          const agentList = await client.agents.list();
          setAgents(
            (agentList.data as Array<{
              agent_id: string;
              agent_config?: {
                agent_name?: string;
                name?: string;
                instructions?: string;
              };
              [key: string]: unknown;
            }>) || []
          );

          // if we delete current agent, switch to another
          if (selectedAgentId === agentId) {
            const remainingAgents = agentList.data?.filter(
              (a: { [key: string]: unknown }) =>
                (a as { agent_id: string }).agent_id !== agentId
            );
            if (remainingAgents && remainingAgents.length > 0) {
              const newAgent = remainingAgents[0] as {
                agent_id: string;
                agent_config?: {
                  agent_name?: string;
                  name?: string;
                  instructions?: string;
                };
                [key: string]: unknown;
              };
              setSelectedAgentId(newAgent.agent_id);
              SessionUtils.saveCurrentAgentId(newAgent.agent_id);
              await loadAgentConfig(newAgent.agent_id);
              await loadAgentSessions(newAgent.agent_id);
            } else {
              // no agents left
              setSelectedAgentId("");
              setCurrentSession(null);
              setSelectedAgentConfig(null);
            }
          }
        } catch (error) {
          console.error("Error deleting agent:", error);

          // check if this is known server bug where deletion succeeds but returns 500
          // The error message will typically contain status codes or "Could not find agent"
          const errorMessage =
            error instanceof Error ? error.message : String(error);
          const isKnownServerBug =
            errorMessage.includes("500") ||
            errorMessage.includes("Internal Server Error") ||
            errorMessage.includes("Could not find agent") ||
            errorMessage.includes("400");

          if (isKnownServerBug) {
            console.log(
              "Agent deletion succeeded despite error, cleaning up UI"
            );
            SessionUtils.clearAgentCache(agentId);
            try {
              const agentList = await client.agents.list();
              setAgents(
                (agentList.data as Array<{
                  agent_id: string;
                  agent_config?: {
                    agent_name?: string;
                    name?: string;
                    instructions?: string;
                  };
                  [key: string]: unknown;
                }>) || []
              );

              if (selectedAgentId === agentId) {
                const remainingAgents = agentList.data?.filter(
                  (a: { [key: string]: unknown }) =>
                    (a as { agent_id: string }).agent_id !== agentId
                );
                if (remainingAgents && remainingAgents.length > 0) {
                  const newAgent = remainingAgents[0] as {
                    agent_id: string;
                    agent_config?: {
                      agent_name?: string;
                      name?: string;
                      instructions?: string;
                    };
                    [key: string]: unknown;
                  };
                  setSelectedAgentId(newAgent.agent_id);
                  SessionUtils.saveCurrentAgentId(newAgent.agent_id);
                  await loadAgentConfig(newAgent.agent_id);
                  await loadAgentSessions(newAgent.agent_id);
                } else {
                  // no agents left
                  setSelectedAgentId("");
                  setCurrentSession(null);
                  setSelectedAgentConfig(null);
                }
              }
            } catch (refreshError) {
              console.error("Error refreshing agents list:", refreshError);
            }
          } else {
            // show error that we don't know about to user
            console.error("Unexpected error during agent deletion:", error);
            if (error instanceof Error) {
              alert(`Failed to delete agent: ${error.message}`);
            }
          }
        }
      }
    },
    [client, selectedAgentId, loadAgentConfig, loadAgentSessions]
  );

  const handleModelChange = useCallback((newModel: string) => {
    setSelectedModel(newModel);
    setCurrentSession(prev =>
      prev
        ? {
            ...prev,
            selectedModel: newModel,
            updatedAt: Date.now(),
          }
        : prev
    );
  }, []);

  useEffect(() => {
    if (currentSession) {
      SessionUtils.saveCurrentSessionId(
        currentSession.id,
        currentSession.agentId
      );
      // cache session data
      SessionUtils.saveSessionData(currentSession.agentId, currentSession);
      // only update selectedModel if the session has a valid model and it's different from current
      if (
        currentSession.selectedModel &&
        currentSession.selectedModel !== selectedModel
      ) {
        setSelectedModel(currentSession.selectedModel);
      }
    }
  }, [currentSession, selectedModel]);

  useEffect(() => {
    const fetchModels = async () => {
      try {
        setModelsLoading(true);
        setModelsError(null);
        const modelList = await client.models.list();

        // store all models (including embedding models for vector DB creation)
        setModels(modelList);

        // set default LLM model for chat
        const llmModels = modelList.filter(
          (model): model is ModelWithMetadata =>
            (model as ModelWithMetadata).custom_metadata?.model_type === "llm"
        );
        if (llmModels.length > 0) {
          handleModelChange(llmModels[0].id);
        }
      } catch (err) {
        console.error("Error fetching models:", err);
        setModelsError("Failed to fetch available models");
      } finally {
        setModelsLoading(false);
      }
    };

    fetchModels();
  }, [client, handleModelChange]);

  // load agent sessions after both agents and models are ready
  useEffect(() => {
    if (
      selectedAgentId &&
      !selectedAgentId.startsWith("chat-") &&
      selectedAgentId !== "__responses_mode__" &&
      !agentsLoading &&
      !modelsLoading &&
      selectedModel &&
      !currentSession
    ) {
      loadAgentSessions(selectedAgentId);
    }
  }, [
    selectedAgentId,
    agentsLoading,
    modelsLoading,
    selectedModel,
    currentSession,
    loadAgentSessions,
  ]);

  const handleInputChange = (e: React.ChangeEvent<HTMLTextAreaElement>) => {
    setInput(e.target.value);
  };

  const handleSubmit = async (event?: { preventDefault?: () => void }) => {
    event?.preventDefault?.();
    if (!input.trim()) return;

    const userMessage: Message = {
      id: Date.now().toString(),
      role: "user",
      content: input.trim(),
      createdAt: new Date(),
    };

    setCurrentSession(prev => {
      if (!prev) return prev;
      const updatedSession = {
        ...prev,
        messages: [...prev.messages, userMessage],
        updatedAt: Date.now(),
      };
      // update cache with new message
      SessionUtils.saveSessionData(prev.agentId, updatedSession);
      return updatedSession;
    });
    setInput("");

    await handleSubmitWithContent(userMessage.content);
  };

  const handleSubmitViaResponses = async (content: string) => {
    setIsGenerating(true);
    setError(null);

    if (abortControllerRef.current) {
      abortControllerRef.current.abort();
    }
    const abortController = new AbortController();
    abortControllerRef.current = abortController;

    try {
      const assistantMessage: Message = {
        id: (Date.now() + 1).toString(),
        role: "assistant",
        content: "",
        createdAt: new Date(),
      };

      setCurrentSession(prev => {
        if (!prev) return prev;
        const updated = {
          ...prev,
          messages: [...prev.messages, assistantMessage],
          updatedAt: Date.now(),
        };
        SessionUtils.saveSessionData(prev.agentId, updated);
        return updated;
      });

      const body: Record<string, unknown> = {
        model: selectedModel,
        input: content,
        stream: true,
      };
      if (lastResponseIdRef.current) {
        body.previous_response_id = lastResponseIdRef.current;
      }
      if (
        newAgentInstructions &&
        newAgentInstructions !== "You are a helpful assistant."
      ) {
        body.instructions = newAgentInstructions;
      }
      // Map selected tools to Responses API format
      if (selectedToolgroups.length > 0) {
        const tools: Record<string, unknown>[] = [];
        for (const toolId of selectedToolgroups) {
          if (toolId === "web_search" || toolId.includes("websearch")) {
            tools.push({ type: "web_search" });
          } else if (
            toolId === "file_search" ||
            toolId.includes("file_search")
          ) {
            const toolConfig: Record<string, unknown> = { type: "file_search" };
            if (selectedVectorDBs.length > 0) {
              toolConfig.vector_store_ids = selectedVectorDBs;
            }
            tools.push(toolConfig);
          }
        }
        if (tools.length > 0) {
          body.tools = tools;
        }
      }

      const res = await fetch("/api/v1/responses", {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify(body),
        signal: abortController.signal,
      });

      if (!res.ok) {
        const errText = await res.text();
        throw new Error(`Failed to get response: ${res.status} ${errText}`);
      }

      const reader = res.body?.getReader();
      if (!reader) throw new Error("No response body");

      const decoder = new TextDecoder();
      let buffer = "";
      let accumulatedText = "";

      while (true) {
        const { done, value } = await reader.read();
        if (done) break;

        buffer += decoder.decode(value, { stream: true });
        const lines = buffer.split("\n");
        buffer = lines.pop() || "";

        for (const line of lines) {
          if (!line.startsWith("data: ")) continue;
          const data = line.slice(6).trim();
          if (data === "[DONE]") continue;

          try {
            const event = JSON.parse(data);

            // Capture the response ID for conversation continuity
            if (event.response?.id) {
              lastResponseIdRef.current = event.response.id;
            }

            // Extract text from output_text.delta events
            if (event.type === "response.output_text.delta" && event.delta) {
              accumulatedText += event.delta;
              const textSnapshot = accumulatedText;
              flushSync(() => {
                setCurrentSession(prev => {
                  if (!prev) return prev;
                  const msgs = [...prev.messages];
                  const lastMsg = msgs[msgs.length - 1];
                  if (lastMsg?.role === "assistant") {
                    msgs[msgs.length - 1] = {
                      ...lastMsg,
                      content: textSnapshot,
                    };
                  }
                  return { ...prev, messages: msgs, updatedAt: Date.now() };
                });
              });
            }

            // Handle completed response (non-streaming fallback)
            if (event.type === "response.completed" && event.response) {
              const resp = event.response;
              if (resp.id) lastResponseIdRef.current = resp.id;
              if (!accumulatedText && resp.output) {
                for (const item of resp.output) {
                  if (item.type === "message" && item.content) {
                    for (const block of item.content) {
                      if (block.type === "output_text" && block.text) {
                        accumulatedText += block.text;
                      }
                    }
                  }
                }
                if (accumulatedText) {
                  const finalText = accumulatedText;
                  setCurrentSession(prev => {
                    if (!prev) return prev;
                    const msgs = [...prev.messages];
                    const lastMsg = msgs[msgs.length - 1];
                    if (lastMsg?.role === "assistant") {
                      msgs[msgs.length - 1] = {
                        ...lastMsg,
                        content: finalText,
                      };
                    }
                    return { ...prev, messages: msgs, updatedAt: Date.now() };
                  });
                }
              }
            }
          } catch {
            // skip unparseable SSE lines
          }
        }
      }

      // Save session
      setCurrentSession(prev => {
        if (prev) SessionUtils.saveSessionData(prev.agentId, prev);
        return prev;
      });
    } catch (err) {
      if (err instanceof Error && err.name === "AbortError") return;
      console.error("Error in responses API:", err);
      setError(err instanceof Error ? err.message : "Unknown error");
    } finally {
      setIsGenerating(false);
      abortControllerRef.current = null;
    }
  };

  const handleSubmitWithContent = async (content: string) => {
    if (!currentSession) return;

    // Use Responses API if no agent is selected (agents API unavailable)
    // Always use Responses API (agents API is unavailable)
    await handleSubmitViaResponses(content);
    return;

    setIsGenerating(true);
    setError(null);

    if (abortControllerRef.current) {
      abortControllerRef.current.abort();
    }

    const abortController = new AbortController();
    abortControllerRef.current = abortController;

    try {
      const userMessage = {
        role: "user" as const,
        content,
      };

      const turnParams: TurnCreateParams = {
        messages: [userMessage],
        stream: true,
      };

      const response = await client.agents.turn.create(
        selectedAgentId,
        currentSession.id,
        turnParams,
        {
          signal: abortController.signal,
          timeout: 300000, // 5 minutes timeout for RAG queries
        } as { signal: AbortSignal; timeout: number }
      );

      const assistantMessage: Message = {
        id: (Date.now() + 1).toString(),
        role: "assistant",
        content: "",
        createdAt: new Date(),
      };

      const processChunk = (
        chunk: unknown
      ): { text: string | null; isToolCall: boolean } => {
        const chunkObj = chunk as Record<string, unknown>;

        // helper to check if content contains function call JSON
        const containsToolCall = (content: string): boolean => {
          return (
            content.includes('"type": "function"') ||
            content.includes('"name": "file_search"') ||
            content.includes('"parameters":') ||
            !!content.match(/\{"type":\s*"function".*?\}/)
          );
        };

        let isToolCall = false;
        let potentialContent = "";

        if (typeof chunk === "string") {
          potentialContent = chunk;
          isToolCall = containsToolCall(chunk);
        }

        if (
          chunkObj?.delta &&
          typeof chunkObj.delta === "object" &&
          chunkObj.delta !== null
        ) {
          const delta = chunkObj.delta as Record<string, unknown>;
          if ("tool_calls" in delta) {
            isToolCall = true;
          }
          if (typeof delta.text === "string") {
            potentialContent = delta.text;
            if (containsToolCall(delta.text)) {
              isToolCall = true;
            }
          }
        }

        if (
          chunkObj?.event &&
          typeof chunkObj.event === "object" &&
          chunkObj.event !== null
        ) {
          const event = chunkObj.event as Record<string, unknown>;

          if (
            event?.payload &&
            typeof event.payload === "object" &&
            event.payload !== null
          ) {
            const payload = event.payload as Record<string, unknown>;
            if (typeof payload.content === "string") {
              potentialContent = payload.content;
              if (containsToolCall(payload.content)) {
                isToolCall = true;
              }
            }

            if (
              payload?.delta &&
              typeof payload.delta === "object" &&
              payload.delta !== null
            ) {
              const delta = payload.delta as Record<string, unknown>;
              if (typeof delta.text === "string") {
                potentialContent = delta.text;
                if (containsToolCall(delta.text)) {
                  isToolCall = true;
                }
              }
            }
          }

          if (
            event?.delta &&
            typeof event.delta === "object" &&
            event.delta !== null
          ) {
            const delta = event.delta as Record<string, unknown>;
            if (typeof delta.text === "string") {
              potentialContent = delta.text;
              if (containsToolCall(delta.text)) {
                isToolCall = true;
              }
            }
            if (typeof delta.content === "string") {
              // eslint-disable-next-line @typescript-eslint/no-unused-vars
              potentialContent = delta.content;
              if (containsToolCall(delta.content)) {
                isToolCall = true;
              }
            }
          }
        }

        // if it's a tool call, skip it (don't display in chat)
        if (isToolCall) {
          return { text: null, isToolCall: true };
        }

        let text: string | null = null;

        if (
          chunkObj?.delta &&
          typeof chunkObj.delta === "object" &&
          chunkObj.delta !== null
        ) {
          const delta = chunkObj.delta as Record<string, unknown>;
          if (typeof delta.text === "string") {
            text = extractCleanText(delta.text);
          }
        }

        if (
          !text &&
          chunkObj?.event &&
          typeof chunkObj.event === "object" &&
          chunkObj.event !== null
        ) {
          const event = chunkObj.event as Record<string, unknown>;

          if (
            event?.payload &&
            typeof event.payload === "object" &&
            event.payload !== null
          ) {
            const payload = event.payload as Record<string, unknown>;

            if (typeof payload.content === "string") {
              text = extractCleanText(payload.content);
            }

            if (
              !text &&
              payload?.turn &&
              typeof payload.turn === "object" &&
              payload.turn !== null
            ) {
              const turn = payload.turn as Record<string, unknown>;
              if (
                turn?.output_message &&
                typeof turn.output_message === "object" &&
                turn.output_message !== null
              ) {
                const outputMessage = turn.output_message as Record<
                  string,
                  unknown
                >;
                if (typeof outputMessage.content === "string") {
                  text = extractCleanText(outputMessage.content);
                }
              }

              if (
                !text &&
                turn?.steps &&
                Array.isArray(turn.steps) &&
                turn.steps.length > 0
              ) {
                for (const step of turn.steps) {
                  if (step && typeof step === "object" && step !== null) {
                    const stepObj = step as Record<string, unknown>;
                    if (
                      stepObj?.model_response &&
                      typeof stepObj.model_response === "object" &&
                      stepObj.model_response !== null
                    ) {
                      const modelResponse = stepObj.model_response as Record<
                        string,
                        unknown
                      >;
                      if (typeof modelResponse.content === "string") {
                        text = extractCleanText(modelResponse.content);
                        break;
                      }
                    }
                  }
                }
              }
            }

            if (
              !text &&
              payload?.delta &&
              typeof payload.delta === "object" &&
              payload.delta !== null
            ) {
              const delta = payload.delta as Record<string, unknown>;
              if (typeof delta.text === "string") {
                text = extractCleanText(delta.text);
              }
            }
          }

          if (
            !text &&
            event?.delta &&
            typeof event.delta === "object" &&
            event.delta !== null
          ) {
            const delta = event.delta as Record<string, unknown>;
            if (typeof delta.text === "string") {
              text = extractCleanText(delta.text);
            }
            if (!text && typeof delta.content === "string") {
              text = extractCleanText(delta.content);
            }
          }
        }

        if (
          !text &&
          chunkObj?.choices &&
          Array.isArray(chunkObj.choices) &&
          chunkObj.choices.length > 0
        ) {
          const choice = chunkObj.choices[0] as Record<string, unknown>;
          if (
            choice?.delta &&
            typeof choice.delta === "object" &&
            choice.delta !== null
          ) {
            const delta = choice.delta as Record<string, unknown>;
            if (typeof delta.content === "string") {
              text = extractCleanText(delta.content);
            }
          }
        }

        if (!text && typeof chunk === "string") {
          text = extractCleanText(chunk);
        }

        return { text, isToolCall: false };
      };
      setCurrentSession(prev => {
        if (!prev) return null;
        const updatedSession = {
          ...prev,
          messages: [...prev.messages, assistantMessage],
          updatedAt: Date.now(),
        };
        // update cache with assistant message
        SessionUtils.saveSessionData(prev.agentId, updatedSession);
        return updatedSession;
      });

      let fullContent = "";

      for await (const chunk of response) {
        const { text: deltaText } = processChunk(chunk);

        // logging for debugging function calls
        // if (deltaText && deltaText.includes("file_search")) {
        //   console.log("🔍 Function call detected in text output:", deltaText);
        //   console.log("🔍 Original chunk:", JSON.stringify(chunk, null, 2));
        // }

        if (chunk && typeof chunk === "object" && "event" in chunk) {
          const event = (
            chunk as {
              event: {
                payload?: {
                  event_type?: string;
                  turn?: { output_message?: { content?: string } };
                };
              };
            }
          ).event;
          if (event?.payload?.event_type === "turn_complete") {
            const content = event?.payload?.turn?.output_message?.content;
            if (content && content.includes("file_search")) {
              console.log("🔍 Function call found in turn_complete:", content);
            }
          }
        }

        if (deltaText) {
          fullContent += deltaText;

          flushSync(() => {
            setCurrentSession(prev => {
              if (!prev) return null;
              const newMessages = [...prev.messages];
              const last = newMessages[newMessages.length - 1];
              if (last.role === "assistant") {
                last.content = fullContent;
              }
              const updatedSession = {
                ...prev,
                messages: newMessages,
                updatedAt: Date.now(),
              };
              // update cache with streaming content
              if (fullContent.length % 100 === 0) {
                // Only cache every 100 characters
                SessionUtils.saveSessionData(prev.agentId, updatedSession);
              }
              return updatedSession;
            });
          });
        }
      }
    } catch (err) {
      if (err instanceof Error && err.name === "AbortError") {
        console.log("Request aborted");
        return;
      }

      console.error("Error sending message:", err);
      setError("Failed to send message. Please try again.");
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
      // cache final session state after streaming completes
      setCurrentSession(prev => {
        if (prev) {
          SessionUtils.saveSessionData(prev.agentId, prev);
        }
        return prev;
      });
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
      role: message.role,
      content: message.content,
      createdAt: new Date(),
    };
    setCurrentSession(prev =>
      prev
        ? {
            ...prev,
            messages: [...prev.messages, newMessage],
            updatedAt: Date.now(),
          }
        : prev
    );
    handleSubmitWithContent(newMessage.content);
  };

  const clearChat = () => {
    if (abortControllerRef.current) {
      abortControllerRef.current.abort();
      abortControllerRef.current = null;
      setIsGenerating(false);
    }

    setCurrentSession(prev =>
      prev ? { ...prev, messages: [], updatedAt: Date.now() } : prev
    );
    setError(null);
  };

  const handleRAGFileUpload = async (file: File) => {
    if (!selectedAgentConfig?.toolgroups || !selectedAgentId) {
      setError("No agent selected or agent has no RAG tools configured");
      return;
    }

    // find RAG toolgroups that have vector_db_ids configured
    const ragToolgroups = selectedAgentConfig.toolgroups.filter(toolgroup => {
      if (typeof toolgroup === "object" && toolgroup.name?.includes("rag")) {
        return toolgroup.args && "vector_db_ids" in toolgroup.args;
      }
      return false;
    });

    if (ragToolgroups.length === 0) {
      setError("Current agent has no vector databases configured for RAG");
      return;
    }

    try {
      setError(null);
      console.log("Uploading file using RAG tool...");

      setUploadNotification({
        show: true,
        message: `📄 Uploading and indexing "${file.name}"...`,
        type: "loading",
      });

      const vectorDbIds = ragToolgroups.flatMap(toolgroup => {
        if (
          typeof toolgroup === "object" &&
          toolgroup.args &&
          "vector_db_ids" in toolgroup.args
        ) {
          return toolgroup.args.vector_db_ids as string[];
        }
        return [];
      });

      // determine mime type from file extension - this should be in the OGX Client IMO
      const getContentType = (filename: string): string => {
        const ext = filename.toLowerCase().split(".").pop();
        switch (ext) {
          case "pdf":
            return "application/pdf";
          case "txt":
            return "text/plain";
          case "md":
            return "text/markdown";
          case "html":
            return "text/html";
          case "csv":
            return "text/csv";
          case "json":
            return "application/json";
          case "docx":
            return "application/vnd.openxmlformats-officedocument.wordprocessingml.document";
          case "doc":
            return "application/msword";
          default:
            return "application/octet-stream";
        }
      };

      const mimeType = getContentType(file.name);
      let fileContent: string;

      // handle text files vs binary files differently
      const isTextFile =
        mimeType.startsWith("text/") ||
        mimeType === "application/json" ||
        mimeType === "text/markdown" ||
        mimeType === "text/html" ||
        mimeType === "text/csv";

      if (isTextFile) {
        fileContent = await file.text();
      } else {
        // for PDFs and other binary files, create a data URL
        // use FileReader for efficient base64 conversion
        fileContent = await new Promise<string>((resolve, reject) => {
          const reader = new FileReader();
          reader.onload = () => resolve(reader.result as string);
          reader.onerror = () => reject(reader.error);
          reader.readAsDataURL(file);
        });
      }

      for (const vectorDbId of vectorDbIds) {
        await client.toolRuntime.ragTool.insert({
          documents: [
            {
              content: fileContent,
              document_id: `${file.name}-${Date.now()}`,
              metadata: {
                filename: file.name,
                file_size: file.size,
                uploaded_at: new Date().toISOString(),
                agent_id: selectedAgentId,
              },
              mime_type: mimeType,
            },
          ],
          vector_db_id: vectorDbId,
          // TODO: parameterize this somewhere, probably in settings
          chunk_size_in_tokens: 512,
        });
      }

      console.log("✅ File successfully uploaded using RAG tool");

      setUploadNotification({
        show: true,
        message: `📄 File "${file.name}" uploaded and indexed successfully!`,
        type: "success",
      });

      setTimeout(() => {
        setUploadNotification(prev => ({ ...prev, show: false }));
      }, 4000);
    } catch (err) {
      console.error("Error uploading file using RAG tool:", err);
      const errorMessage =
        err instanceof Error
          ? `Failed to upload file: ${err.message}`
          : "Failed to upload file using RAG tool";

      setUploadNotification({
        show: true,
        message: errorMessage,
        type: "error",
      });

      setTimeout(() => {
        setUploadNotification(prev => ({ ...prev, show: false }));
      }, 6000);
    }
  };

  return (
    <div className="flex flex-col h-full w-full max-w-7xl mx-auto">
      {/* Upload Notification */}
      {uploadNotification.show && (
        <div
          className={`fixed top-4 right-4 z-50 p-4 rounded-lg shadow-lg transition-all duration-300 ${
            uploadNotification.type === "success"
              ? "bg-green-100 border border-green-300 text-green-800"
              : uploadNotification.type === "error"
                ? "bg-red-100 border border-red-300 text-red-800"
                : "bg-blue-100 border border-blue-300 text-blue-800"
          }`}
        >
          <div className="flex items-center gap-2">
            {uploadNotification.type === "loading" && (
              <div className="animate-spin rounded-full h-4 w-4 border-2 border-blue-600 border-t-transparent"></div>
            )}
            <span className="text-sm font-medium">
              {uploadNotification.message}
            </span>
            {uploadNotification.type !== "loading" && (
              <button
                onClick={() =>
                  setUploadNotification(prev => ({ ...prev, show: false }))
                }
                className="ml-2 text-gray-400 hover:text-gray-600"
              >
                ✕
              </button>
            )}
          </div>
        </div>
      )}

      {/* Header */}
      <div className="mb-6">
        <div className="flex justify-between items-center mb-4">
          <h1 className="text-3xl font-bold">Agent Session</h1>
          <div className="flex items-center gap-3">
            {!agentsLoading && agents.length > 0 && (
              <div className="flex items-center gap-2">
                <label className="text-sm font-medium">Agent Session:</label>
                <Select
                  value={selectedAgentId}
                  onValueChange={agentId => {
                    setSelectedAgentId(agentId);
                    SessionUtils.saveCurrentAgentId(agentId);
                    // Load local session for responses-mode agents
                    const savedSession = SessionUtils.loadSessionData(agentId);
                    if (savedSession) {
                      setCurrentSession(savedSession);
                    } else {
                      const localSession: ChatSession = {
                        id: `session-${Date.now()}`,
                        agentId,
                        name:
                          agents.find(a => a.agent_id === agentId)?.agent_config
                            ?.name || "Chat",
                        messages: [],
                        createdAt: Date.now(),
                        updatedAt: Date.now(),
                      };
                      setCurrentSession(localSession);
                    }
                    lastResponseIdRef.current = null;
                    // Load agent instructions
                    const agent = agents.find(a => a.agent_id === agentId);
                    if (agent?.agent_config?.instructions) {
                      setNewAgentInstructions(agent.agent_config.instructions);
                    }
                  }}
                  disabled={agentsLoading}
                >
                  <SelectTrigger className="w-[200px]">
                    <SelectValue
                      placeholder={
                        agentsLoading ? "Loading..." : "Select Agent Session"
                      }
                    />
                  </SelectTrigger>
                  <SelectContent>
                    {agents.map(agent => (
                      <SelectItem key={agent.agent_id} value={agent.agent_id}>
                        {(() => {
                          if (
                            agent.agent_config &&
                            "name" in agent.agent_config &&
                            typeof agent.agent_config.name === "string"
                          ) {
                            return agent.agent_config.name;
                          }
                          if (
                            agent.agent_config &&
                            "agent_name" in agent.agent_config &&
                            typeof agent.agent_config.agent_name === "string"
                          ) {
                            return agent.agent_config.agent_name;
                          }
                          return `Agent ${agent.agent_id.slice(0, 8)}...`;
                        })()}
                      </SelectItem>
                    ))}
                  </SelectContent>
                </Select>
                {selectedAgentId && (
                  <Button
                    onClick={() => deleteAgent(selectedAgentId)}
                    variant="outline"
                    size="sm"
                    className="text-destructive hover:text-destructive hover:bg-destructive/10"
                    title="Delete current agent"
                  >
                    <Trash2 className="h-3 w-3" />
                  </Button>
                )}
              </div>
            )}
            <Button
              onClick={() => setShowCreateAgent(true)}
              variant="outline"
              size="sm"
            >
              + New Agent
            </Button>
            {!agentsLoading && agents.length > 0 && (
              <Button
                variant="outline"
                onClick={clearChat}
                disabled={isGenerating}
              >
                Clear Chat
              </Button>
            )}
          </div>
        </div>
      </div>
      {/* Main Two-Column Layout */}
      <div className="flex flex-1 gap-6 min-h-0 flex-col lg:flex-row">
        {/* Left Column - Configuration Panel */}
        <div className="w-full lg:w-80 lg:flex-shrink-0 space-y-6 p-4 border border-border rounded-lg bg-muted/30">
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
                    {models
                      .filter(
                        (model): model is ModelWithMetadata =>
                          (model as ModelWithMetadata).custom_metadata
                            ?.model_type === "llm"
                      )
                      .map(model => (
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
                  Agent Instructions
                </label>
                <div className="w-full h-24 px-3 py-2 text-sm border border-input rounded-md bg-muted text-muted-foreground">
                  {(selectedAgentId &&
                    agents.find(a => a.agent_id === selectedAgentId)
                      ?.agent_config?.instructions) ||
                    "No agent selected"}
                </div>
                <p className="text-xs text-muted-foreground mt-1">
                  Instructions are set when creating an agent and cannot be
                  changed.
                </p>
              </div>
            </div>
          </div>

          {/* Agent Tools */}
          <div className="space-y-4 text-left">
            <h3 className="text-lg font-semibold border-b pb-2 text-left">
              Agent Tools
            </h3>
            <div className="space-y-3">
              <div>
                <label className="text-sm font-medium block mb-2 text-muted-foreground">
                  Configured Tools
                </label>
                <div className="space-y-2">
                  {selectedAgentConfig?.toolgroups &&
                  selectedAgentConfig.toolgroups.length > 0 ? (
                    selectedAgentConfig.toolgroups.map(
                      (
                        toolgroup:
                          | string
                          | { name: string; args: Record<string, unknown> },
                        index: number
                      ) => {
                        const toolName =
                          typeof toolgroup === "string"
                            ? toolgroup
                            : toolgroup.name;
                        const toolArgs =
                          typeof toolgroup === "object" ? toolgroup.args : null;

                        const isRAGTool = toolName.includes("rag");
                        const displayName = isRAGTool ? "RAG Search" : toolName;
                        const displayIcon = isRAGTool
                          ? "🔍"
                          : toolName.includes("search")
                            ? "🌐"
                            : "🔧";

                        return (
                          <div
                            key={index}
                            className="p-3 border border-input rounded-md bg-muted text-muted-foreground"
                          >
                            <div className="flex items-center justify-between">
                              <div className="flex items-center gap-2">
                                <span className="text-sm">{displayIcon}</span>
                                <span className="text-sm font-medium text-primary">
                                  {displayName}
                                </span>
                              </div>
                            </div>
                            {isRAGTool && toolArgs && toolArgs.vector_db_ids ? (
                              <div className="mt-2 text-xs text-muted-foreground">
                                <span className="font-medium">
                                  Vector Databases:
                                </span>
                                <div className="mt-1 flex flex-wrap gap-1">
                                  {Array.isArray(toolArgs.vector_db_ids) ? (
                                    toolArgs.vector_db_ids.map(
                                      (dbId: string, idx: number) => (
                                        <code
                                          key={idx}
                                          className="px-1.5 py-0.5 bg-muted-foreground/10 rounded text-xs"
                                        >
                                          {dbId}
                                        </code>
                                      )
                                    )
                                  ) : (
                                    <code className="px-1.5 py-0.5 bg-muted-foreground/10 rounded text-xs">
                                      {String(toolArgs.vector_db_ids)}
                                    </code>
                                  )}
                                </div>
                              </div>
                            ) : null}
                            {!isRAGTool &&
                              toolArgs &&
                              Object.keys(toolArgs).length > 0 && (
                                <div className="mt-2 text-xs text-muted-foreground">
                                  <span className="font-medium">
                                    Configuration:
                                  </span>{" "}
                                  {Object.keys(toolArgs).length} parameter
                                  {Object.keys(toolArgs).length > 1 ? "s" : ""}
                                </div>
                              )}
                          </div>
                        );
                      }
                    )
                  ) : (
                    <div className="p-3 border border-input rounded-md bg-muted text-center">
                      <p className="text-sm text-muted-foreground">
                        No tools configured
                      </p>
                      <p className="text-xs text-muted-foreground mt-1">
                        This agent only has text generation capabilities
                      </p>
                    </div>
                  )}
                </div>
                <p className="text-xs text-muted-foreground mt-2">
                  Tools are configured when creating an agent and provide
                  additional capabilities like web search, math calculations, or
                  RAG document retrieval.
                </p>
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

          {!agentsLoading && agents.length === 0 ? (
            <div className="flex-1 flex items-center justify-center">
              <div className="text-center space-y-4 max-w-md">
                <div className="text-6xl mb-4">🦙</div>
                <h2 className="text-2xl font-semibold text-muted-foreground">
                  Create an Agent with OGX
                </h2>
                <p className="text-muted-foreground">
                  To get started, create your first agent. Each agent is
                  configured with specific instructions, models, and tools to
                  help you with different tasks.
                </p>
                <Button
                  onClick={() => setShowCreateAgent(true)}
                  size="lg"
                  className="mt-4"
                >
                  Create Your First Agent
                </Button>
              </div>
            </div>
          ) : (
            <Chat
              className="flex-1"
              messages={currentSession?.messages || []}
              handleSubmit={handleSubmit}
              input={input}
              handleInputChange={handleInputChange}
              isGenerating={isGenerating}
              append={append}
              suggestions={suggestions}
              setMessages={messages =>
                setCurrentSession(prev =>
                  prev ? { ...prev, messages, updatedAt: Date.now() } : prev
                )
              }
              onRAGFileUpload={handleRAGFileUpload}
            />
          )}
        </div>
      </div>

      {/* Create Agent Modal */}
      {showCreateAgent && (
        <div className="fixed inset-0 bg-black/50 flex items-center justify-center z-50">
          <Card className="w-[500px] p-6 space-y-4">
            <h3 className="text-lg font-semibold">Create New Agent</h3>

            <div className="space-y-4">
              <div>
                <label className="text-sm font-medium block mb-2">
                  Agent Name (optional)
                </label>
                <Input
                  value={newAgentName}
                  onChange={e => setNewAgentName(e.target.value)}
                  placeholder="My Custom Agent"
                />
              </div>

              <div>
                <label className="text-sm font-medium block mb-2">Model</label>
                <Select value={selectedModel} onValueChange={setSelectedModel}>
                  <SelectTrigger>
                    <SelectValue placeholder="Select Model" />
                  </SelectTrigger>
                  <SelectContent>
                    {models
                      .filter(
                        (model): model is ModelWithMetadata =>
                          (model as ModelWithMetadata).custom_metadata
                            ?.model_type === "llm"
                      )
                      .map(model => (
                        <SelectItem key={model.id} value={model.id}>
                          {model.id}
                        </SelectItem>
                      ))}
                  </SelectContent>
                </Select>
              </div>

              <div>
                <label className="text-sm font-medium block mb-2">
                  System Instructions
                </label>
                <textarea
                  value={newAgentInstructions}
                  onChange={e => setNewAgentInstructions(e.target.value)}
                  placeholder="You are a helpful assistant."
                  className="w-full h-32 px-3 py-2 text-sm border border-input rounded-md resize-none focus:outline-none focus:ring-2 focus:ring-ring focus:ring-offset-2"
                />
              </div>

              <div>
                <label className="text-sm font-medium block mb-2">
                  Tools (optional)
                </label>
                <p className="text-xs text-muted-foreground mb-2">
                  Available toolgroups: {availableToolgroups.length} found
                </p>
                <div className="space-y-2">
                  {availableToolgroups.length === 0 ? (
                    <p className="text-sm text-muted-foreground">
                      No tools available
                    </p>
                  ) : (
                    availableToolgroups.map(toolgroup => (
                      <label
                        key={toolgroup.identifier}
                        className="flex items-center space-x-2"
                      >
                        <input
                          type="checkbox"
                          checked={selectedToolgroups.includes(
                            toolgroup.identifier
                          )}
                          onChange={e => {
                            if (e.target.checked) {
                              setSelectedToolgroups(prev => {
                                const newSelection = [
                                  ...prev,
                                  toolgroup.identifier,
                                ];
                                return newSelection;
                              });
                            } else {
                              setSelectedToolgroups(prev => {
                                const newSelection = prev.filter(
                                  id => id !== toolgroup.identifier
                                );
                                return newSelection;
                              });
                            }
                          }}
                          className="rounded border-input"
                        />
                        <span className="text-sm">
                          <code className="bg-muted px-1 rounded text-xs">
                            {toolgroup.identifier}
                          </code>
                          <span className="text-muted-foreground ml-2">
                            ({toolgroup.provider_id})
                          </span>
                        </span>
                      </label>
                    ))
                  )}
                </div>
                {selectedToolgroups.length === 0 && (
                  <p className="text-xs text-muted-foreground mt-1">
                    No tools selected - agent will only have text generation
                    capabilities.
                  </p>
                )}
                <p className="text-xs text-muted-foreground mt-2 p-2 bg-muted/50 border border-border rounded">
                  <strong>Note:</strong> Selected tools will be configured for
                  the agent. Some tools like RAG may require additional vector
                  DB configuration, and web search tools need API keys. Basic
                  text generation agents work without tools.
                </p>
              </div>

              {/* Vector DB Configuration for RAG */}
              {selectedToolgroups.includes("builtin::file_search") && (
                <div>
                  <label className="text-sm font-medium block mb-2">
                    Vector Databases for RAG
                  </label>
                  <div className="flex items-center gap-2 mb-2">
                    <Button
                      type="button"
                      variant="outline"
                      size="sm"
                      onClick={() => setShowCreateVectorDB(true)}
                    >
                      + Create Vector DB
                    </Button>
                    <span className="text-xs text-muted-foreground">
                      {availableVectorDBs.length} available
                    </span>
                  </div>
                  <div className="space-y-2 max-h-32 overflow-y-auto">
                    {availableVectorDBs.length === 0 ? (
                      <p className="text-sm text-muted-foreground">
                        No vector databases available. Create one to use RAG
                        tools.
                      </p>
                    ) : (
                      availableVectorDBs.map(vectorDB => (
                        <label
                          key={vectorDB.identifier}
                          className="flex items-center space-x-2"
                        >
                          <input
                            type="checkbox"
                            checked={selectedVectorDBs.includes(
                              vectorDB.identifier
                            )}
                            onChange={e => {
                              if (e.target.checked) {
                                setSelectedVectorDBs(prev => [
                                  ...prev,
                                  vectorDB.identifier,
                                ]);
                              } else {
                                setSelectedVectorDBs(prev =>
                                  prev.filter(id => id !== vectorDB.identifier)
                                );
                              }
                            }}
                            className="rounded border-input"
                          />
                          <span className="text-sm">
                            <code className="bg-muted px-1 rounded text-xs">
                              {vectorDB.identifier}
                            </code>
                            {vectorDB.vector_db_name && (
                              <span className="text-muted-foreground ml-2">
                                ({vectorDB.vector_db_name})
                              </span>
                            )}
                          </span>
                        </label>
                      ))
                    )}
                  </div>
                  {selectedVectorDBs.length === 0 &&
                    selectedToolgroups.includes("builtin::file_search") && (
                      <p className="text-xs text-muted-foreground mt-1">
                        ⚠️ RAG tool selected but no vector databases chosen.
                        Create or select a vector database.
                      </p>
                    )}
                </div>
              )}
            </div>

            <div className="flex gap-2 pt-4">
              <Button
                onClick={async () => {
                  try {
                    await createNewAgent(
                      newAgentName,
                      newAgentInstructions,
                      selectedModel,
                      selectedToolgroups,
                      selectedVectorDBs
                    );
                    setShowCreateAgent(false);
                    setNewAgentName("");
                    setNewAgentInstructions("You are a helpful assistant.");
                    setSelectedToolgroups([]);
                    setSelectedVectorDBs([]);
                  } catch (error) {
                    console.error("Failed to create agent:", error);
                  }
                }}
                className="flex-1"
                disabled={!selectedModel || !newAgentInstructions.trim()}
              >
                Create Agent
              </Button>
              <Button
                variant="outline"
                onClick={() => {
                  setShowCreateAgent(false);
                  setNewAgentName("");
                  setNewAgentInstructions("You are a helpful assistant.");
                  setSelectedToolgroups([]);
                  setSelectedVectorDBs([]);
                }}
                className="flex-1"
              >
                Cancel
              </Button>
            </div>
          </Card>
        </div>
      )}

      {/* Create Vector DB Modal */}
      {showCreateVectorDB && (
        <div className="fixed inset-0 bg-black/50 flex items-center justify-center z-50">
          <VectorDBCreator
            models={models}
            onVectorDBCreated={handleVectorDBCreated}
            onCancel={() => setShowCreateVectorDB(false)}
          />
        </div>
      )}
    </div>
  );
}
