"use client";

import { useState, useEffect, useCallback } from "react";
import { Button } from "@/components/ui/button";
import {
  Select,
  SelectContent,
  SelectItem,
  SelectTrigger,
  SelectValue,
} from "@/components/ui/select";
import { Input } from "@/components/ui/input";
import { Card } from "@/components/ui/card";
import { Trash2 } from "lucide-react";
import type { Message } from "@/components/chat-playground/chat-message";
import { useAuthClient } from "@/hooks/use-auth-client";
import { cleanMessageContent } from "@/lib/message-content-utils";
import type {
  Session,
  SessionCreateParams,
} from "llama-stack-client/resources/agents";

export interface ChatSession {
  id: string;
  name: string;
  messages: Message[];
  selectedModel: string;
  systemMessage: string;
  agentId: string;
  session?: Session;
  createdAt: number;
  updatedAt: number;
}

interface SessionManagerProps {
  currentSession: ChatSession | null;
  onSessionChange: (session: ChatSession) => void;
  onNewSession: () => void;
  selectedAgentId: string;
}

const CURRENT_SESSION_KEY = "chat-playground-current-session";

// ensures this only happens client side
const safeLocalStorage = {
  getItem: (key: string): string | null => {
    if (typeof window === "undefined") return null;
    try {
      return localStorage.getItem(key);
    } catch (err) {
      console.error("Error accessing localStorage:", err);
      return null;
    }
  },
  setItem: (key: string, value: string): void => {
    if (typeof window === "undefined") return;
    try {
      localStorage.setItem(key, value);
    } catch (err) {
      console.error("Error writing to localStorage:", err);
    }
  },
  removeItem: (key: string): void => {
    if (typeof window === "undefined") return;
    try {
      localStorage.removeItem(key);
    } catch (err) {
      console.error("Error removing from localStorage:", err);
    }
  },
};

const generateSessionId = (): string => {
  return globalThis.crypto.randomUUID();
};

export function Conversations({
  currentSession,
  onSessionChange,
  selectedAgentId,
}: SessionManagerProps) {
  const [sessions, setSessions] = useState<ChatSession[]>([]);
  const [showCreateForm, setShowCreateForm] = useState(false);
  const [newSessionName, setNewSessionName] = useState("");
  const [loading, setLoading] = useState(false);
  const client = useAuthClient();

  const loadAgentSessions = useCallback(async () => {
    if (!selectedAgentId) return;

    setLoading(true);
    try {
      const response = await client.agents.session.list(selectedAgentId);
      console.log("Sessions response:", response);

      if (!response.data || !Array.isArray(response.data)) {
        console.warn("Invalid sessions response, starting fresh");
        setSessions([]);
        return;
      }

      const agentSessions: ChatSession[] = response.data
        .filter(sessionData => {
          const isValid =
            sessionData &&
            typeof sessionData === "object" &&
            sessionData.session_id &&
            sessionData.session_name;
          if (!isValid) {
            console.warn("Filtering out invalid session:", sessionData);
          }
          return isValid;
        })
        .map(sessionData => ({
          id: sessionData.session_id,
          name: sessionData.session_name,
          messages: [],
          selectedModel: currentSession?.selectedModel || "",
          systemMessage:
            currentSession?.systemMessage || "You are a helpful assistant.",
          agentId: selectedAgentId,
          session: sessionData,
          createdAt: sessionData.started_at
            ? new Date(sessionData.started_at).getTime()
            : Date.now(),
          updatedAt: sessionData.started_at
            ? new Date(sessionData.started_at).getTime()
            : Date.now(),
        }));
      setSessions(agentSessions);
    } catch (error) {
      console.error("Error loading agent sessions:", error);
      setSessions([]);
    } finally {
      setLoading(false);
    }
  }, [
    selectedAgentId,
    client,
    currentSession?.selectedModel,
    currentSession?.systemMessage,
  ]);

  useEffect(() => {
    if (selectedAgentId) {
      loadAgentSessions();
    }
  }, [selectedAgentId, loadAgentSessions]);

  const createNewSession = async () => {
    if (!selectedAgentId) return;

    const sessionName =
      newSessionName.trim() || `Session ${sessions.length + 1}`;
    setLoading(true);

    try {
      const response = await client.agents.session.create(selectedAgentId, {
        session_name: sessionName,
      } as SessionCreateParams);

      const newSession: ChatSession = {
        id: response.session_id,
        name: sessionName,
        messages: [],
        selectedModel: currentSession?.selectedModel || "",
        systemMessage:
          currentSession?.systemMessage || "You are a helpful assistant.",
        agentId: selectedAgentId,
        createdAt: Date.now(),
        updatedAt: Date.now(),
      };

      setSessions(prev => [...prev, newSession]);
      SessionUtils.saveCurrentSessionId(newSession.id, selectedAgentId);
      onSessionChange(newSession);

      setNewSessionName("");
      setShowCreateForm(false);
    } catch (error) {
      console.error("Error creating session:", error);
    } finally {
      setLoading(false);
    }
  };

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
          // Add user messages from input_messages
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

          // Add assistant message from output_message
          if (turn.output_message && turn.output_message.content) {
            messages.push({
              id: `${turn.turn_id}-assistant-${messages.length}`,
              role: "assistant",
              content: cleanMessageContent(turn.output_message.content),
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

  const switchToSession = useCallback(
    async (sessionId: string) => {
      const session = sessions.find(s => s.id === sessionId);
      if (session) {
        setLoading(true);
        try {
          // Load messages for this session
          const messages = await loadSessionMessages(
            selectedAgentId,
            sessionId
          );
          const sessionWithMessages = {
            ...session,
            messages,
          };

          SessionUtils.saveCurrentSessionId(sessionId, selectedAgentId);
          onSessionChange(sessionWithMessages);
        } catch (error) {
          console.error("Error switching to session:", error);
          // Fallback to session without messages
          SessionUtils.saveCurrentSessionId(sessionId, selectedAgentId);
          onSessionChange(session);
        } finally {
          setLoading(false);
        }
      }
    },
    [sessions, selectedAgentId, loadSessionMessages, onSessionChange]
  );

  const deleteSession = async (sessionId: string) => {
    if (!selectedAgentId) {
      return;
    }

    if (
      confirm(
        "Are you sure you want to delete this session? This action cannot be undone."
      )
    ) {
      setLoading(true);
      try {
        await client.agents.session.delete(selectedAgentId, sessionId);

        const updatedSessions = sessions.filter(s => s.id !== sessionId);
        setSessions(updatedSessions);

        if (currentSession?.id === sessionId) {
          const newCurrentSession = updatedSessions[0] || null;
          if (newCurrentSession) {
            SessionUtils.saveCurrentSessionId(
              newCurrentSession.id,
              selectedAgentId
            );
            onSessionChange(newCurrentSession);
          } else {
            SessionUtils.clearCurrentSession(selectedAgentId);
            onNewSession();
          }
        }
      } catch (error) {
        console.error("Error deleting session:", error);
      } finally {
        setLoading(false);
      }
    }
  };

  useEffect(() => {
    if (currentSession) {
      setSessions(prevSessions => {
        const updatedSessions = prevSessions.map(session =>
          session.id === currentSession.id ? currentSession : session
        );

        if (!prevSessions.find(s => s.id === currentSession.id)) {
          updatedSessions.push(currentSession);
        }

        return updatedSessions;
      });
    }
  }, [currentSession]);

  if (!selectedAgentId) {
    return null;
  }

  return (
    <div className="relative">
      <div className="flex items-center gap-2">
        <Select
          value={currentSession?.id || ""}
          onValueChange={switchToSession}
        >
          <SelectTrigger className="w-[200px]">
            <SelectValue placeholder="Select Session" />
          </SelectTrigger>
          <SelectContent>
            {sessions.map(session => (
              <SelectItem key={session.id} value={session.id}>
                {session.name}
              </SelectItem>
            ))}
          </SelectContent>
        </Select>

        <Button
          onClick={() => setShowCreateForm(true)}
          variant="outline"
          size="sm"
          disabled={loading || !selectedAgentId}
        >
          + New
        </Button>

        {currentSession && (
          <Button
            onClick={() => deleteSession(currentSession.id)}
            variant="outline"
            size="sm"
            className="text-destructive hover:text-destructive hover:bg-destructive/10"
            title="Delete current session"
          >
            <Trash2 className="h-3 w-3" />
          </Button>
        )}
      </div>

      {showCreateForm && (
        <Card className="absolute top-full left-0 mt-2 p-4 space-y-3 w-80 z-50 bg-background border shadow-lg">
          <h3 className="text-md font-semibold">Create New Session</h3>

          <Input
            value={newSessionName}
            onChange={e => setNewSessionName(e.target.value)}
            placeholder="Session name (optional)"
            onKeyDown={e => {
              if (e.key === "Enter") {
                createNewSession();
              } else if (e.key === "Escape") {
                setShowCreateForm(false);
                setNewSessionName("");
              }
            }}
          />

          <div className="flex gap-2">
            <Button
              onClick={createNewSession}
              className="flex-1"
              disabled={loading}
            >
              {loading ? "Creating..." : "Create"}
            </Button>
            <Button
              variant="outline"
              onClick={() => {
                setShowCreateForm(false);
                setNewSessionName("");
              }}
              className="flex-1"
            >
              Cancel
            </Button>
          </div>
        </Card>
      )}

      {currentSession && sessions.length > 1 && (
        <div className="absolute top-full left-0 mt-1 text-xs text-gray-500 whitespace-nowrap">
          {sessions.length} sessions • Current: {currentSession.name}
          {currentSession.messages.length > 0 &&
            ` • ${currentSession.messages.length} messages`}
        </div>
      )}
    </div>
  );
}

export const SessionUtils = {
  loadCurrentSessionId: (agentId?: string): string | null => {
    const key = agentId
      ? `${CURRENT_SESSION_KEY}-${agentId}`
      : CURRENT_SESSION_KEY;
    return safeLocalStorage.getItem(key);
  },

  saveCurrentSessionId: (sessionId: string, agentId?: string) => {
    const key = agentId
      ? `${CURRENT_SESSION_KEY}-${agentId}`
      : CURRENT_SESSION_KEY;
    safeLocalStorage.setItem(key, sessionId);
  },

  createDefaultSession: (
    agentId: string,
    inheritModel?: string
  ): ChatSession => ({
    id: generateSessionId(),
    name: "Default Session",
    messages: [],
    selectedModel: inheritModel || "",
    systemMessage: "You are a helpful assistant.",
    agentId,
    createdAt: Date.now(),
    updatedAt: Date.now(),
  }),

  clearCurrentSession: (agentId?: string) => {
    const key = agentId
      ? `${CURRENT_SESSION_KEY}-${agentId}`
      : CURRENT_SESSION_KEY;
    safeLocalStorage.removeItem(key);
  },

  loadCurrentAgentId: (): string | null => {
    return safeLocalStorage.getItem("chat-playground-current-agent");
  },

  saveCurrentAgentId: (agentId: string) => {
    safeLocalStorage.setItem("chat-playground-current-agent", agentId);
  },

  // Comprehensive session caching
  saveSessionData: (agentId: string, sessionData: ChatSession) => {
    const key = `chat-playground-session-data-${agentId}-${sessionData.id}`;
    safeLocalStorage.setItem(
      key,
      JSON.stringify({
        ...sessionData,
        cachedAt: Date.now(),
      })
    );
  },

  loadSessionData: (agentId: string, sessionId: string): ChatSession | null => {
    const key = `chat-playground-session-data-${agentId}-${sessionId}`;
    const cached = safeLocalStorage.getItem(key);
    if (!cached) return null;

    try {
      const data = JSON.parse(cached);
      // Check if cache is fresh (less than 1 hour old)
      const cacheAge = Date.now() - (data.cachedAt || 0);
      if (cacheAge > 60 * 60 * 1000) {
        safeLocalStorage.removeItem(key);
        return null;
      }

      // Convert date strings back to Date objects
      return {
        ...data,
        messages: data.messages.map(
          (msg: { createdAt: string; [key: string]: unknown }) => ({
            ...msg,
            createdAt: new Date(msg.createdAt),
          })
        ),
      };
    } catch (error) {
      console.error("Error parsing cached session data:", error);
      safeLocalStorage.removeItem(key);
      return null;
    }
  },

  // Agent config caching
  saveAgentConfig: (
    agentId: string,
    config: {
      toolgroups?: Array<
        string | { name: string; args: Record<string, unknown> }
      >;
      [key: string]: unknown;
    }
  ) => {
    const key = `chat-playground-agent-config-${agentId}`;
    safeLocalStorage.setItem(
      key,
      JSON.stringify({
        config,
        cachedAt: Date.now(),
      })
    );
  },

  loadAgentConfig: (
    agentId: string
  ): {
    toolgroups?: Array<
      string | { name: string; args: Record<string, unknown> }
    >;
    [key: string]: unknown;
  } | null => {
    const key = `chat-playground-agent-config-${agentId}`;
    const cached = safeLocalStorage.getItem(key);
    if (!cached) return null;

    try {
      const data = JSON.parse(cached);
      // Check if cache is fresh (less than 30 minutes old)
      const cacheAge = Date.now() - (data.cachedAt || 0);
      if (cacheAge > 30 * 60 * 1000) {
        safeLocalStorage.removeItem(key);
        return null;
      }
      return data.config;
    } catch (error) {
      console.error("Error parsing cached agent config:", error);
      safeLocalStorage.removeItem(key);
      return null;
    }
  },

  // Clear all cached data for an agent
  clearAgentCache: (agentId: string) => {
    const keys = Object.keys(localStorage).filter(
      key =>
        key.includes(`chat-playground-session-data-${agentId}`) ||
        key.includes(`chat-playground-agent-config-${agentId}`)
    );
    keys.forEach(key => safeLocalStorage.removeItem(key));
  },

  // Persist the local agents list (for responses-mode agents)
  saveAgentsList: (
    agents: Array<{
      agent_id: string;
      agent_config?: {
        name?: string;
        agent_name?: string;
        instructions?: string;
      };
    }>
  ) => {
    safeLocalStorage.setItem(
      "chat-playground-agents-list",
      JSON.stringify(agents)
    );
  },

  loadAgentsList: (): Array<{
    agent_id: string;
    agent_config?: {
      name?: string;
      agent_name?: string;
      instructions?: string;
    };
  }> => {
    const cached = safeLocalStorage.getItem("chat-playground-agents-list");
    if (!cached) return [];
    try {
      return JSON.parse(cached);
    } catch {
      return [];
    }
  },

  // Save which session ID is active for a given agent
  saveActiveSessionId: (agentId: string, sessionId: string) => {
    safeLocalStorage.setItem(
      `chat-playground-active-session-${agentId}`,
      sessionId
    );
  },

  loadActiveSessionId: (agentId: string): string | null => {
    return safeLocalStorage.getItem(
      `chat-playground-active-session-${agentId}`
    );
  },
};
