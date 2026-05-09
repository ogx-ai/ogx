import React from "react";
import { render, screen, waitFor, act } from "@testing-library/react";
import "@testing-library/jest-dom";
import { Conversations, SessionUtils } from "./conversations";
import type { Message } from "@/components/chat-playground/chat-message";

interface ChatSession {
  id: string;
  name: string;
  messages: Message[];
  selectedModel: string;
  systemMessage: string;
  agentId: string;
  createdAt: number;
  updatedAt: number;
}

const mockOnSessionChange = jest.fn();
const mockOnNewSession = jest.fn();

// Mock the auth client
const mockClient = {
  agents: {
    session: {
      list: jest.fn(),
      create: jest.fn(),
      delete: jest.fn(),
      retrieve: jest.fn(),
    },
  },
};

// Mock the useAuthClient hook
jest.mock("@/hooks/use-auth-client", () => ({
  useAuthClient: jest.fn(() => mockClient),
}));

// Mock additional SessionUtils methods that are now being used
jest.mock("./conversations", () => {
  const actual = jest.requireActual("./conversations");
  return {
    ...actual,
    SessionUtils: {
      ...actual.SessionUtils,
      saveSessionData: jest.fn(),
      loadSessionData: jest.fn(),
      saveAgentConfig: jest.fn(),
      loadAgentConfig: jest.fn(),
      clearAgentCache: jest.fn(),
    },
  };
});

const localStorageMock = {
  getItem: jest.fn(),
  setItem: jest.fn(),
  removeItem: jest.fn(),
  clear: jest.fn(),
};

Object.defineProperty(window, "localStorage", {
  value: localStorageMock,
  writable: true,
});

// Mock crypto.randomUUID for test environment
let uuidCounter = 0;
Object.defineProperty(globalThis, "crypto", {
  value: {
    randomUUID: jest.fn(() => `test-uuid-${++uuidCounter}`),
  },
  writable: true,
});

describe("SessionManager", () => {
  const mockSession: ChatSession = {
    id: "session_123",
    name: "Test Session",
    messages: [
      {
        id: "msg_1",
        role: "user",
        content: "Hello",
        createdAt: new Date(),
      },
    ],
    selectedModel: "test-model",
    systemMessage: "You are a helpful assistant.",
    agentId: "agent_123",
    createdAt: 1710000000,
    updatedAt: 1710001000,
  };

  const mockAgentSessions = [
    {
      session_id: "session_123",
      session_name: "Test Session",
      started_at: "2024-01-01T00:00:00Z",
      turns: [],
    },
    {
      session_id: "session_456",
      session_name: "Another Session",
      started_at: "2024-01-01T01:00:00Z",
      turns: [],
    },
  ];

  beforeEach(() => {
    jest.clearAllMocks();
    localStorageMock.getItem.mockReturnValue(null);
    localStorageMock.setItem.mockImplementation(() => {});
    mockClient.agents.session.list.mockResolvedValue({
      data: mockAgentSessions,
    });
    mockClient.agents.session.create.mockResolvedValue({
      session_id: "new_session_123",
    });
    mockClient.agents.session.delete.mockResolvedValue(undefined);
    mockClient.agents.session.retrieve.mockResolvedValue({
      session_id: "test-session",
      session_name: "Test Session",
      started_at: new Date().toISOString(),
      turns: [],
    });
    uuidCounter = 0; // Reset UUID counter for consistent test behavior
  });

  describe("Component Rendering", () => {
    test("does not render when no agent is selected", async () => {
      const { container } = await act(async () => {
        return render(
          <Conversations
            selectedAgentId=""
            currentSession={null}
            onSessionChange={mockOnSessionChange}
            onNewSession={mockOnNewSession}
          />
        );
      });

      expect(container.firstChild).toBeNull();
    });

    test("renders loading state initially", async () => {
      mockClient.agents.session.list.mockImplementation(
        () => new Promise(() => {}) // Never resolves to simulate loading
      );

      await act(async () => {
        render(
          <Conversations
            selectedAgentId="agent_123"
            currentSession={null}
            onSessionChange={mockOnSessionChange}
            onNewSession={mockOnNewSession}
          />
        );
      });

      expect(screen.getByText("Select Session")).toBeInTheDocument();
      // When loading, the "+ New" button should be disabled
      expect(screen.getByText("+ New")).toBeDisabled();
    });

    test("renders session selector when agent sessions are loaded", async () => {
      await act(async () => {
        render(
          <Conversations
            selectedAgentId="agent_123"
            currentSession={null}
            onSessionChange={mockOnSessionChange}
            onNewSession={mockOnNewSession}
          />
        );
      });

      await waitFor(() => {
        expect(screen.getByText("Select Session")).toBeInTheDocument();
      });
    });

    test("renders current session name when session is selected", async () => {
      await act(async () => {
        render(
          <Conversations
            selectedAgentId="agent_123"
            currentSession={mockSession}
            onSessionChange={mockOnSessionChange}
            onNewSession={mockOnNewSession}
          />
        );
      });

      await waitFor(() => {
        expect(screen.getByText("Test Session")).toBeInTheDocument();
      });
    });
  });

  describe("Agent API Integration", () => {
    test("loads sessions from agent API on mount", async () => {
      await act(async () => {
        render(
          <Conversations
            selectedAgentId="agent_123"
            currentSession={mockSession}
            onSessionChange={mockOnSessionChange}
            onNewSession={mockOnNewSession}
          />
        );
      });

      await waitFor(() => {
        expect(mockClient.agents.session.list).toHaveBeenCalledWith(
          "agent_123"
        );
      });
    });

    test("handles API errors gracefully", async () => {
      mockClient.agents.session.list.mockRejectedValue(new Error("API Error"));
      const consoleSpy = jest
        .spyOn(console, "error")
        .mockImplementation(() => {});

      await act(async () => {
        render(
          <Conversations
            selectedAgentId="agent_123"
            currentSession={mockSession}
            onSessionChange={mockOnSessionChange}
            onNewSession={mockOnNewSession}
          />
        );
      });

      await waitFor(() => {
        expect(consoleSpy).toHaveBeenCalledWith(
          "Error loading agent sessions:",
          expect.any(Error)
        );
      });

      consoleSpy.mockRestore();
    });
  });

  describe("Error Handling", () => {
    test("component renders without crashing when API is unavailable", async () => {
      mockClient.agents.session.list.mockRejectedValue(
        new Error("Network Error")
      );
      const consoleSpy = jest
        .spyOn(console, "error")
        .mockImplementation(() => {});

      await act(async () => {
        render(
          <Conversations
            selectedAgentId="agent_123"
            currentSession={mockSession}
            onSessionChange={mockOnSessionChange}
            onNewSession={mockOnNewSession}
          />
        );
      });

      // Should still render the session manager with the select trigger
      expect(screen.getByRole("combobox")).toBeInTheDocument();
      expect(screen.getByText("+ New")).toBeInTheDocument();
      consoleSpy.mockRestore();
    });
  });
});

describe("SessionUtils", () => {
  beforeEach(() => {
    jest.clearAllMocks();
    localStorageMock.getItem.mockReturnValue(null);
    localStorageMock.setItem.mockImplementation(() => {});
  });

  describe("saveCurrentSessionId", () => {
    test("saves session ID to localStorage", () => {
      SessionUtils.saveCurrentSessionId("test-session-id");

      expect(localStorageMock.setItem).toHaveBeenCalledWith(
        "chat-playground-current-session",
        "test-session-id"
      );
    });
  });

  describe("createDefaultSession", () => {
    test("creates default session with agent ID", () => {
      const result = SessionUtils.createDefaultSession("agent_123");

      expect(result).toEqual(
        expect.objectContaining({
          name: "Default Session",
          messages: [],
          selectedModel: "",
          systemMessage: "You are a helpful assistant.",
          agentId: "agent_123",
        })
      );
      expect(result.id).toBeTruthy();
      expect(result.createdAt).toBeTruthy();
      expect(result.updatedAt).toBeTruthy();
    });

    test("creates default session with inherited model", () => {
      const result = SessionUtils.createDefaultSession(
        "agent_123",
        "inherited-model"
      );

      expect(result.selectedModel).toBe("inherited-model");
      expect(result.agentId).toBe("agent_123");
    });

    test("creates unique session IDs", () => {
      const originalNow = Date.now;
      let mockTime = 1710005000;
      Date.now = jest.fn(() => ++mockTime);

      const session1 = SessionUtils.createDefaultSession("agent_123");
      const session2 = SessionUtils.createDefaultSession("agent_123");

      expect(session1.id).not.toBe(session2.id);

      Date.now = originalNow;
    });

    test("sets creation and update timestamps", () => {
      const result = SessionUtils.createDefaultSession("agent_123");

      expect(result.createdAt).toBeTruthy();
      expect(result.updatedAt).toBeTruthy();
      expect(typeof result.createdAt).toBe("number");
      expect(typeof result.updatedAt).toBe("number");
    });
  });
});
