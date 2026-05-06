import React from "react";
import {
  render,
  screen,
  fireEvent,
  waitFor,
  act,
} from "@testing-library/react";
import "@testing-library/jest-dom";
import ChatPlaygroundPage from "./page";

const mockClient = {
  models: {
    list: jest.fn(),
  },
  responses: {
    create: jest.fn(),
  },
  conversations: {
    create: jest.fn(),
    items: {
      list: jest.fn(),
    },
  },
  vectorStores: {
    list: jest.fn(),
  },
};

jest.mock("@/hooks/use-auth-client", () => ({
  useAuthClient: jest.fn(() => mockClient),
}));

jest.mock("next/navigation", () => ({
  useSearchParams: jest.fn(() => ({
    get: jest.fn(() => null),
  })),
}));

jest.mock("@/components/chat-playground/chat", () => ({
  Chat: jest.fn(
    ({
      messages,
      handleSubmit,
      input,
      handleInputChange,
      isGenerating,
      append,
      suggestions,
    }) => (
      <div data-testid="chat-component">
        <div data-testid="messages-count">{messages.length}</div>
        <input
          data-testid="chat-input"
          value={input}
          onChange={handleInputChange}
          disabled={isGenerating}
        />
        <button data-testid="submit-button" onClick={handleSubmit}>
          Submit
        </button>
        {suggestions?.map((suggestion: string, index: number) => (
          <button
            key={index}
            data-testid={`suggestion-${index}`}
            onClick={() => append({ role: "user", content: suggestion })}
          >
            {suggestion}
          </button>
        ))}
      </div>
    )
  ),
}));

const mockModels = [
  {
    id: "test-model-1",
    custom_metadata: {
      model_type: "llm",
    },
  },
  {
    id: "test-model-2",
    custom_metadata: {
      model_type: "llm",
    },
  },
];

describe("ChatPlaygroundPage", () => {
  beforeEach(() => {
    jest.clearAllMocks();
    Element.prototype.scrollIntoView = jest.fn();
    mockClient.models.list.mockResolvedValue(mockModels);
    mockClient.vectorStores.list.mockResolvedValue({ data: [] });
    mockClient.conversations.create.mockResolvedValue({
      id: "conv_test123",
    });
  });

  describe("Initial Rendering", () => {
    test("renders model selector and loads models", async () => {
      await act(async () => {
        render(<ChatPlaygroundPage />);
      });

      await waitFor(() => {
        expect(mockClient.models.list).toHaveBeenCalled();
      });

      expect(screen.getAllByRole("combobox")).toHaveLength(1);
    });

    test("shows settings panel", async () => {
      await act(async () => {
        render(<ChatPlaygroundPage />);
      });

      await waitFor(() => {
        expect(screen.getByText("Settings")).toBeInTheDocument();
        expect(screen.getByText("Model Configuration")).toBeInTheDocument();
        expect(screen.getByText("Tools")).toBeInTheDocument();
      });
    });

    test("shows clear button", async () => {
      await act(async () => {
        render(<ChatPlaygroundPage />);
      });

      expect(screen.getByText("Clear Chat")).toBeInTheDocument();
    });
  });

  describe("Error Handling", () => {
    test("handles model loading errors gracefully", async () => {
      mockClient.models.list.mockRejectedValue(
        new Error("Failed to load models")
      );
      const consoleSpy = jest
        .spyOn(console, "error")
        .mockImplementation(() => {});

      await act(async () => {
        render(<ChatPlaygroundPage />);
      });

      await waitFor(() => {
        expect(consoleSpy).toHaveBeenCalledWith(
          "Error fetching models:",
          expect.any(Error)
        );
      });

      consoleSpy.mockRestore();
    });
  });

  describe("Chat Interaction", () => {
    test("creates conversation and sends message using responses API", async () => {
      const mockStream = {
        [Symbol.asyncIterator]: async function* () {
          yield {
            type: "response.created",
            response: { id: "resp_123" },
          };
          yield {
            type: "response.output_text.delta",
            delta: "Hello ",
          };
          yield {
            type: "response.output_text.delta",
            delta: "world!",
          };
          yield {
            type: "response.completed",
            response: { id: "resp_123" },
          };
        },
      };

      mockClient.responses.create.mockResolvedValue(mockStream);

      await act(async () => {
        render(<ChatPlaygroundPage />);
      });

      // Select a model first
      await waitFor(() => {
        expect(screen.getByRole("combobox")).toBeInTheDocument();
      });

      const combobox = screen.getByRole("combobox");
      await act(async () => {
        fireEvent.click(combobox);
      });

      await waitFor(() => {
        const option = screen
          .getAllByText("test-model-1")
          .find(
            el =>
              el.getAttribute("role") === "option" ||
              el.closest('[role="option"]')
          );
        if (option) fireEvent.click(option);
      });

      await waitFor(() => {
        expect(screen.getByTestId("chat-component")).toBeInTheDocument();
      });

      const chatInput = screen.getByTestId("chat-input");
      await act(async () => {
        fireEvent.change(chatInput, {
          target: { value: "Hello, how are you?" },
        });
      });

      const submitButton = screen.getByTestId("submit-button");
      await act(async () => {
        fireEvent.click(submitButton);
      });

      await waitFor(() => {
        expect(mockClient.conversations.create).toHaveBeenCalledWith({});
        expect(mockClient.responses.create).toHaveBeenCalledWith(
          expect.objectContaining({
            stream: true,
            conversation: "conv_test123",
          }),
          expect.any(Object)
        );
      });
    });

    test("handles API errors during message sending", async () => {
      mockClient.responses.create.mockRejectedValue(new Error("API Error"));

      const consoleSpy = jest
        .spyOn(console, "error")
        .mockImplementation(() => {});

      await act(async () => {
        render(<ChatPlaygroundPage />);
      });

      // Select model
      const combobox = screen.getByRole("combobox");
      await act(async () => {
        fireEvent.click(combobox);
      });

      await waitFor(() => {
        const option = screen
          .getAllByText("test-model-1")
          .find(
            el =>
              el.getAttribute("role") === "option" ||
              el.closest('[role="option"]')
          );
        if (option) fireEvent.click(option);
      });

      await waitFor(() => {
        expect(screen.getByTestId("chat-component")).toBeInTheDocument();
      });

      const chatInput = screen.getByTestId("chat-input");
      await act(async () => {
        fireEvent.change(chatInput, {
          target: { value: "Test message" },
        });
      });

      const submitButton = screen.getByTestId("submit-button");
      await act(async () => {
        fireEvent.click(submitButton);
      });

      await waitFor(() => {
        expect(consoleSpy).toHaveBeenCalledWith(
          "Error sending message:",
          expect.any(Error)
        );
      });

      consoleSpy.mockRestore();
    });
  });

  describe("Tools Configuration", () => {
    test("shows web search, file search, and MCP options", async () => {
      await act(async () => {
        render(<ChatPlaygroundPage />);
      });

      await waitFor(() => {
        expect(screen.getByText("Web Search")).toBeInTheDocument();
        expect(screen.getByText("File Search")).toBeInTheDocument();
        expect(screen.getByText("MCP Server")).toBeInTheDocument();
      });
    });
  });
});
