import React from "react";
import { render, screen } from "@testing-library/react";
import "@testing-library/jest-dom";
import { ChatCompletionDetailView } from "./chat-completion-detail";
import { ChatCompletion } from "@/lib/types";

// Initial test file setup for ChatCompletionDetailView

describe("ChatCompletionDetailView", () => {
  test("renders skeleton UI when isLoading is true", () => {
    const { container } = render(
      <ChatCompletionDetailView
        completion={null}
        isLoading={true}
        error={null}
        id="test-id"
      />
    );
    // Use the data-slot attribute for Skeletons
    const skeletons = container.querySelectorAll('[data-slot="skeleton"]');
    expect(skeletons.length).toBeGreaterThan(0);
  });

  test("renders error message when error prop is provided", () => {
    render(
      <ChatCompletionDetailView
        completion={null}
        isLoading={false}
        error={{ name: "Error", message: "Network Error" }}
        id="err-id"
      />
    );
    expect(
      screen.getByText(/Error loading details for ID err-id: Network Error/)
    ).toBeInTheDocument();
  });

  test("renders default error message when error.message is empty", () => {
    render(
      <ChatCompletionDetailView
        completion={null}
        isLoading={false}
        error={{ name: "Error", message: "" }}
        id="err-id"
      />
    );
    // Use regex to match the error message regardless of whitespace
    expect(
      screen.getByText(/Error loading details for ID\s*err-id\s*:/)
    ).toBeInTheDocument();
  });

  test("renders error message when error prop is an object without message", () => {
    render(
      <ChatCompletionDetailView
        completion={null}
        isLoading={false}
        error={{} as Error}
        id="err-id"
      />
    );
    // Use regex to match the error message regardless of whitespace
    expect(
      screen.getByText(/Error loading details for ID\s*err-id\s*:/)
    ).toBeInTheDocument();
  });

  test("renders not found message when completion is null and not loading/error", () => {
    render(
      <ChatCompletionDetailView
        completion={null}
        isLoading={false}
        error={null}
        id="notfound-id"
      />
    );
    expect(
      screen.getByText("No details found for ID: notfound-id.")
    ).toBeInTheDocument();
  });

  test("renders input, output, and properties for valid completion", () => {
    const mockCompletion: ChatCompletion = {
      id: "comp_123",
      object: "chat.completion",
      created: 1710000000,
      model: "llama-test-model",
      choices: [
        {
          index: 0,
          message: { role: "assistant", content: "Test output" },
          finish_reason: "stop",
        },
      ],
      input_messages: [{ role: "user", content: "Test input" }],
    };
    render(
      <ChatCompletionDetailView
        completion={mockCompletion}
        isLoading={false}
        error={null}
        id={mockCompletion.id}
      />
    );
    // Input
    expect(screen.getByText("Input")).toBeInTheDocument();
    expect(screen.getByText("Test input")).toBeInTheDocument();
    // Output
    expect(screen.getByText("Output")).toBeInTheDocument();
    expect(screen.getByText("Test output")).toBeInTheDocument();
    // Properties
    expect(screen.getByText("Properties")).toBeInTheDocument();
    expect(screen.getByText("Created:")).toBeInTheDocument();
    expect(
      screen.getByText(new Date(1710000000 * 1000).toLocaleString())
    ).toBeInTheDocument();
    expect(screen.getByText("ID:")).toBeInTheDocument();
    expect(screen.getByText("comp_123")).toBeInTheDocument();
    expect(screen.getByText("Model:")).toBeInTheDocument();
    expect(screen.getByText("llama-test-model")).toBeInTheDocument();
    expect(screen.getByText("Finish Reason:")).toBeInTheDocument();
    expect(screen.getByText("stop")).toBeInTheDocument();
  });

  test("renders tool call in output and properties when present", () => {
    const toolCall = {
      function: { name: "search", arguments: '{"query":"llama"}' },
    };
    const mockCompletion: ChatCompletion = {
      id: "comp_tool",
      object: "chat.completion",
      created: 1710001000,
      model: "llama-tool-model",
      choices: [
        {
          index: 0,
          message: {
            role: "assistant",
            content: "Tool output",
            tool_calls: [toolCall],
          },
          finish_reason: "stop",
        },
      ],
      input_messages: [{ role: "user", content: "Tool input" }],
    };
    render(
      <ChatCompletionDetailView
        completion={mockCompletion}
        isLoading={false}
        error={null}
        id={mockCompletion.id}
      />
    );
    // Output should include the tool call block (should be present twice: input and output)
    const toolCallLabels = screen.getAllByText("Tool Call");
    expect(toolCallLabels.length).toBeGreaterThanOrEqual(1); // At least one, but could be two
    // The tool call block should contain the formatted tool call string in both input and output
    const toolCallBlocks = screen.getAllByText('search({"query":"llama"})');
    expect(toolCallBlocks.length).toBe(2);
    // Properties should include the tool call name
    expect(screen.getByText("Functions/Tools Called:")).toBeInTheDocument();
    expect(screen.getByText("search")).toBeInTheDocument();
  });

  test("handles missing/empty fields gracefully", () => {
    const mockCompletion: ChatCompletion = {
      id: "comp_edge",
      object: "chat.completion",
      created: 1710002000,
      model: "llama-edge-model",
      choices: [], // No choices
      input_messages: [], // No input messages
    };
    render(
      <ChatCompletionDetailView
        completion={mockCompletion}
        isLoading={false}
        error={null}
        id={mockCompletion.id}
      />
    );
    // Input section should be present but empty
    expect(screen.getByText("Input")).toBeInTheDocument();
    // Output section should show fallback message
    expect(
      screen.getByText("No message found in assistant's choice.")
    ).toBeInTheDocument();
    // Properties should show N/A for finish reason
    expect(screen.getByText("Finish Reason:")).toBeInTheDocument();
    expect(screen.getByText("N/A")).toBeInTheDocument();
  });
});
