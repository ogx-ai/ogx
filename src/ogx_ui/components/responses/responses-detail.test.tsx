import React from "react";
import { render, screen } from "@testing-library/react";
import "@testing-library/jest-dom";
import { ResponseDetailView } from "./responses-detail";
import { OpenAIResponse, InputItemListResponse } from "@/lib/types";

describe("ResponseDetailView", () => {
  const defaultProps = {
    response: null,
    inputItems: null,
    isLoading: false,
    isLoadingInputItems: false,
    error: null,
    inputItemsError: null,
    id: "test_id",
  };

  describe("Loading State", () => {
    test("renders loading skeleton when isLoading is true", () => {
      const { container } = render(
        <ResponseDetailView {...defaultProps} isLoading={true} />
      );

      // Check for skeleton elements
      const skeletons = container.querySelectorAll('[data-slot="skeleton"]');
      expect(skeletons.length).toBeGreaterThan(0);

      // The title is replaced by a skeleton when loading, so we shouldn't expect the text
    });
  });

  describe("Error State", () => {
    test("renders error message when error prop is provided", () => {
      const errorMessage = "Network Error";
      render(
        <ResponseDetailView
          {...defaultProps}
          error={{ name: "Error", message: errorMessage }}
        />
      );

      expect(screen.getByText("Responses Details")).toBeInTheDocument();
      // The error message is split across elements, so we check for parts
      expect(
        screen.getByText(/Error loading details for ID/)
      ).toBeInTheDocument();
      expect(screen.getByText(/test_id/)).toBeInTheDocument();
      expect(screen.getByText(/Network Error/)).toBeInTheDocument();
    });

    test("renders default error message when error.message is not available", () => {
      render(
        <ResponseDetailView
          {...defaultProps}
          error={{ name: "Error", message: "" }}
        />
      );

      expect(
        screen.getByText(/Error loading details for ID/)
      ).toBeInTheDocument();
      expect(screen.getByText(/test_id/)).toBeInTheDocument();
    });
  });

  describe("Not Found State", () => {
    test("renders not found message when response is null and not loading/error", () => {
      render(<ResponseDetailView {...defaultProps} response={null} />);

      expect(screen.getByText("Responses Details")).toBeInTheDocument();
      // The message is split across elements
      expect(screen.getByText(/No details found for ID:/)).toBeInTheDocument();
      expect(screen.getByText(/test_id/)).toBeInTheDocument();
    });
  });

  describe("Response Data Rendering", () => {
    const mockResponse: OpenAIResponse = {
      id: "resp_123",
      object: "response",
      created_at: 1710000000,
      model: "llama-test-model",
      status: "completed",
      output: [
        {
          type: "message",
          role: "assistant",
          content: "Test response output",
        },
      ],
      input: [
        {
          type: "message",
          role: "user",
          content: "Test input message",
        },
      ],
      temperature: 0.7,
      top_p: 0.9,
      parallel_tool_calls: true,
      previous_response_id: "prev_resp_456",
    };

    test("renders response data with input and output sections", () => {
      render(<ResponseDetailView {...defaultProps} response={mockResponse} />);

      // Check main sections
      expect(screen.getByText("Responses Details")).toBeInTheDocument();
      expect(screen.getByText("Input")).toBeInTheDocument();
      expect(screen.getByText("Output")).toBeInTheDocument();

      // Check input content
      expect(screen.getByText("Test input message")).toBeInTheDocument();
      expect(screen.getByText("User")).toBeInTheDocument();

      // Check output content
      expect(screen.getByText("Test response output")).toBeInTheDocument();
      expect(screen.getByText("Assistant")).toBeInTheDocument();
    });

    test("renders properties sidebar with all response metadata", () => {
      render(<ResponseDetailView {...defaultProps} response={mockResponse} />);

      // Check properties - use regex to handle text split across elements
      expect(screen.getByText(/Created/)).toBeInTheDocument();
      expect(
        screen.getByText(new Date(1710000000 * 1000).toLocaleString())
      ).toBeInTheDocument();

      // Check for the specific ID label (not Previous Response ID)
      expect(
        screen.getByText((content, element) => {
          return element?.tagName === "STRONG" && content === "ID:";
        })
      ).toBeInTheDocument();
      expect(screen.getByText("resp_123")).toBeInTheDocument();

      expect(screen.getByText(/Model/)).toBeInTheDocument();
      expect(screen.getByText("llama-test-model")).toBeInTheDocument();

      expect(screen.getByText(/Status/)).toBeInTheDocument();
      expect(screen.getByText("completed")).toBeInTheDocument();

      expect(screen.getByText(/Temperature/)).toBeInTheDocument();
      expect(screen.getByText("0.7")).toBeInTheDocument();

      expect(screen.getByText(/Top P/)).toBeInTheDocument();
      expect(screen.getByText("0.9")).toBeInTheDocument();

      expect(screen.getByText(/Parallel Tool Calls/)).toBeInTheDocument();
      expect(screen.getByText("Yes")).toBeInTheDocument();

      expect(screen.getByText(/Previous Response ID/)).toBeInTheDocument();
      expect(screen.getByText("prev_resp_456")).toBeInTheDocument();
    });

    test("handles optional properties correctly", () => {
      const minimalResponse: OpenAIResponse = {
        id: "resp_minimal",
        object: "response",
        created_at: 1710000000,
        model: "test-model",
        status: "completed",
        output: [],
        input: [],
      };

      render(
        <ResponseDetailView {...defaultProps} response={minimalResponse} />
      );

      // Should show required properties
      expect(screen.getByText("resp_minimal")).toBeInTheDocument();
      expect(screen.getByText("test-model")).toBeInTheDocument();
      expect(screen.getByText("completed")).toBeInTheDocument();

      // Should not show optional properties
      expect(screen.queryByText("Temperature")).not.toBeInTheDocument();
      expect(screen.queryByText("Top P")).not.toBeInTheDocument();
      expect(screen.queryByText("Parallel Tool Calls")).not.toBeInTheDocument();
      expect(
        screen.queryByText("Previous Response ID")
      ).not.toBeInTheDocument();
    });

    test("renders error information when response has error", () => {
      const errorResponse: OpenAIResponse = {
        ...mockResponse,
        error: {
          code: "invalid_request",
          message: "The request was invalid",
        },
      };

      render(<ResponseDetailView {...defaultProps} response={errorResponse} />);

      // The error is shown in the properties sidebar, not as a separate "Error" label
      expect(
        screen.getByText("invalid_request: The request was invalid")
      ).toBeInTheDocument();
    });
  });

  describe("Input Items Handling", () => {
    const mockResponse: OpenAIResponse = {
      id: "resp_123",
      object: "response",
      created_at: 1710000000,
      model: "test-model",
      status: "completed",
      output: [{ type: "message", role: "assistant", content: "output" }],
      input: [{ type: "message", role: "user", content: "fallback input" }],
    };

    test("shows loading state for input items", () => {
      render(
        <ResponseDetailView
          {...defaultProps}
          response={mockResponse}
          isLoadingInputItems={true}
        />
      );

      // Check for skeleton loading in input items section
      const { container } = render(
        <ResponseDetailView
          {...defaultProps}
          response={mockResponse}
          isLoadingInputItems={true}
        />
      );

      const skeletons = container.querySelectorAll('[data-slot="skeleton"]');
      expect(skeletons.length).toBeGreaterThan(0);
    });

    test("shows error message for input items with fallback", () => {
      render(
        <ResponseDetailView
          {...defaultProps}
          response={mockResponse}
          inputItemsError={{
            name: "Error",
            message: "Failed to load input items",
          }}
        />
      );

      expect(
        screen.getByText(
          "Error loading input items: Failed to load input items"
        )
      ).toBeInTheDocument();
      expect(
        screen.getByText("Falling back to response input data.")
      ).toBeInTheDocument();

      // Should still show fallback input data
      expect(screen.getByText("fallback input")).toBeInTheDocument();
    });

    test("uses input items data when available", () => {
      const mockInputItems: InputItemListResponse = {
        object: "list",
        data: [
          {
            type: "message",
            role: "user",
            content: "input from items API",
          },
        ],
      };

      render(
        <ResponseDetailView
          {...defaultProps}
          response={mockResponse}
          inputItems={mockInputItems}
        />
      );

      // Should show input items data, not response.input
      expect(screen.getByText("input from items API")).toBeInTheDocument();
      expect(screen.queryByText("fallback input")).not.toBeInTheDocument();
    });

    test("falls back to response.input when input items is empty", () => {
      const emptyInputItems: InputItemListResponse = {
        object: "list",
        data: [],
      };

      render(
        <ResponseDetailView
          {...defaultProps}
          response={mockResponse}
          inputItems={emptyInputItems}
        />
      );

      // Should show fallback input data
      expect(screen.getByText("fallback input")).toBeInTheDocument();
    });

    test("shows no input message when no data available", () => {
      const responseWithoutInput: OpenAIResponse = {
        ...mockResponse,
        input: [],
      };

      render(
        <ResponseDetailView
          {...defaultProps}
          response={responseWithoutInput}
          inputItems={null}
        />
      );

      expect(screen.getByText("No input data available.")).toBeInTheDocument();
    });
  });

  describe("Input Display Components", () => {
    test("renders string content input correctly", () => {
      const mockResponse: OpenAIResponse = {
        id: "resp_123",
        object: "response",
        created_at: 1710000000,
        model: "test-model",
        status: "completed",
        output: [],
        input: [
          {
            type: "message",
            role: "user",
            content: "Simple string input",
          },
        ],
      };

      render(<ResponseDetailView {...defaultProps} response={mockResponse} />);

      expect(screen.getByText("Simple string input")).toBeInTheDocument();
      expect(screen.getByText("User")).toBeInTheDocument();
    });

    test("renders array content input correctly", () => {
      const mockResponse: OpenAIResponse = {
        id: "resp_123",
        object: "response",
        created_at: 1710000000,
        model: "test-model",
        status: "completed",
        output: [],
        input: [
          {
            type: "message",
            role: "user",
            content: [
              { type: "input_text", text: "First part" },
              { type: "output_text", text: "Second part" },
            ],
          },
        ],
      };

      render(<ResponseDetailView {...defaultProps} response={mockResponse} />);

      expect(screen.getByText("First part Second part")).toBeInTheDocument();
      expect(screen.getByText("User")).toBeInTheDocument();
    });

    test("renders non-message input types correctly", () => {
      const mockResponse: OpenAIResponse = {
        id: "resp_123",
        object: "response",
        created_at: 1710000000,
        model: "test-model",
        status: "completed",
        output: [],
        input: [
          {
            type: "function_call",
            content: "function call content",
          },
        ],
      };

      render(<ResponseDetailView {...defaultProps} response={mockResponse} />);

      expect(screen.getByText("function call content")).toBeInTheDocument();
      // Use getAllByText to find the specific "Input" with the type detail
      const inputElements = screen.getAllByText("Input");
      expect(inputElements.length).toBeGreaterThan(0);
      expect(screen.getByText("(function_call)")).toBeInTheDocument();
    });

    test("handles input with object content", () => {
      const mockResponse: OpenAIResponse = {
        id: "resp_123",
        object: "response",
        created_at: 1710000000,
        model: "test-model",
        status: "completed",
        output: [],
        input: [
          {
            type: "custom_type",
            content: JSON.stringify({ key: "value", nested: { data: "test" } }),
          },
        ],
      };

      render(<ResponseDetailView {...defaultProps} response={mockResponse} />);

      // Should show JSON stringified content (without quotes around keys in the rendered output)
      expect(screen.getByText(/key.*value/)).toBeInTheDocument();
      // Use getAllByText to find the specific "Input" with the type detail
      const inputElements = screen.getAllByText("Input");
      expect(inputElements.length).toBeGreaterThan(0);
      expect(screen.getByText("(custom_type)")).toBeInTheDocument();
    });

    test("renders function call input correctly", () => {
      const mockResponse: OpenAIResponse = {
        id: "resp_123",
        object: "response",
        created_at: 1710000000,
        model: "test-model",
        status: "completed",
        output: [],
        input: [
          {
            type: "function_call",
            id: "call_456",
            status: "completed",
            name: "input_function",
            arguments: '{"param": "value"}',
          },
        ],
      };

      render(<ResponseDetailView {...defaultProps} response={mockResponse} />);

      expect(
        screen.getByText('input_function({"param": "value"})')
      ).toBeInTheDocument();
      expect(screen.getByText("Function Call")).toBeInTheDocument();
    });

    test("renders web search call input correctly", () => {
      const mockResponse: OpenAIResponse = {
        id: "resp_123",
        object: "response",
        created_at: 1710000000,
        model: "test-model",
        status: "completed",
        output: [],
        input: [
          {
            type: "web_search_call",
            id: "search_789",
            status: "completed",
          },
        ],
      };

      render(<ResponseDetailView {...defaultProps} response={mockResponse} />);

      expect(
        screen.getByText("web_search_call(status: completed)")
      ).toBeInTheDocument();
      expect(screen.getByText("Function Call")).toBeInTheDocument();
      expect(screen.getByText("(Web Search)")).toBeInTheDocument();
    });
  });

  describe("Output Display Components", () => {
    test("renders message output with string content", () => {
      const mockResponse: OpenAIResponse = {
        id: "resp_123",
        object: "response",
        created_at: 1710000000,
        model: "test-model",
        status: "completed",
        output: [
          {
            type: "message",
            role: "assistant",
            content: "Simple string output",
          },
        ],
        input: [],
      };

      render(<ResponseDetailView {...defaultProps} response={mockResponse} />);

      expect(screen.getByText("Simple string output")).toBeInTheDocument();
      expect(screen.getByText("Assistant")).toBeInTheDocument();
    });

    test("renders message output with array content", () => {
      const mockResponse: OpenAIResponse = {
        id: "resp_123",
        object: "response",
        created_at: 1710000000,
        model: "test-model",
        status: "completed",
        output: [
          {
            type: "message",
            role: "assistant",
            content: [
              { type: "output_text", text: "First output" },
              { type: "input_text", text: "Second output" },
            ],
          },
        ],
        input: [],
      };

      render(<ResponseDetailView {...defaultProps} response={mockResponse} />);

      expect(
        screen.getByText("First output Second output")
      ).toBeInTheDocument();
      expect(screen.getByText("Assistant")).toBeInTheDocument();
    });

    test("renders function call output correctly", () => {
      const mockResponse: OpenAIResponse = {
        id: "resp_123",
        object: "response",
        created_at: 1710000000,
        model: "test-model",
        status: "completed",
        output: [
          {
            type: "function_call",
            id: "call_123",
            status: "completed",
            name: "search_function",
            arguments: '{"query": "test"}',
          },
        ],
        input: [],
      };

      render(<ResponseDetailView {...defaultProps} response={mockResponse} />);

      expect(
        screen.getByText('search_function({"query": "test"})')
      ).toBeInTheDocument();
      expect(screen.getByText("Function Call")).toBeInTheDocument();
    });

    test("renders function call output without arguments", () => {
      const mockResponse: OpenAIResponse = {
        id: "resp_123",
        object: "response",
        created_at: 1710000000,
        model: "test-model",
        status: "completed",
        output: [
          {
            type: "function_call",
            id: "call_123",
            status: "completed",
            name: "simple_function",
          },
        ],
        input: [],
      };

      render(<ResponseDetailView {...defaultProps} response={mockResponse} />);

      expect(screen.getByText("simple_function({})")).toBeInTheDocument();
      expect(screen.getByText(/Function Call/)).toBeInTheDocument();
    });

    test("renders web search call output correctly", () => {
      const mockResponse: OpenAIResponse = {
        id: "resp_123",
        object: "response",
        created_at: 1710000000,
        model: "test-model",
        status: "completed",
        output: [
          {
            type: "web_search_call",
            id: "search_123",
            status: "completed",
          },
        ],
        input: [],
      };

      render(<ResponseDetailView {...defaultProps} response={mockResponse} />);

      expect(
        screen.getByText("web_search_call(status: completed)")
      ).toBeInTheDocument();
      expect(screen.getByText(/Function Call/)).toBeInTheDocument();
      expect(screen.getByText("(Web Search)")).toBeInTheDocument();
    });

    test("renders unknown output types with JSON fallback", () => {
      const mockResponse: OpenAIResponse = {
        id: "resp_123",
        object: "response",
        created_at: 1710000000,
        model: "test-model",
        status: "completed",
        output: [
          {
            type: "unknown_type",
            custom_field: "custom_value",
            data: { nested: "object" },
          } as unknown,
        ],
        input: [],
      };

      render(<ResponseDetailView {...defaultProps} response={mockResponse} />);

      // Should show JSON stringified content
      expect(
        screen.getByText(/custom_field.*custom_value/)
      ).toBeInTheDocument();
      expect(screen.getByText("(unknown_type)")).toBeInTheDocument();
    });

    test("shows no output message when output array is empty", () => {
      const mockResponse: OpenAIResponse = {
        id: "resp_123",
        object: "response",
        created_at: 1710000000,
        model: "test-model",
        status: "completed",
        output: [],
        input: [],
      };

      render(<ResponseDetailView {...defaultProps} response={mockResponse} />);

      expect(screen.getByText("No output data available.")).toBeInTheDocument();
    });

    test("groups function call with its output correctly", () => {
      const mockResponse: OpenAIResponse = {
        id: "resp_123",
        object: "response",
        created_at: 1710000000,
        model: "test-model",
        status: "completed",
        output: [
          {
            type: "function_call",
            id: "call_123",
            status: "completed",
            name: "get_weather",
            arguments: '{"city": "Tokyo"}',
          },
          {
            type: "message",
            role: "assistant",
            call_id: "call_123",
            content: "sunny and warm",
          } as unknown, // Using any to bypass the type restriction for this test
        ],
        input: [],
      };

      render(<ResponseDetailView {...defaultProps} response={mockResponse} />);

      // Should show the function call and message as separate items (not grouped)
      expect(screen.getByText("Function Call")).toBeInTheDocument();
      expect(
        screen.getByText('get_weather({"city": "Tokyo"})')
      ).toBeInTheDocument();
      expect(screen.getByText("Assistant")).toBeInTheDocument();
      expect(screen.getByText("sunny and warm")).toBeInTheDocument();

      // Should NOT have the grouped "Arguments" and "Output" labels
      expect(screen.queryByText("Arguments")).not.toBeInTheDocument();
    });

    test("groups function call with function_call_output correctly", () => {
      const mockResponse: OpenAIResponse = {
        id: "resp_123",
        object: "response",
        created_at: 1710000000,
        model: "test-model",
        status: "completed",
        output: [
          {
            type: "function_call",
            call_id: "call_123",
            status: "completed",
            name: "get_weather",
            arguments: '{"city": "Tokyo"}',
          },
          {
            type: "function_call_output",
            id: "fc_68364957013081...",
            status: "completed",
            call_id: "call_123",
            output: "sunny and warm",
          } as unknown,
        ],
        input: [],
      };

      render(<ResponseDetailView {...defaultProps} response={mockResponse} />);

      // Should show the function call grouped with its clean output
      expect(screen.getByText("Function Call")).toBeInTheDocument();
      expect(screen.getByText("Arguments")).toBeInTheDocument();
      expect(
        screen.getByText('get_weather({"city": "Tokyo"})')
      ).toBeInTheDocument();
      // Use getAllByText since there are multiple "Output" elements (card title and output label)
      const outputElements = screen.getAllByText("Output");
      expect(outputElements.length).toBeGreaterThan(0);
      expect(screen.getByText("sunny and warm")).toBeInTheDocument();
    });
  });

  describe("Edge Cases and Error Handling", () => {
    test("handles missing role in message input", () => {
      const mockResponse: OpenAIResponse = {
        id: "resp_123",
        object: "response",
        created_at: 1710000000,
        model: "test-model",
        status: "completed",
        output: [],
        input: [
          {
            type: "message",
            content: "Message without role",
          },
        ],
      };

      render(<ResponseDetailView {...defaultProps} response={mockResponse} />);

      expect(screen.getByText("Message without role")).toBeInTheDocument();
      expect(screen.getByText("Unknown")).toBeInTheDocument(); // Default role
    });

    test("handles missing name in function call output", () => {
      const mockResponse: OpenAIResponse = {
        id: "resp_123",
        object: "response",
        created_at: 1710000000,
        model: "test-model",
        status: "completed",
        output: [
          {
            type: "function_call",
            id: "call_123",
            status: "completed",
          },
        ],
        input: [],
      };

      render(<ResponseDetailView {...defaultProps} response={mockResponse} />);

      // When name is missing, it falls back to JSON.stringify of the entire output
      const functionCallElements = screen.getAllByText(/function_call/);
      expect(functionCallElements.length).toBeGreaterThan(0);
      expect(screen.getByText(/call_123/)).toBeInTheDocument();
    });
  });
});
