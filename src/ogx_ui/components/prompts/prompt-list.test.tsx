import React from "react";
import { render, screen, fireEvent } from "@testing-library/react";
import "@testing-library/jest-dom";
import { PromptList } from "./prompt-list";
import type { Prompt } from "./types";

describe("PromptList", () => {
  const mockOnEdit = jest.fn();
  const mockOnDelete = jest.fn();

  const defaultProps = {
    prompts: [],
    onEdit: mockOnEdit,
    onDelete: mockOnDelete,
  };

  beforeEach(() => {
    jest.clearAllMocks();
  });

  describe("Empty State", () => {
    test("renders empty message when no prompts", () => {
      render(<PromptList {...defaultProps} />);

      expect(screen.getByText("No prompts yet")).toBeInTheDocument();
    });

    test("shows filtered empty message when search has no results", () => {
      const prompts: Prompt[] = [
        {
          prompt_id: "prompt_123",
          prompt: "Hello world",
          version: 1,
          variables: [],
          is_default: false,
        },
      ];

      render(<PromptList {...defaultProps} prompts={prompts} />);

      // Search for something that doesn't exist
      const searchInput = screen.getByPlaceholderText("Search prompts...");
      fireEvent.change(searchInput, { target: { value: "nonexistent" } });

      expect(
        screen.getByText("No prompts match your filters")
      ).toBeInTheDocument();
    });
  });

  describe("Prompts Display", () => {
    const mockPrompts: Prompt[] = [
      {
        prompt_id: "prompt_123",
        prompt: "Hello {{name}}, how are you?",
        version: 1,
        variables: ["name"],
        is_default: true,
      },
      {
        prompt_id: "prompt_456",
        prompt: "Summarize this {{text}} in {{length}} words",
        version: 2,
        variables: ["text", "length"],
        is_default: false,
      },
      {
        prompt_id: "prompt_789",
        prompt: "Simple prompt with no variables",
        version: 1,
        variables: [],
        is_default: false,
      },
    ];

    test("renders prompts table with correct headers", () => {
      render(<PromptList {...defaultProps} prompts={mockPrompts} />);

      expect(screen.getByText("ID")).toBeInTheDocument();
      expect(screen.getByText("Content")).toBeInTheDocument();
      expect(screen.getByText("Variables")).toBeInTheDocument();
      expect(screen.getByText("Version")).toBeInTheDocument();
      expect(screen.getByText("Actions")).toBeInTheDocument();
    });

    test("renders prompt data correctly", () => {
      render(<PromptList {...defaultProps} prompts={mockPrompts} />);

      // Check prompt IDs
      expect(screen.getByText("prompt_123")).toBeInTheDocument();
      expect(screen.getByText("prompt_456")).toBeInTheDocument();
      expect(screen.getByText("prompt_789")).toBeInTheDocument();

      // Check content
      expect(
        screen.getByText("Hello {{name}}, how are you?")
      ).toBeInTheDocument();
      expect(
        screen.getByText("Summarize this {{text}} in {{length}} words")
      ).toBeInTheDocument();
      expect(
        screen.getByText("Simple prompt with no variables")
      ).toBeInTheDocument();

      // Check versions
      expect(screen.getAllByText("1")).toHaveLength(2); // Two prompts with version 1
      expect(screen.getByText("2")).toBeInTheDocument();

      // Check default badge
      expect(screen.getByText("Default")).toBeInTheDocument();
    });

    test("renders variables correctly", () => {
      render(<PromptList {...defaultProps} prompts={mockPrompts} />);

      // Check variables display
      expect(screen.getByText("name")).toBeInTheDocument();
      expect(screen.getByText("text")).toBeInTheDocument();
      expect(screen.getByText("length")).toBeInTheDocument();
      expect(screen.getByText("None")).toBeInTheDocument(); // For prompt with no variables
    });

    test("prompt ID links are clickable and call onEdit", () => {
      render(<PromptList {...defaultProps} prompts={mockPrompts} />);

      // Click on the first prompt ID link
      const promptLink = screen.getByRole("button", { name: "prompt_123" });
      fireEvent.click(promptLink);

      expect(mockOnEdit).toHaveBeenCalledWith(mockPrompts[0]);
    });

    test("edit buttons call onEdit", () => {
      const { container } = render(
        <PromptList {...defaultProps} prompts={mockPrompts} />
      );

      // Find the action buttons in the table - they should be in the last column
      const actionCells = container.querySelectorAll("td:last-child");
      const firstActionCell = actionCells[0];
      const editButton = firstActionCell?.querySelector("button");

      expect(editButton).toBeInTheDocument();
      fireEvent.click(editButton!);

      expect(mockOnEdit).toHaveBeenCalledWith(mockPrompts[0]);
    });

    test("delete buttons call onDelete with confirmation", () => {
      const originalConfirm = window.confirm;
      window.confirm = jest.fn(() => true);

      const { container } = render(
        <PromptList {...defaultProps} prompts={mockPrompts} />
      );

      // Find the delete button (second button in the first action cell)
      const actionCells = container.querySelectorAll("td:last-child");
      const firstActionCell = actionCells[0];
      const buttons = firstActionCell?.querySelectorAll("button");
      const deleteButton = buttons?.[1]; // Second button should be delete

      expect(deleteButton).toBeInTheDocument();
      fireEvent.click(deleteButton!);

      expect(window.confirm).toHaveBeenCalledWith(
        "Are you sure you want to delete this prompt? This action cannot be undone."
      );
      expect(mockOnDelete).toHaveBeenCalledWith("prompt_123");

      window.confirm = originalConfirm;
    });

    test("delete does not execute when confirmation is cancelled", () => {
      const originalConfirm = window.confirm;
      window.confirm = jest.fn(() => false);

      const { container } = render(
        <PromptList {...defaultProps} prompts={mockPrompts} />
      );

      const actionCells = container.querySelectorAll("td:last-child");
      const firstActionCell = actionCells[0];
      const buttons = firstActionCell?.querySelectorAll("button");
      const deleteButton = buttons?.[1]; // Second button should be delete

      expect(deleteButton).toBeInTheDocument();
      fireEvent.click(deleteButton!);

      expect(mockOnDelete).not.toHaveBeenCalled();

      window.confirm = originalConfirm;
    });
  });

  describe("Search Functionality", () => {
    const mockPrompts: Prompt[] = [
      {
        prompt_id: "user_greeting",
        prompt: "Hello {{name}}, welcome!",
        version: 1,
        variables: ["name"],
        is_default: true,
      },
      {
        prompt_id: "system_summary",
        prompt: "Summarize the following text",
        version: 1,
        variables: [],
        is_default: false,
      },
    ];

    test("filters prompts by prompt ID", () => {
      render(<PromptList {...defaultProps} prompts={mockPrompts} />);

      const searchInput = screen.getByPlaceholderText("Search prompts...");
      fireEvent.change(searchInput, { target: { value: "user" } });

      expect(screen.getByText("user_greeting")).toBeInTheDocument();
      expect(screen.queryByText("system_summary")).not.toBeInTheDocument();
    });

    test("filters prompts by content", () => {
      render(<PromptList {...defaultProps} prompts={mockPrompts} />);

      const searchInput = screen.getByPlaceholderText("Search prompts...");
      fireEvent.change(searchInput, { target: { value: "welcome" } });

      expect(screen.getByText("user_greeting")).toBeInTheDocument();
      expect(screen.queryByText("system_summary")).not.toBeInTheDocument();
    });

    test("search is case insensitive", () => {
      render(<PromptList {...defaultProps} prompts={mockPrompts} />);

      const searchInput = screen.getByPlaceholderText("Search prompts...");
      fireEvent.change(searchInput, { target: { value: "HELLO" } });

      expect(screen.getByText("user_greeting")).toBeInTheDocument();
      expect(screen.queryByText("system_summary")).not.toBeInTheDocument();
    });

    test("clearing search shows all prompts", () => {
      render(<PromptList {...defaultProps} prompts={mockPrompts} />);

      const searchInput = screen.getByPlaceholderText("Search prompts...");

      // Filter first
      fireEvent.change(searchInput, { target: { value: "user" } });
      expect(screen.queryByText("system_summary")).not.toBeInTheDocument();

      // Clear search
      fireEvent.change(searchInput, { target: { value: "" } });
      expect(screen.getByText("user_greeting")).toBeInTheDocument();
      expect(screen.getByText("system_summary")).toBeInTheDocument();
    });
  });
});
