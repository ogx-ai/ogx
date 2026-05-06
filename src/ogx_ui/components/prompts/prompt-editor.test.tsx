import React from "react";
import { render, screen, fireEvent } from "@testing-library/react";
import "@testing-library/jest-dom";
import { PromptEditor } from "./prompt-editor";
import type { Prompt } from "./types";

describe("PromptEditor", () => {
  const mockOnSave = jest.fn();
  const mockOnCancel = jest.fn();
  const mockOnDelete = jest.fn();

  const defaultProps = {
    onSave: mockOnSave,
    onCancel: mockOnCancel,
    onDelete: mockOnDelete,
  };

  beforeEach(() => {
    jest.clearAllMocks();
  });

  describe("Create Mode", () => {
    test("renders create form correctly", () => {
      render(<PromptEditor {...defaultProps} />);

      expect(screen.getByLabelText("Prompt Content *")).toBeInTheDocument();
      expect(screen.getByText("Variables")).toBeInTheDocument();
      expect(screen.getByText("Preview")).toBeInTheDocument();
      expect(screen.getByText("Create Prompt")).toBeInTheDocument();
      expect(screen.getByText("Cancel")).toBeInTheDocument();
    });

    test("shows preview placeholder when no content", () => {
      render(<PromptEditor {...defaultProps} />);

      expect(
        screen.getByText("Enter content to preview the compiled prompt")
      ).toBeInTheDocument();
    });

    test("submits form with correct data", () => {
      render(<PromptEditor {...defaultProps} />);

      const promptInput = screen.getByLabelText("Prompt Content *");
      fireEvent.change(promptInput, {
        target: { value: "Hello {{name}}, welcome!" },
      });

      fireEvent.click(screen.getByText("Create Prompt"));

      expect(mockOnSave).toHaveBeenCalledWith({
        prompt: "Hello {{name}}, welcome!",
        variables: [],
      });
    });

    test("prevents submission with empty prompt", () => {
      render(<PromptEditor {...defaultProps} />);

      fireEvent.click(screen.getByText("Create Prompt"));

      expect(mockOnSave).not.toHaveBeenCalled();
    });
  });

  describe("Edit Mode", () => {
    const mockPrompt: Prompt = {
      prompt_id: "prompt_123",
      prompt: "Hello {{name}}, how is {{weather}}?",
      version: 1,
      variables: ["name", "weather"],
      is_default: true,
    };

    test("renders edit form with existing data", () => {
      render(<PromptEditor {...defaultProps} prompt={mockPrompt} />);

      expect(
        screen.getByDisplayValue("Hello {{name}}, how is {{weather}}?")
      ).toBeInTheDocument();
      expect(screen.getAllByText("name")).toHaveLength(2); // One in variables, one in preview
      expect(screen.getAllByText("weather")).toHaveLength(2); // One in variables, one in preview
      expect(screen.getByText("Update Prompt")).toBeInTheDocument();
      expect(screen.getByText("Delete Prompt")).toBeInTheDocument();
    });

    test("submits updated data correctly", () => {
      render(<PromptEditor {...defaultProps} prompt={mockPrompt} />);

      const promptInput = screen.getByLabelText("Prompt Content *");
      fireEvent.change(promptInput, {
        target: { value: "Updated: Hello {{name}}!" },
      });

      fireEvent.click(screen.getByText("Update Prompt"));

      expect(mockOnSave).toHaveBeenCalledWith({
        prompt: "Updated: Hello {{name}}!",
        variables: ["name", "weather"],
      });
    });
  });

  describe("Variables Management", () => {
    test("adds new variable", () => {
      render(<PromptEditor {...defaultProps} />);

      const variableInput = screen.getByPlaceholderText(
        "Add variable name (e.g. user_name, topic)"
      );
      fireEvent.change(variableInput, { target: { value: "testVar" } });
      fireEvent.click(screen.getByText("Add"));

      expect(screen.getByText("testVar")).toBeInTheDocument();
    });

    test("prevents adding duplicate variables", () => {
      render(<PromptEditor {...defaultProps} />);

      const variableInput = screen.getByPlaceholderText(
        "Add variable name (e.g. user_name, topic)"
      );

      // Add first variable
      fireEvent.change(variableInput, { target: { value: "test" } });
      fireEvent.click(screen.getByText("Add"));

      // Try to add same variable again
      fireEvent.change(variableInput, { target: { value: "test" } });

      // Button should be disabled
      expect(screen.getByText("Add")).toBeDisabled();
    });

    test("removes variable", () => {
      const mockPrompt: Prompt = {
        prompt_id: "prompt_123",
        prompt: "Hello {{name}}",
        version: 1,
        variables: ["name", "location"],
        is_default: true,
      };

      render(<PromptEditor {...defaultProps} prompt={mockPrompt} />);

      // Check that both variables are present initially
      expect(screen.getAllByText("name").length).toBeGreaterThan(0);
      expect(screen.getAllByText("location").length).toBeGreaterThan(0);

      // Remove the location variable by clicking the X button with the specific title
      const removeLocationButton = screen.getByTitle(
        "Remove location variable"
      );
      fireEvent.click(removeLocationButton);

      // Name should still be there, location should be gone from the variables section
      expect(screen.getAllByText("name").length).toBeGreaterThan(0);
      expect(
        screen.queryByTitle("Remove location variable")
      ).not.toBeInTheDocument();
    });

    test("adds variable on Enter key", () => {
      render(<PromptEditor {...defaultProps} />);

      const variableInput = screen.getByPlaceholderText(
        "Add variable name (e.g. user_name, topic)"
      );
      fireEvent.change(variableInput, { target: { value: "enterVar" } });

      // Simulate Enter key press
      fireEvent.keyPress(variableInput, {
        key: "Enter",
        code: "Enter",
        charCode: 13,
        preventDefault: jest.fn(),
      });

      // Check if the variable was added by looking for the badge
      expect(screen.getAllByText("enterVar").length).toBeGreaterThan(0);
    });
  });

  describe("Preview Functionality", () => {
    test("shows live preview with variables", () => {
      render(<PromptEditor {...defaultProps} />);

      // Add prompt content
      const promptInput = screen.getByLabelText("Prompt Content *");
      fireEvent.change(promptInput, {
        target: { value: "Hello {{name}}, welcome to {{place}}!" },
      });

      // Add variables
      const variableInput = screen.getByPlaceholderText(
        "Add variable name (e.g. user_name, topic)"
      );
      fireEvent.change(variableInput, { target: { value: "name" } });
      fireEvent.click(screen.getByText("Add"));

      fireEvent.change(variableInput, { target: { value: "place" } });
      fireEvent.click(screen.getByText("Add"));

      // Check that preview area shows the content
      expect(screen.getByText("Compiled Prompt")).toBeInTheDocument();
    });

    test("shows variable value inputs in preview", () => {
      const mockPrompt: Prompt = {
        prompt_id: "prompt_123",
        prompt: "Hello {{name}}",
        version: 1,
        variables: ["name"],
        is_default: true,
      };

      render(<PromptEditor {...defaultProps} prompt={mockPrompt} />);

      expect(screen.getByText("Variable Values")).toBeInTheDocument();
      expect(
        screen.getByPlaceholderText("Enter value for name")
      ).toBeInTheDocument();
    });

    test("shows color legend for variable states", () => {
      render(<PromptEditor {...defaultProps} />);

      // Add content to show preview
      const promptInput = screen.getByLabelText("Prompt Content *");
      fireEvent.change(promptInput, {
        target: { value: "Hello {{name}}" },
      });

      expect(screen.getByText("Used")).toBeInTheDocument();
      expect(screen.getByText("Unused")).toBeInTheDocument();
      expect(screen.getByText("Undefined")).toBeInTheDocument();
    });
  });

  describe("Error Handling", () => {
    test("displays error message", () => {
      const errorMessage = "Prompt contains undeclared variables";
      render(<PromptEditor {...defaultProps} error={errorMessage} />);

      expect(screen.getByText(errorMessage)).toBeInTheDocument();
    });
  });

  describe("Delete Functionality", () => {
    const mockPrompt: Prompt = {
      prompt_id: "prompt_123",
      prompt: "Hello {{name}}",
      version: 1,
      variables: ["name"],
      is_default: true,
    };

    test("shows delete button in edit mode", () => {
      render(<PromptEditor {...defaultProps} prompt={mockPrompt} />);

      expect(screen.getByText("Delete Prompt")).toBeInTheDocument();
    });

    test("hides delete button in create mode", () => {
      render(<PromptEditor {...defaultProps} />);

      expect(screen.queryByText("Delete Prompt")).not.toBeInTheDocument();
    });

    test("calls onDelete with confirmation", () => {
      const originalConfirm = window.confirm;
      window.confirm = jest.fn(() => true);

      render(<PromptEditor {...defaultProps} prompt={mockPrompt} />);

      fireEvent.click(screen.getByText("Delete Prompt"));

      expect(window.confirm).toHaveBeenCalledWith(
        "Are you sure you want to delete this prompt? This action cannot be undone."
      );
      expect(mockOnDelete).toHaveBeenCalledWith("prompt_123");

      window.confirm = originalConfirm;
    });

    test("does not delete when confirmation is cancelled", () => {
      const originalConfirm = window.confirm;
      window.confirm = jest.fn(() => false);

      render(<PromptEditor {...defaultProps} prompt={mockPrompt} />);

      fireEvent.click(screen.getByText("Delete Prompt"));

      expect(mockOnDelete).not.toHaveBeenCalled();

      window.confirm = originalConfirm;
    });
  });

  describe("Cancel Functionality", () => {
    test("calls onCancel when cancel button is clicked", () => {
      render(<PromptEditor {...defaultProps} />);

      fireEvent.click(screen.getByText("Cancel"));

      expect(mockOnCancel).toHaveBeenCalled();
    });
  });
});
