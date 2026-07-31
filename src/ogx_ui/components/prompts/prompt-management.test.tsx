import React from "react";
import { render, screen, fireEvent, waitFor } from "@testing-library/react";
import "@testing-library/jest-dom";
import { PromptManagement } from "./prompt-management";
import type { Prompt } from "./types";

// Mock the auth client
const mockPromptsClient = {
  list: jest.fn(),
  create: jest.fn(),
  update: jest.fn(),
  delete: jest.fn(),
};

jest.mock("@/hooks/use-auth-client", () => ({
  useAuthClient: () => ({
    prompts: mockPromptsClient,
  }),
}));

describe("PromptManagement", () => {
  beforeEach(() => {
    jest.clearAllMocks();
  });

  describe("Loading State", () => {
    test("renders loading state initially", () => {
      mockPromptsClient.list.mockReturnValue(new Promise(() => {})); // Never resolves
      render(<PromptManagement />);

      expect(screen.getByText("Loading prompts...")).toBeInTheDocument();
      expect(screen.getByText("Prompts")).toBeInTheDocument();
    });
  });

  describe("Empty State", () => {
    test("renders empty state when no prompts", async () => {
      mockPromptsClient.list.mockResolvedValue([]);
      render(<PromptManagement />);

      await waitFor(() => {
        expect(screen.getByText("No prompts found.")).toBeInTheDocument();
      });

      expect(screen.getByText("Create Your First Prompt")).toBeInTheDocument();
    });

    test("opens modal when clicking 'Create Your First Prompt'", async () => {
      mockPromptsClient.list.mockResolvedValue([]);
      render(<PromptManagement />);

      await waitFor(() => {
        expect(
          screen.getByText("Create Your First Prompt")
        ).toBeInTheDocument();
      });

      fireEvent.click(screen.getByText("Create Your First Prompt"));

      expect(screen.getByText("Create New Prompt")).toBeInTheDocument();
    });
  });

  describe("Error State", () => {
    test("renders error state when API fails", async () => {
      const error = new Error("API not found");
      mockPromptsClient.list.mockRejectedValue(error);
      render(<PromptManagement />);

      await waitFor(() => {
        expect(screen.getByText(/Error:/)).toBeInTheDocument();
      });
    });

    test("renders specific error for 404", async () => {
      const error = new Error("404 Not found");
      mockPromptsClient.list.mockRejectedValue(error);
      render(<PromptManagement />);

      await waitFor(() => {
        expect(
          screen.getByText(/Prompts API endpoint not found/)
        ).toBeInTheDocument();
      });
    });
  });

  describe("Prompts List", () => {
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
        prompt: "Summarize this {{text}}",
        version: 2,
        variables: ["text"],
        is_default: false,
      },
    ];

    test("renders prompts list correctly", async () => {
      mockPromptsClient.list.mockResolvedValue(mockPrompts);
      render(<PromptManagement />);

      await waitFor(() => {
        expect(screen.getByText("prompt_123")).toBeInTheDocument();
      });

      expect(screen.getByText("prompt_456")).toBeInTheDocument();
      expect(
        screen.getByText("Hello {{name}}, how are you?")
      ).toBeInTheDocument();
      expect(screen.getByText("Summarize this {{text}}")).toBeInTheDocument();
    });

    test("opens modal when clicking 'New Prompt' button", async () => {
      mockPromptsClient.list.mockResolvedValue(mockPrompts);
      render(<PromptManagement />);

      await waitFor(() => {
        expect(screen.getByText("prompt_123")).toBeInTheDocument();
      });

      fireEvent.click(screen.getByText("New Prompt"));

      expect(screen.getByText("Create New Prompt")).toBeInTheDocument();
    });
  });

  describe("Modal Operations", () => {
    const mockPrompts: Prompt[] = [
      {
        prompt_id: "prompt_123",
        prompt: "Hello {{name}}",
        version: 1,
        variables: ["name"],
        is_default: true,
      },
    ];

    test("closes modal when clicking cancel", async () => {
      mockPromptsClient.list.mockResolvedValue(mockPrompts);
      render(<PromptManagement />);

      await waitFor(() => {
        expect(screen.getByText("prompt_123")).toBeInTheDocument();
      });

      // Open modal
      fireEvent.click(screen.getByText("New Prompt"));
      expect(screen.getByText("Create New Prompt")).toBeInTheDocument();

      // Close modal
      fireEvent.click(screen.getByText("Cancel"));
      expect(screen.queryByText("Create New Prompt")).not.toBeInTheDocument();
    });

    test("creates new prompt successfully", async () => {
      const newPrompt: Prompt = {
        prompt_id: "prompt_new",
        prompt: "New prompt content",
        version: 1,
        variables: [],
        is_default: false,
      };

      mockPromptsClient.list.mockResolvedValue(mockPrompts);
      mockPromptsClient.create.mockResolvedValue(newPrompt);
      render(<PromptManagement />);

      await waitFor(() => {
        expect(screen.getByText("prompt_123")).toBeInTheDocument();
      });

      // Open modal
      fireEvent.click(screen.getByText("New Prompt"));

      // Fill form
      const promptInput = screen.getByLabelText("Prompt Content *");
      fireEvent.change(promptInput, {
        target: { value: "New prompt content" },
      });

      // Submit form
      fireEvent.click(screen.getByText("Create Prompt"));

      await waitFor(() => {
        expect(mockPromptsClient.create).toHaveBeenCalledWith({
          prompt: "New prompt content",
          variables: [],
        });
      });
    });

    test("handles create error gracefully", async () => {
      const error = {
        detail: {
          errors: [{ msg: "Prompt contains undeclared variables: ['test']" }],
        },
      };

      mockPromptsClient.list.mockResolvedValue(mockPrompts);
      mockPromptsClient.create.mockRejectedValue(error);
      render(<PromptManagement />);

      await waitFor(() => {
        expect(screen.getByText("prompt_123")).toBeInTheDocument();
      });

      // Open modal
      fireEvent.click(screen.getByText("New Prompt"));

      // Fill form
      const promptInput = screen.getByLabelText("Prompt Content *");
      fireEvent.change(promptInput, { target: { value: "Hello {{test}}" } });

      // Submit form
      fireEvent.click(screen.getByText("Create Prompt"));

      await waitFor(() => {
        expect(
          screen.getByText("Prompt contains undeclared variables: ['test']")
        ).toBeInTheDocument();
      });
    });

    test("updates existing prompt successfully", async () => {
      const updatedPrompt: Prompt = {
        ...mockPrompts[0],
        prompt: "Updated content",
      };

      mockPromptsClient.list.mockResolvedValue(mockPrompts);
      mockPromptsClient.update.mockResolvedValue(updatedPrompt);
      const { container } = render(<PromptManagement />);

      await waitFor(() => {
        expect(screen.getByText("prompt_123")).toBeInTheDocument();
      });

      // Click edit button (first button in the action cell of the first row)
      const actionCells = container.querySelectorAll("td:last-child");
      const firstActionCell = actionCells[0];
      const editButton = firstActionCell?.querySelector("button");

      expect(editButton).toBeInTheDocument();
      fireEvent.click(editButton!);

      expect(screen.getByText("Edit Prompt")).toBeInTheDocument();

      // Update content
      const promptInput = screen.getByLabelText("Prompt Content *");
      fireEvent.change(promptInput, { target: { value: "Updated content" } });

      // Submit form
      fireEvent.click(screen.getByText("Update Prompt"));

      await waitFor(() => {
        expect(mockPromptsClient.update).toHaveBeenCalledWith("prompt_123", {
          prompt: "Updated content",
          variables: ["name"],
          version: 1,
          set_as_default: true,
        });
      });
    });

    test("deletes prompt successfully", async () => {
      mockPromptsClient.list.mockResolvedValue(mockPrompts);
      mockPromptsClient.delete.mockResolvedValue(undefined);

      // Mock window.confirm
      const originalConfirm = window.confirm;
      window.confirm = jest.fn(() => true);

      const { container } = render(<PromptManagement />);

      await waitFor(() => {
        expect(screen.getByText("prompt_123")).toBeInTheDocument();
      });

      // Click delete button (second button in the action cell of the first row)
      const actionCells = container.querySelectorAll("td:last-child");
      const firstActionCell = actionCells[0];
      const buttons = firstActionCell?.querySelectorAll("button");
      const deleteButton = buttons?.[1]; // Second button should be delete

      expect(deleteButton).toBeInTheDocument();
      fireEvent.click(deleteButton!);

      await waitFor(() => {
        expect(mockPromptsClient.delete).toHaveBeenCalledWith("prompt_123");
      });

      // Restore window.confirm
      window.confirm = originalConfirm;
    });
  });
});
