import React from "react";
import { render, waitFor } from "@testing-library/react";
import "@testing-library/jest-dom";
import { FileDetail } from "./file-detail";

const mockReplace = jest.fn();
const mockRouter = { replace: mockReplace };
const mockClient = {
  files: {
    retrieve: jest.fn(),
  },
  vectorStores: {
    list: jest.fn(),
    files: {
      list: jest.fn(),
    },
  },
};

jest.mock("@/hooks/use-auth-client", () => ({
  useAuthClient: () => mockClient,
}));

jest.mock("next/navigation", () => ({
  useParams: () => ({ id: "file-orphaned" }),
  useRouter: () => mockRouter,
}));

describe("FileDetail", () => {
  beforeEach(() => {
    jest.clearAllMocks();
  });

  it("does not log a Files API error when vector-store fallback succeeds", async () => {
    mockClient.files.retrieve.mockRejectedValue(new Error("404"));
    mockClient.vectorStores.list.mockResolvedValue({
      data: [{ id: "vs_123" }],
    });
    mockClient.vectorStores.files.list.mockResolvedValue({
      data: [{ id: "file-orphaned" }],
    });
    const consoleError = jest.spyOn(console, "error").mockImplementation();

    render(<FileDetail />);

    await waitFor(() => {
      expect(mockReplace).toHaveBeenCalledWith(
        "/logs/vector-stores/vs_123/files/file-orphaned"
      );
    });
    expect(consoleError).not.toHaveBeenCalled();
    consoleError.mockRestore();
  });
});
