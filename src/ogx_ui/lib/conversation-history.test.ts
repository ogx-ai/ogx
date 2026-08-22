import {
  getConversationHistory,
  getConversation,
  addConversation,
  removeConversation,
  updateConversation,
  type ConversationHistoryEntry,
} from "./conversation-history";

const STORAGE_KEY = "llama-stack-conversations";

describe("conversation-history", () => {
  beforeEach(() => {
    localStorage.clear();
  });

  const makeEntry = (
    overrides: Partial<ConversationHistoryEntry> = {}
  ): ConversationHistoryEntry => ({
    id: "conv_1",
    createdAt: 1000,
    firstMessage: "Hello",
    ...overrides,
  });

  describe("addConversation", () => {
    test("adds a conversation to localStorage", () => {
      addConversation(makeEntry());
      const history = getConversationHistory();
      expect(history).toHaveLength(1);
      expect(history[0].id).toBe("conv_1");
    });

    test("does not add duplicate conversations", () => {
      addConversation(makeEntry());
      addConversation(makeEntry());
      expect(getConversationHistory()).toHaveLength(1);
    });

    test("dispatches conversations-updated event", () => {
      const handler = jest.fn();
      window.addEventListener("conversations-updated", handler);
      addConversation(makeEntry());
      expect(handler).toHaveBeenCalledTimes(1);
      window.removeEventListener("conversations-updated", handler);
    });
  });

  describe("removeConversation", () => {
    test("removes a conversation by id", () => {
      addConversation(makeEntry({ id: "conv_1" }));
      addConversation(makeEntry({ id: "conv_2", createdAt: 2000 }));
      removeConversation("conv_1");
      const history = getConversationHistory();
      expect(history).toHaveLength(1);
      expect(history[0].id).toBe("conv_2");
    });

    test("no-ops for non-existent id", () => {
      addConversation(makeEntry());
      removeConversation("conv_nonexistent");
      expect(getConversationHistory()).toHaveLength(1);
    });

    test("dispatches conversations-updated event", () => {
      addConversation(makeEntry());
      const handler = jest.fn();
      window.addEventListener("conversations-updated", handler);
      removeConversation("conv_1");
      expect(handler).toHaveBeenCalledTimes(1);
      window.removeEventListener("conversations-updated", handler);
    });
  });

  describe("getConversation", () => {
    test("returns conversation by id", () => {
      addConversation(makeEntry({ id: "conv_1", firstMessage: "Hi" }));
      const conv = getConversation("conv_1");
      expect(conv).toBeDefined();
      expect(conv!.firstMessage).toBe("Hi");
    });

    test("returns undefined for non-existent id", () => {
      expect(getConversation("missing")).toBeUndefined();
    });
  });

  describe("updateConversation", () => {
    test("updates fields on an existing conversation", () => {
      addConversation(makeEntry({ id: "conv_1", model: "gpt-4" }));
      updateConversation("conv_1", { model: "gpt-4o" });
      expect(getConversation("conv_1")!.model).toBe("gpt-4o");
    });

    test("no-ops for non-existent id", () => {
      updateConversation("missing", { model: "gpt-4" });
      expect(getConversationHistory()).toHaveLength(0);
    });

    test("persists fileIdMap", () => {
      addConversation(makeEntry({ id: "conv_1" }));
      const fileIdMap = {
        "0a643196-4e67-414d-a3e1-a5f3d192516d":
          "file-80f374998546452597827eaf338a9c3c",
      };
      updateConversation("conv_1", { fileIdMap });
      const conv = getConversation("conv_1");
      expect(conv!.fileIdMap).toEqual(fileIdMap);
    });

    test("merging fileIdMap preserves existing entries", () => {
      const map1 = { "uuid-a": "file-aaa" };
      const map2 = { "uuid-b": "file-bbb" };
      addConversation(makeEntry({ id: "conv_1", fileIdMap: map1 }));
      const existing = getConversation("conv_1")?.fileIdMap || {};
      updateConversation("conv_1", {
        fileIdMap: { ...existing, ...map2 },
      });
      const conv = getConversation("conv_1");
      expect(conv!.fileIdMap).toEqual({
        "uuid-a": "file-aaa",
        "uuid-b": "file-bbb",
      });
    });
  });

  describe("getConversationHistory", () => {
    test("returns conversations sorted by createdAt descending", () => {
      addConversation(makeEntry({ id: "old", createdAt: 1000 }));
      addConversation(makeEntry({ id: "new", createdAt: 3000 }));
      addConversation(makeEntry({ id: "mid", createdAt: 2000 }));
      const history = getConversationHistory();
      expect(history.map(h => h.id)).toEqual(["new", "mid", "old"]);
    });

    test("returns empty array when localStorage is empty", () => {
      expect(getConversationHistory()).toEqual([]);
    });

    test("returns empty array when localStorage has invalid JSON", () => {
      localStorage.setItem(STORAGE_KEY, "not-json");
      expect(getConversationHistory()).toEqual([]);
    });
  });
});
