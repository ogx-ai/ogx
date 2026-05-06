const STORAGE_KEY = "llama-stack-conversations";
const MAX_ENTRIES = 50;

export interface SavedToolsConfig {
  webSearch: boolean;
  fileSearch: {
    enabled: boolean;
    vectorStoreIds: string[];
  };
  mcp: {
    enabled: boolean;
    serverLabel: string;
    serverUrl: string;
  };
}

export interface ConversationHistoryEntry {
  id: string;
  createdAt: number;
  firstMessage?: string;
  model?: string;
  systemInstructions?: string;
  toolsConfig?: SavedToolsConfig;
}

export function getConversationHistory(): ConversationHistoryEntry[] {
  if (typeof window === "undefined") return [];
  try {
    const raw = localStorage.getItem(STORAGE_KEY);
    if (!raw) return [];
    const entries: ConversationHistoryEntry[] = JSON.parse(raw);
    return entries.sort((a, b) => b.createdAt - a.createdAt);
  } catch {
    return [];
  }
}

export function getConversation(
  id: string
): ConversationHistoryEntry | undefined {
  return getConversationHistory().find(e => e.id === id);
}

export function addConversation(entry: ConversationHistoryEntry): void {
  if (typeof window === "undefined") return;
  try {
    const existing = getConversationHistory();
    if (existing.some(e => e.id === entry.id)) return;
    const updated = [entry, ...existing].slice(0, MAX_ENTRIES);
    localStorage.setItem(STORAGE_KEY, JSON.stringify(updated));
    window.dispatchEvent(new Event("conversations-updated"));
  } catch {
    // localStorage may be unavailable
  }
}

export function updateConversation(
  id: string,
  updates: Partial<ConversationHistoryEntry>
): void {
  if (typeof window === "undefined") return;
  try {
    const existing = getConversationHistory();
    const idx = existing.findIndex(e => e.id === id);
    if (idx === -1) return;
    existing[idx] = { ...existing[idx], ...updates };
    localStorage.setItem(STORAGE_KEY, JSON.stringify(existing));
  } catch {
    // localStorage may be unavailable
  }
}
