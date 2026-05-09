"use client";

import { ChatCompletionsTable } from "@/components/chat-completions/chat-completions-table";

export default function ChatCompletionsPage() {
  return <ChatCompletionsTable paginationOptions={{ limit: 20 }} />;
}
