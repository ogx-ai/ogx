"use client";

import React from "react";
import LogsLayout from "@/components/layout/logs-layout";

export default function ChatCompletionsLayout({
  children,
}: {
  children: React.ReactNode;
}) {
  return (
    <LogsLayout
      sectionLabel="Chat Completions"
      basePath="/logs/chat-completions"
    >
      {children}
    </LogsLayout>
  );
}
