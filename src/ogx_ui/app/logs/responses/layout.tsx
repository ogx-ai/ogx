"use client";

import React from "react";
import LogsLayout from "@/components/layout/logs-layout";

export default function ResponsesLayout({
  children,
}: {
  children: React.ReactNode;
}) {
  return (
    <LogsLayout sectionLabel="Responses" basePath="/logs/responses">
      {children}
    </LogsLayout>
  );
}
