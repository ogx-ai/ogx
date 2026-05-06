"use client";

import { ResponsesTable } from "@/components/responses/responses-table";

export default function ResponsesPage() {
  return <ResponsesTable paginationOptions={{ limit: 20 }} />;
}
