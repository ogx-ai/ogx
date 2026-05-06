"use client";

import React from "react";
import { usePathname, useParams } from "next/navigation";
import {
  PageBreadcrumb,
  BreadcrumbSegment,
} from "@/components/layout/page-breadcrumb";
import { truncateText } from "@/lib/truncate-text";

interface LogsLayoutProps {
  children: React.ReactNode;
  sectionLabel: string;
  basePath: string;
}

export default function LogsLayout({
  children,
  sectionLabel,
  basePath,
}: LogsLayoutProps) {
  const pathname = usePathname();
  const params = useParams();

  let segments: BreadcrumbSegment[] = [];

  if (pathname === basePath) {
    segments = [{ label: sectionLabel }];
  }

  const idParam = params?.id;
  if (idParam && typeof idParam === "string") {
    segments = [
      { label: sectionLabel, href: basePath },
      { label: `Details (${truncateText(idParam, 20)})` },
    ];
  }

  return (
    <div className="container mx-auto p-4 h-[calc(100vh-64px)] flex flex-col">
      {segments.length > 0 && (
        <PageBreadcrumb segments={segments} className="mb-4" />
      )}
      <div className="flex-1 min-h-0 flex flex-col">{children}</div>
    </div>
  );
}
