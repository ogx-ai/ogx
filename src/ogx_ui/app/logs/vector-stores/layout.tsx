"use client";

import { useParams, usePathname } from "next/navigation";
import {
  PageBreadcrumb,
  BreadcrumbSegment,
} from "@/components/layout/page-breadcrumb";

export default function VectorStoreDetailLayout({
  children,
}: {
  children: React.ReactNode;
}) {
  const params = useParams();
  const pathname = usePathname();
  const vectorStoreId = params.id as string;

  const breadcrumbSegments: BreadcrumbSegment[] = [
    { label: "Vector Stores", href: "/logs/vector-stores" },
    { label: `Details (${vectorStoreId})` },
  ];

  const isBaseDetailPage = pathname === `/logs/vector-stores/${vectorStoreId}`;

  return (
    <div className="space-y-4">
      {isBaseDetailPage && <PageBreadcrumb segments={breadcrumbSegments} />}
      {children}
    </div>
  );
}
