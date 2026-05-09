import React from "react";
import { Card, CardContent, CardHeader, CardTitle } from "@/components/ui/card";
import { Skeleton } from "@/components/ui/skeleton";

export function DetailLoadingView() {
  return (
    <>
      <Skeleton className="h-8 w-3/4 mb-6" /> {/* Title Skeleton */}
      <div className="flex flex-col md:flex-row gap-6">
        <div className="flex-1 min-w-0 space-y-6">
          {[...Array(2)].map((_, i) => (
            <Card key={`main-skeleton-card-${i}`}>
              <CardHeader>
                <CardTitle>
                  <Skeleton className="h-6 w-1/2" />
                </CardTitle>
              </CardHeader>
              <CardContent className="space-y-2">
                <Skeleton className="h-4 w-full" />
                <Skeleton className="h-4 w-full" />
                <Skeleton className="h-4 w-3/4" />
              </CardContent>
            </Card>
          ))}
        </div>
        <div className="w-80 flex-shrink-0">
          <div className="p-4 border rounded-lg shadow-sm bg-white space-y-3">
            <Skeleton className="h-6 w-1/3 mb-3" />{" "}
            {/* Properties Title Skeleton */}
            {[...Array(5)].map((_, i) => (
              <div key={`prop-skeleton-${i}`} className="space-y-1">
                <Skeleton className="h-4 w-1/4" />
                <Skeleton className="h-4 w-1/2" />
              </div>
            ))}
          </div>
        </div>
      </div>
    </>
  );
}

export function DetailErrorView({
  title,
  id,
  error,
}: {
  title: string;
  id: string;
  error: Error;
}) {
  return (
    <>
      <h1 className="text-2xl font-bold mb-6">{title}</h1>
      <p>
        Error loading details for ID {id}: {error.message}
      </p>
    </>
  );
}

export function DetailNotFoundView({
  title,
  id,
}: {
  title: string;
  id: string;
}) {
  return (
    <>
      <h1 className="text-2xl font-bold mb-6">{title}</h1>
      <p>No details found for ID: {id}.</p>
    </>
  );
}

export interface PropertyItemProps {
  label: string;
  value: React.ReactNode;
  className?: string;
  hasBorder?: boolean;
}

export function PropertyItem({
  label,
  value,
  className = "",
  hasBorder = false,
}: PropertyItemProps) {
  return (
    <li
      className={`${hasBorder ? "pt-1 mt-1 border-t border-gray-200" : ""} ${className}`}
    >
      <strong>{label}:</strong>{" "}
      {typeof value === "string" || typeof value === "number" ? (
        <span className="text-gray-900 dark:text-gray-100 font-medium">
          {value}
        </span>
      ) : (
        value
      )}
    </li>
  );
}

export interface PropertiesCardProps {
  children: React.ReactNode;
}

export function PropertiesCard({ children }: PropertiesCardProps) {
  return (
    <Card>
      <CardHeader>
        <CardTitle>Properties</CardTitle>
      </CardHeader>
      <CardContent>
        <ul className="space-y-2 text-sm text-gray-600 dark:text-gray-400">
          {children}
        </ul>
      </CardContent>
    </Card>
  );
}

export interface DetailLayoutProps {
  title: string;
  mainContent: React.ReactNode;
  sidebar: React.ReactNode;
}

export function DetailLayout({
  title,
  mainContent,
  sidebar,
}: DetailLayoutProps) {
  return (
    <>
      <h1 className="text-2xl font-bold mb-6">{title}</h1>
      <div className="flex flex-col md:flex-row gap-6">
        <div className="flex-1 min-w-0 space-y-6 overflow-x-hidden">
          {mainContent}
        </div>
        <div className="w-80 flex-shrink-0">{sidebar}</div>
      </div>
    </>
  );
}
