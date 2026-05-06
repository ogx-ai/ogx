"use client";

import React, { useEffect, useState } from "react";
import Link from "next/link";
import { useAuthClient } from "@/hooks/use-auth-client";
import {
  Card,
  CardHeader,
  CardTitle,
  CardDescription,
  CardContent,
} from "@/components/ui/card";
import { Badge } from "@/components/ui/badge";
import { Skeleton } from "@/components/ui/skeleton";
import {
  Box,
  File,
  Database,
  HeartPulse,
  MessageCircle,
  FileText,
  Info,
} from "lucide-react";

interface DashboardStats {
  models: number | null;
  files: number | null;
  vectorStores: number | null;
  health: "healthy" | "unhealthy" | null;
}

function StatCard({
  icon: Icon,
  label,
  value,
  loading,
}: {
  icon: React.ElementType;
  label: string;
  value: string;
  loading: boolean;
}) {
  return (
    <Card>
      <CardHeader>
        <div className="flex items-center gap-3">
          <div className="rounded-md bg-muted p-2">
            <Icon className="h-5 w-5 text-muted-foreground" />
          </div>
          <div>
            <CardDescription>{label}</CardDescription>
            <CardTitle className="text-2xl">
              {loading ? <Skeleton className="h-7 w-12" /> : value}
            </CardTitle>
          </div>
        </div>
      </CardHeader>
    </Card>
  );
}

const quickLinks = [
  {
    title: "Chat Playground",
    description: "Test models with interactive chat",
    href: "/chat-playground",
    icon: MessageCircle,
  },
  {
    title: "Models",
    description: "View and manage registered models",
    href: "/models",
    icon: Box,
  },
  {
    title: "Files",
    description: "Browse uploaded files",
    href: "/logs/files",
    icon: File,
  },
  {
    title: "Vector Stores",
    description: "Manage vector stores and embeddings",
    href: "/logs/vector-stores",
    icon: Database,
  },
  {
    title: "Prompts",
    description: "Manage prompt templates",
    href: "/prompts",
    icon: FileText,
  },
];

export default function Home() {
  const client = useAuthClient();
  const [stats, setStats] = useState<DashboardStats>({
    models: null,
    files: null,
    vectorStores: null,
    health: null,
  });
  const [version, setVersion] = useState<string | null>(null);
  const [loading, setLoading] = useState(true);

  useEffect(() => {
    async function fetchStats() {
      const results: DashboardStats = {
        models: null,
        files: null,
        vectorStores: null,
        health: null,
      };
      let ver: string | null = null;

      await Promise.allSettled([
        (async () => {
          try {
            const models = await client.models.list();
            const modelData = Array.isArray(models)
              ? models
              : (models as Record<string, unknown>).data;
            results.models = Array.isArray(modelData) ? modelData.length : 0;
          } catch {
            /* graceful fallback */
          }
        })(),
        (async () => {
          try {
            const res = await client.files.list();
            const data = (res as Record<string, unknown>).data;
            results.files = Array.isArray(data) ? data.length : 0;
          } catch {
            /* graceful fallback */
          }
        })(),
        (async () => {
          try {
            const res = await client.vectorStores.list();
            const data = (res as Record<string, unknown>).data;
            results.vectorStores = Array.isArray(data) ? data.length : 0;
          } catch {
            /* graceful fallback */
          }
        })(),
        (async () => {
          try {
            const res = await fetch("/api/v1alpha/admin/health");
            results.health = res.ok ? "healthy" : "unhealthy";
          } catch {
            /* graceful fallback */
          }
        })(),
        (async () => {
          try {
            const res = await fetch("/api/v1alpha/admin/version");
            if (res.ok) {
              const data = await res.json();
              ver = data.version ?? JSON.stringify(data);
            }
          } catch {
            /* graceful fallback */
          }
        })(),
      ]);
      setStats(results);
      setVersion(ver);
      setLoading(false);
    }

    fetchStats();
  }, [client]);

  return (
    <div className="space-y-8">
      <div>
        <h1 className="text-3xl font-bold tracking-tight">Dashboard</h1>
        <p className="text-muted-foreground mt-1">
          Overview of your OGX server
        </p>
      </div>

      <div className="grid gap-4 sm:grid-cols-2 lg:grid-cols-4">
        <StatCard
          icon={Box}
          label="Models"
          value={stats.models !== null ? String(stats.models) : "\u2014"}
          loading={loading}
        />
        <StatCard
          icon={File}
          label="Files"
          value={stats.files !== null ? String(stats.files) : "\u2014"}
          loading={loading}
        />
        <StatCard
          icon={Database}
          label="Vector Stores"
          value={
            stats.vectorStores !== null ? String(stats.vectorStores) : "\u2014"
          }
          loading={loading}
        />
        <Card>
          <CardHeader>
            <div className="flex items-center gap-3">
              <div className="rounded-md bg-muted p-2">
                <HeartPulse className="h-5 w-5 text-muted-foreground" />
              </div>
              <div>
                <CardDescription>Server Health</CardDescription>
                <CardTitle className="text-2xl">
                  {loading ? (
                    <Skeleton className="h-7 w-20" />
                  ) : stats.health === "healthy" ? (
                    <Badge variant="default" className="bg-green-600 text-sm">
                      Healthy
                    </Badge>
                  ) : stats.health === "unhealthy" ? (
                    <Badge variant="destructive" className="text-sm">
                      Unhealthy
                    </Badge>
                  ) : (
                    "\u2014"
                  )}
                </CardTitle>
              </div>
            </div>
          </CardHeader>
        </Card>
      </div>

      <div>
        <h2 className="text-xl font-semibold mb-4">Quick Links</h2>
        <div className="grid gap-4 sm:grid-cols-2 lg:grid-cols-3">
          {quickLinks.map(link => (
            <Link key={link.href} href={link.href}>
              <Card className="h-full transition-colors hover:bg-muted/50 cursor-pointer">
                <CardHeader>
                  <div className="flex items-center gap-3">
                    <link.icon className="h-5 w-5 text-muted-foreground" />
                    <CardTitle className="text-base">{link.title}</CardTitle>
                  </div>
                  <CardDescription>{link.description}</CardDescription>
                </CardHeader>
              </Card>
            </Link>
          ))}
        </div>
      </div>

      <div>
        <h2 className="text-xl font-semibold mb-4">Server Info</h2>
        <Card>
          <CardContent className="flex items-center gap-3 text-sm text-muted-foreground">
            <Info className="h-4 w-4 shrink-0" />
            <span>
              Version:{" "}
              {loading ? (
                <Skeleton className="inline-block h-4 w-24 align-middle" />
              ) : (
                (version ?? "\u2014")
              )}
            </span>
          </CardContent>
        </Card>
      </div>
    </div>
  );
}
