"use client";

import React, { useState, useEffect, useCallback } from "react";
import { useAuthClient } from "@/hooks/use-auth-client";
import { Badge } from "@/components/ui/badge";
import {
  Card,
  CardContent,
  CardDescription,
  CardHeader,
  CardTitle,
} from "@/components/ui/card";
import { Tabs, TabsContent, TabsList, TabsTrigger } from "@/components/ui/tabs";
import {
  Table,
  TableBody,
  TableCell,
  TableHead,
  TableHeader,
  TableRow,
} from "@/components/ui/table";
import { Skeleton } from "@/components/ui/skeleton";
import { Input } from "@/components/ui/input";
import { Collapsible, CollapsibleTrigger } from "@/components/ui/collapsible";
import { Button } from "@/components/ui/button";
import { Search, ChevronDown, ChevronRight, RefreshCw } from "lucide-react";

interface ProviderInfo {
  api: string;
  provider_id: string;
  provider_type: string;
  config?: Record<string, unknown>;
}

interface RouteInfo {
  route: string;
  method: string;
  provider_type: string;
}

interface HealthStatus {
  status: string;
}

interface VersionInfo {
  version: string;
}

type FetchStatus = "loading" | "success" | "error";

export function SystemOverview() {
  const client = useAuthClient();

  const [healthStatus, setHealthStatus] = useState<HealthStatus | null>(null);
  const [healthFetchStatus, setHealthFetchStatus] =
    useState<FetchStatus>("loading");
  const [healthError, setHealthError] = useState<string | null>(null);

  const [versionInfo, setVersionInfo] = useState<VersionInfo | null>(null);
  const [versionFetchStatus, setVersionFetchStatus] =
    useState<FetchStatus>("loading");

  const [providers, setProviders] = useState<ProviderInfo[]>([]);
  const [providerFetchStatus, setProviderFetchStatus] =
    useState<FetchStatus>("loading");
  const [providerError, setProviderError] = useState<string | null>(null);

  const [routes, setRoutes] = useState<RouteInfo[]>([]);
  const [routeFetchStatus, setRouteFetchStatus] =
    useState<FetchStatus>("loading");
  const [routeError, setRouteError] = useState<string | null>(null);

  const [providerSearch, setProviderSearch] = useState("");
  const [providerApiFilter, setProviderApiFilter] = useState<string>("all");
  const [expandedProviders, setExpandedProviders] = useState<Set<string>>(
    new Set()
  );

  const [routeSearch, setRouteSearch] = useState("");

  const fetchHealth = useCallback(async () => {
    setHealthFetchStatus("loading");
    try {
      const response = await fetch("/api/v1alpha/admin/health");
      if (!response.ok) throw new Error(`HTTP ${response.status}`);
      const data = await response.json();
      setHealthStatus(data);
      setHealthFetchStatus("success");
      setHealthError(null);
    } catch (err) {
      console.error("Failed to fetch health:", err);
      setHealthStatus(null);
      setHealthFetchStatus("error");
      setHealthError(err instanceof Error ? err.message : "Unknown error");
    }
  }, []);

  const fetchVersion = useCallback(async () => {
    setVersionFetchStatus("loading");
    try {
      const response = await fetch("/api/v1alpha/admin/version");
      if (!response.ok) throw new Error(`HTTP ${response.status}`);
      const data = await response.json();
      setVersionInfo(data);
      setVersionFetchStatus("success");
    } catch (err) {
      console.error("Failed to fetch version:", err);
      setVersionInfo(null);
      setVersionFetchStatus("error");
    }
  }, []);

  const fetchProviders = useCallback(async () => {
    setProviderFetchStatus("loading");
    try {
      const providersResponse = await client.providers.list();
      const providersList = Array.isArray(providersResponse)
        ? providersResponse
        : ((providersResponse as unknown as { data: ProviderInfo[] }).data ??
          []);
      setProviders(providersList as ProviderInfo[]);
      setProviderFetchStatus("success");
      setProviderError(null);
    } catch (err) {
      console.error("Failed to fetch providers:", err);
      setProviderFetchStatus("error");
      setProviderError(err instanceof Error ? err.message : "Unknown error");
    }
  }, [client]);

  const fetchRoutes = useCallback(async () => {
    setRouteFetchStatus("loading");
    try {
      const response = await fetch("/api/v1alpha/admin/routes");
      if (response.status === 404 || response.status === 501) {
        setRoutes([]);
        setRouteFetchStatus("success");
        return;
      }
      if (!response.ok) throw new Error(`HTTP ${response.status}`);
      const data = await response.json();
      setRoutes(Array.isArray(data) ? data : []);
      setRouteFetchStatus("success");
      setRouteError(null);
    } catch (err) {
      console.error("Failed to fetch routes:", err);
      setRouteFetchStatus("error");
      setRouteError(err instanceof Error ? err.message : "Unknown error");
    }
  }, []);

  useEffect(() => {
    fetchHealth();
    fetchVersion();
    fetchProviders();
    fetchRoutes();
  }, [fetchHealth, fetchVersion, fetchProviders, fetchRoutes]);

  const handleRefresh = () => {
    fetchHealth();
    fetchVersion();
    fetchProviders();
    fetchRoutes();
  };

  const toggleProviderExpanded = (providerId: string) => {
    setExpandedProviders(prev => {
      const next = new Set(prev);
      if (next.has(providerId)) {
        next.delete(providerId);
      } else {
        next.add(providerId);
      }
      return next;
    });
  };

  const uniqueApis = Array.from(new Set(providers.map(p => p.api))).sort();

  const filteredProviders = providers.filter(provider => {
    const matchesSearch =
      !providerSearch ||
      provider.provider_id
        .toLowerCase()
        .includes(providerSearch.toLowerCase()) ||
      provider.provider_type
        .toLowerCase()
        .includes(providerSearch.toLowerCase()) ||
      provider.api.toLowerCase().includes(providerSearch.toLowerCase());
    const matchesApi =
      providerApiFilter === "all" || provider.api === providerApiFilter;
    return matchesSearch && matchesApi;
  });

  const filteredRoutes = routes.filter(route => {
    if (!routeSearch) return true;
    const search = routeSearch.toLowerCase();
    return (
      route.route?.toLowerCase().includes(search) ||
      route.method?.toLowerCase().includes(search) ||
      route.provider_type?.toLowerCase().includes(search)
    );
  });

  const renderHealthCard = () => (
    <Card>
      <CardHeader>
        <CardTitle>Server Health</CardTitle>
        <CardDescription>Current server status and version</CardDescription>
      </CardHeader>
      <CardContent>
        <div className="flex flex-col gap-4">
          <div className="flex items-center gap-3">
            <span className="text-sm font-medium text-muted-foreground w-20">
              Status
            </span>
            {healthFetchStatus === "loading" ? (
              <Skeleton className="h-6 w-24" />
            ) : healthFetchStatus === "error" ? (
              <Badge variant="destructive">Unreachable</Badge>
            ) : (
              <Badge
                className={
                  healthStatus?.status === "OK"
                    ? "bg-green-600 hover:bg-green-700 text-white border-transparent"
                    : "bg-yellow-600 hover:bg-yellow-700 text-white border-transparent"
                }
              >
                {healthStatus?.status || "Unknown"}
              </Badge>
            )}
            {healthError && (
              <span className="text-sm text-destructive">{healthError}</span>
            )}
          </div>
          <div className="flex items-center gap-3">
            <span className="text-sm font-medium text-muted-foreground w-20">
              Version
            </span>
            {versionFetchStatus === "loading" ? (
              <Skeleton className="h-5 w-32" />
            ) : versionFetchStatus === "error" ? (
              <span className="text-sm text-muted-foreground">
                Unable to retrieve
              </span>
            ) : (
              <span className="text-sm font-mono">
                {versionInfo?.version || "Unknown"}
              </span>
            )}
          </div>
          <div className="flex items-center gap-3">
            <span className="text-sm font-medium text-muted-foreground w-20">
              Providers
            </span>
            {providerFetchStatus === "loading" ? (
              <Skeleton className="h-5 w-16" />
            ) : (
              <span className="text-sm">{providers.length} registered</span>
            )}
          </div>
          <div className="flex items-center gap-3">
            <span className="text-sm font-medium text-muted-foreground w-20">
              APIs
            </span>
            {providerFetchStatus === "loading" ? (
              <Skeleton className="h-5 w-16" />
            ) : (
              <span className="text-sm">{uniqueApis.length} active</span>
            )}
          </div>
        </div>
      </CardContent>
    </Card>
  );

  const renderProvidersTable = () => {
    if (providerFetchStatus === "loading") {
      return (
        <div className="space-y-2">
          <Skeleton className="h-8 w-full" />
          <Skeleton className="h-4 w-full" />
          <Skeleton className="h-4 w-full" />
          <Skeleton className="h-4 w-full" />
        </div>
      );
    }

    if (providerFetchStatus === "error") {
      return (
        <div className="text-destructive">
          Failed to load providers: {providerError}
        </div>
      );
    }

    if (providers.length === 0) {
      return (
        <div className="text-center py-8">
          <p className="text-muted-foreground">No providers registered.</p>
        </div>
      );
    }

    return (
      <div className="space-y-4">
        <div className="flex items-center gap-4">
          <div className="relative flex-1 max-w-md">
            <Search className="absolute left-3 top-1/2 transform -translate-y-1/2 text-muted-foreground h-4 w-4" />
            <Input
              placeholder="Search providers..."
              value={providerSearch}
              onChange={e => setProviderSearch(e.target.value)}
              className="pl-10"
            />
          </div>
          <div className="flex gap-2 flex-wrap">
            <Button
              variant={providerApiFilter === "all" ? "default" : "outline"}
              size="sm"
              onClick={() => setProviderApiFilter("all")}
            >
              All ({providers.length})
            </Button>
            {uniqueApis.map(api => {
              const count = providers.filter(p => p.api === api).length;
              return (
                <Button
                  key={api}
                  variant={providerApiFilter === api ? "default" : "outline"}
                  size="sm"
                  onClick={() => setProviderApiFilter(api)}
                >
                  {api} ({count})
                </Button>
              );
            })}
          </div>
        </div>

        <div className="overflow-auto">
          <Table>
            <TableHeader>
              <TableRow>
                <TableHead className="w-8"></TableHead>
                <TableHead>Provider ID</TableHead>
                <TableHead>Provider Type</TableHead>
                <TableHead>API</TableHead>
              </TableRow>
            </TableHeader>
            <TableBody>
              {filteredProviders.map(provider => {
                const key = `${provider.api}-${provider.provider_id}`;
                const isExpanded = expandedProviders.has(key);
                const hasConfig =
                  provider.config && Object.keys(provider.config).length > 0;

                return (
                  <React.Fragment key={key}>
                    <TableRow>
                      <TableCell className="w-8 p-2">
                        {hasConfig && (
                          <Collapsible
                            open={isExpanded}
                            onOpenChange={() => toggleProviderExpanded(key)}
                          >
                            <CollapsibleTrigger asChild>
                              <Button
                                variant="ghost"
                                size="sm"
                                className="h-6 w-6 p-0"
                              >
                                {isExpanded ? (
                                  <ChevronDown className="h-4 w-4" />
                                ) : (
                                  <ChevronRight className="h-4 w-4" />
                                )}
                              </Button>
                            </CollapsibleTrigger>
                          </Collapsible>
                        )}
                      </TableCell>
                      <TableCell className="font-mono text-sm">
                        {provider.provider_id}
                      </TableCell>
                      <TableCell>
                        <Badge variant="outline">
                          {provider.provider_type}
                        </Badge>
                      </TableCell>
                      <TableCell>
                        <Badge variant="secondary">{provider.api}</Badge>
                      </TableCell>
                    </TableRow>
                    {hasConfig && isExpanded && (
                      <TableRow>
                        <TableCell colSpan={4} className="bg-muted/50 p-4">
                          <pre className="text-xs font-mono whitespace-pre-wrap overflow-auto max-h-64">
                            {JSON.stringify(provider.config, null, 2)}
                          </pre>
                        </TableCell>
                      </TableRow>
                    )}
                  </React.Fragment>
                );
              })}
            </TableBody>
          </Table>
        </div>

        {filteredProviders.length === 0 && (
          <div className="text-center py-4">
            <p className="text-muted-foreground">
              No providers match your search.
            </p>
          </div>
        )}
      </div>
    );
  };

  const renderRoutesTable = () => {
    if (routeFetchStatus === "loading") {
      return (
        <div className="space-y-2">
          <Skeleton className="h-8 w-full" />
          <Skeleton className="h-4 w-full" />
          <Skeleton className="h-4 w-full" />
          <Skeleton className="h-4 w-full" />
        </div>
      );
    }

    if (routeFetchStatus === "error") {
      return (
        <div className="text-destructive">
          Failed to load routes: {routeError}
        </div>
      );
    }

    if (routes.length === 0) {
      return (
        <div className="text-center py-8">
          <p className="text-muted-foreground">No routes registered.</p>
        </div>
      );
    }

    return (
      <div className="space-y-4">
        <div className="relative flex-1 max-w-md">
          <Search className="absolute left-3 top-1/2 transform -translate-y-1/2 text-muted-foreground h-4 w-4" />
          <Input
            placeholder="Search routes..."
            value={routeSearch}
            onChange={e => setRouteSearch(e.target.value)}
            className="pl-10"
          />
        </div>

        <div className="overflow-auto">
          <Table>
            <TableHeader>
              <TableRow>
                <TableHead>Route</TableHead>
                <TableHead>Method</TableHead>
                <TableHead>Provider</TableHead>
              </TableRow>
            </TableHeader>
            <TableBody>
              {filteredRoutes.map((route, index) => (
                <TableRow key={`${route.route}-${route.method}-${index}`}>
                  <TableCell className="font-mono text-sm">
                    {route.route}
                  </TableCell>
                  <TableCell>
                    <Badge variant="outline">{route.method}</Badge>
                  </TableCell>
                  <TableCell>
                    <Badge variant="secondary">{route.provider_type}</Badge>
                  </TableCell>
                </TableRow>
              ))}
            </TableBody>
          </Table>
        </div>

        {filteredRoutes.length === 0 && (
          <div className="text-center py-4">
            <p className="text-muted-foreground">
              No routes match your search.
            </p>
          </div>
        )}
      </div>
    );
  };

  return (
    <div className="space-y-4">
      <div className="flex items-center justify-between">
        <h1 className="text-2xl font-semibold">System Administration</h1>
        <Button variant="outline" onClick={handleRefresh}>
          <RefreshCw className="h-4 w-4 mr-2" />
          Refresh
        </Button>
      </div>

      {renderHealthCard()}

      <Tabs defaultValue="providers">
        <TabsList>
          <TabsTrigger value="providers">
            Providers
            {providerFetchStatus === "success" && (
              <span className="ml-1.5 text-xs text-muted-foreground">
                ({providers.length})
              </span>
            )}
          </TabsTrigger>
          <TabsTrigger value="routes">
            Routes
            {routeFetchStatus === "success" && (
              <span className="ml-1.5 text-xs text-muted-foreground">
                ({routes.length})
              </span>
            )}
          </TabsTrigger>
        </TabsList>
        <TabsContent value="providers">{renderProvidersTable()}</TabsContent>
        <TabsContent value="routes">{renderRoutesTable()}</TabsContent>
      </Tabs>
    </div>
  );
}
