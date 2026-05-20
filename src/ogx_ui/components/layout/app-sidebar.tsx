"use client";

import { useState, useEffect, useCallback } from "react";
import {
  MessageSquareText,
  MessagesSquare,
  MoveUpRight,
  Database,
  MessageCircle,
  Settings2,
  Compass,
  FileText,
  File,
  ChevronRight,
  Box,
  Plug,
  Wrench,
  Layers,
  MessageSquare,
  Trash2,
} from "lucide-react";
import Link from "next/link";
import { usePathname } from "next/navigation";
import { cn } from "@/lib/utils";
import {
  getConversationHistory,
  removeConversation,
  type ConversationHistoryEntry,
} from "@/lib/conversation-history";
import {
  Collapsible,
  CollapsibleContent,
  CollapsibleTrigger,
} from "@/components/ui/collapsible";

import {
  Sidebar,
  SidebarContent,
  SidebarGroup,
  SidebarGroupContent,
  SidebarGroupLabel,
  SidebarMenu,
  SidebarMenuButton,
  SidebarMenuItem,
  SidebarMenuSub,
  SidebarMenuSubItem,
  SidebarMenuSubButton,
  SidebarHeader,
} from "@/components/ui/sidebar";

const manageItems = [
  {
    title: "Chat Completions",
    url: "/logs/chat-completions",
    icon: MessageSquareText,
  },
  {
    title: "Responses",
    url: "/logs/responses",
    icon: MessagesSquare,
  },
  {
    title: "Vector Stores",
    url: "/logs/vector-stores",
    icon: Database,
  },
  {
    title: "Files",
    url: "/logs/files",
    icon: File,
  },
  {
    title: "Models",
    url: "/models",
    icon: Box,
  },
  {
    title: "Connectors",
    url: "/connectors",
    icon: Plug,
  },
  {
    title: "Tools",
    url: "/tools",
    icon: Wrench,
  },
  {
    title: "Conversations",
    url: "/conversations",
    icon: MessageSquare,
  },
  {
    title: "Batches",
    url: "/batches",
    icon: Layers,
  },
  {
    title: "Prompts",
    url: "/prompts",
    icon: FileText,
  },
  {
    title: "Documentation",
    url: "https://ogx.readthedocs.io/en/latest/references/api_reference/index.html",
    icon: MoveUpRight,
  },
];

const adminItems = [
  {
    title: "System",
    url: "/admin",
    icon: Settings2,
  },
];

const optimizeItems: { title: string; url: string; icon: React.ElementType }[] =
  [
    {
      title: "Evaluations",
      url: "",
      icon: Compass,
    },
    {
      title: "Fine-tuning",
      url: "",
      icon: Settings2,
    },
  ];

interface SidebarItem {
  title: string;
  url: string;
  icon: React.ElementType;
}

export function AppSidebar() {
  const pathname = usePathname();
  const [conversations, setConversations] = useState<
    ConversationHistoryEntry[]
  >([]);
  const [isConversationsOpen, setIsConversationsOpen] = useState(() => {
    if (typeof window === "undefined") return false;
    return sessionStorage.getItem("sidebar-conversations-open") === "true";
  });

  useEffect(() => {
    sessionStorage.setItem(
      "sidebar-conversations-open",
      String(isConversationsOpen)
    );
  }, [isConversationsOpen]);

  // Load conversations from localStorage when expanded or updated
  const refreshConversations = useCallback(() => {
    setConversations(getConversationHistory());
  }, []);

  useEffect(() => {
    if (isConversationsOpen) {
      refreshConversations();
    }
  }, [isConversationsOpen, refreshConversations]);

  // Listen for updates from the chat playground
  useEffect(() => {
    const handler = () => refreshConversations();
    window.addEventListener("conversations-updated", handler);
    return () => window.removeEventListener("conversations-updated", handler);
  }, [refreshConversations]);

  // Refresh when navigating to chat playground
  useEffect(() => {
    if (pathname === "/chat-playground") {
      refreshConversations();
    }
  }, [pathname, refreshConversations]);

  const renderSidebarItems = (items: SidebarItem[]) => {
    return items.map(item => {
      const isActive = pathname.startsWith(item.url);
      return (
        <SidebarMenuItem key={item.title}>
          <SidebarMenuButton
            asChild
            className={cn(
              "justify-start",
              isActive &&
                "bg-gray-200 dark:bg-gray-700 hover:bg-gray-200 dark:hover:bg-gray-700 text-gray-900 dark:text-gray-100"
            )}
          >
            <Link href={item.url}>
              <item.icon
                className={cn(
                  isActive && "text-gray-900 dark:text-gray-100",
                  "mr-2 h-4 w-4"
                )}
              />
              <span>{item.title}</span>
            </Link>
          </SidebarMenuButton>
        </SidebarMenuItem>
      );
    });
  };

  const isChatActive = pathname.startsWith("/chat-playground");

  const formatConversationLabel = (conv: ConversationHistoryEntry) => {
    const date = new Date(conv.createdAt);
    const now = new Date();
    const isToday = date.toDateString() === now.toDateString();
    const yesterday = new Date(now);
    yesterday.setDate(yesterday.getDate() - 1);
    const isYesterday = date.toDateString() === yesterday.toDateString();

    const time = date.toLocaleTimeString("en-US", {
      hour: "numeric",
      minute: "2-digit",
    });

    if (isToday) return `Today ${time}`;
    if (isYesterday) return `Yesterday ${time}`;
    return (
      date.toLocaleDateString("en-US", {
        month: "short",
        day: "numeric",
      }) + ` ${time}`
    );
  };

  return (
    <Sidebar>
      <SidebarHeader>
        <Link href="/" className="flex items-center gap-2 p-2">
          <span className="font-semibold text-lg">OGX</span>
        </Link>
      </SidebarHeader>
      <SidebarContent>
        <SidebarGroup>
          <SidebarGroupLabel>Create</SidebarGroupLabel>
          <SidebarGroupContent>
            <SidebarMenu>
              {/* Chat Playground with collapsible conversations */}
              <Collapsible
                open={isConversationsOpen}
                onOpenChange={setIsConversationsOpen}
                className="group/collapsible"
              >
                <SidebarMenuItem>
                  <SidebarMenuButton
                    asChild
                    className={cn(
                      "justify-start",
                      isChatActive &&
                        "bg-gray-200 dark:bg-gray-700 hover:bg-gray-200 dark:hover:bg-gray-700 text-gray-900 dark:text-gray-100"
                    )}
                  >
                    <Link href="/chat-playground">
                      <MessageCircle
                        className={cn(
                          isChatActive && "text-gray-900 dark:text-gray-100",
                          "mr-2 h-4 w-4"
                        )}
                      />
                      <span className="flex-1">Chat Playground</span>
                    </Link>
                  </SidebarMenuButton>
                  <CollapsibleTrigger asChild>
                    <button
                      className="absolute right-1 top-1/2 -translate-y-1/2 p-1 rounded hover:bg-muted"
                      title="Show conversations"
                    >
                      <ChevronRight className="h-3.5 w-3.5 text-muted-foreground transition-transform group-data-[state=open]/collapsible:rotate-90" />
                    </button>
                  </CollapsibleTrigger>
                </SidebarMenuItem>
                <CollapsibleContent>
                  <SidebarMenuSub>
                    {conversations.length === 0 ? (
                      <SidebarMenuSubItem>
                        <span className="px-2 py-1 text-xs text-muted-foreground">
                          No conversations yet
                        </span>
                      </SidebarMenuSubItem>
                    ) : (
                      conversations.slice(0, 20).map(conv => (
                        <SidebarMenuSubItem
                          key={conv.id}
                          className="group/conv relative"
                        >
                          <SidebarMenuSubButton asChild>
                            <Link
                              href={`/chat-playground?conversation=${conv.id}`}
                              title={conv.firstMessage || conv.id}
                            >
                              <span className="truncate text-xs pr-5">
                                {conv.firstMessage
                                  ? conv.firstMessage.length > 30
                                    ? conv.firstMessage.substring(0, 30) + "..."
                                    : conv.firstMessage
                                  : formatConversationLabel(conv)}
                              </span>
                            </Link>
                          </SidebarMenuSubButton>
                          <button
                            className="absolute right-1 top-1/2 -translate-y-1/2 p-0.5 rounded opacity-0 group-hover/conv:opacity-100 hover:bg-destructive/20 hover:text-destructive transition-opacity"
                            title="Delete conversation"
                            onClick={e => {
                              e.preventDefault();
                              e.stopPropagation();
                              removeConversation(conv.id);
                              refreshConversations();
                            }}
                          >
                            <Trash2 className="h-3 w-3" />
                          </button>
                        </SidebarMenuSubItem>
                      ))
                    )}
                  </SidebarMenuSub>
                </CollapsibleContent>
              </Collapsible>
            </SidebarMenu>
          </SidebarGroupContent>
        </SidebarGroup>

        <SidebarGroup>
          <SidebarGroupLabel>Manage</SidebarGroupLabel>
          <SidebarGroupContent>
            <SidebarMenu>{renderSidebarItems(manageItems)}</SidebarMenu>
          </SidebarGroupContent>
        </SidebarGroup>

        <SidebarGroup>
          <SidebarGroupLabel>Admin</SidebarGroupLabel>
          <SidebarGroupContent>
            <SidebarMenu>{renderSidebarItems(adminItems)}</SidebarMenu>
          </SidebarGroupContent>
        </SidebarGroup>

        <SidebarGroup>
          <SidebarGroupLabel>Optimize</SidebarGroupLabel>
          <SidebarGroupContent>
            <SidebarMenu>
              {optimizeItems.map(item => (
                <SidebarMenuItem key={item.title}>
                  <SidebarMenuButton
                    disabled
                    className="justify-start opacity-60 cursor-not-allowed"
                  >
                    <item.icon className="mr-2 h-4 w-4" />
                    <span>{item.title}</span>
                    <span className="ml-2 text-xs text-gray-500">
                      (Coming Soon)
                    </span>
                  </SidebarMenuButton>
                </SidebarMenuItem>
              ))}
            </SidebarMenu>
          </SidebarGroupContent>
        </SidebarGroup>
      </SidebarContent>
    </Sidebar>
  );
}
