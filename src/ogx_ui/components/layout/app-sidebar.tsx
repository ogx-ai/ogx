"use client";

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
  Box,
  Plug,
  Wrench,
  Shield,
  ShieldCheck,
  Layers,
  MessageSquare,
} from "lucide-react";
import Link from "next/link";
import { usePathname } from "next/navigation";
import { cn } from "@/lib/utils";

import {
  Sidebar,
  SidebarContent,
  SidebarGroup,
  SidebarGroupContent,
  SidebarGroupLabel,
  SidebarMenu,
  SidebarMenuButton,
  SidebarMenuItem,
  SidebarHeader,
} from "@/components/ui/sidebar";

const createItems = [
  {
    title: "Chat Playground",
    url: "/chat-playground",
    icon: MessageCircle,
  },
  {
    title: "Safety",
    url: "/safety",
    icon: ShieldCheck,
  },
];

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
    icon: Shield,
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
            <SidebarMenu>{renderSidebarItems(createItems)}</SidebarMenu>
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
