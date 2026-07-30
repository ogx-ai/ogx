import type { Metadata } from "next";
import { ThemeProvider } from "@/components/ui/theme-provider";
import { SessionProvider } from "@/components/providers/session-provider";
import { Geist, Geist_Mono } from "next/font/google";
import { ModeToggle } from "@/components/ui/mode-toggle";
import "./globals.css";

const geistSans = Geist({
  variable: "--font-geist-sans",
  subsets: ["latin"],
});

const geistMono = Geist_Mono({
  variable: "--font-geist-mono",
  subsets: ["latin"],
});

export const metadata: Metadata = {
  title: "OGX",
  description: "OGX UI",
  icons: {
    icon: "/favicon.ico",
  },
};

import { SidebarProvider, SidebarTrigger } from "@/components/ui/sidebar";
import { AppSidebar } from "@/components/layout/app-sidebar";
import { SignInButton } from "@/components/ui/sign-in-button";

export default function Layout({ children }: { children: React.ReactNode }) {
  return (
    <html lang="en" suppressHydrationWarning>
      <body className={`${geistSans.variable} ${geistMono.variable} font-sans`}>
        <SessionProvider>
          <ThemeProvider
            attribute="class"
            defaultTheme="system"
            enableSystem
            disableTransitionOnChange
          >
            <SidebarProvider>
              <AppSidebar />
              <main className="flex flex-col flex-1">
                {/* Header with aligned elements */}
                <div className="flex items-center p-4 border-b">
                  <div className="flex-none">
                    <SidebarTrigger />
                  </div>
                  <div className="flex-1 text-center"></div>
                  <div className="flex-none flex items-center gap-2">
                    <SignInButton />
                    <ModeToggle />
                  </div>
                </div>
                <div className="flex flex-col flex-1 p-4">{children}</div>
              </main>
            </SidebarProvider>
          </ThemeProvider>
        </SessionProvider>
      </body>
    </html>
  );
}
