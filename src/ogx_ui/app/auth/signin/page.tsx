"use client";

import { signIn, signOut, useSession } from "next-auth/react";
import { Button } from "@/components/ui/button";
import {
  Card,
  CardContent,
  CardDescription,
  CardHeader,
  CardTitle,
} from "@/components/ui/card";
import { Copy, Check, Home, Github } from "lucide-react";
import { useState } from "react";
import { useRouter } from "next/navigation";

export default function SignInPage() {
  const { data: session, status } = useSession();
  const [copied, setCopied] = useState(false);
  const router = useRouter();

  const handleCopyToken = async () => {
    if (session?.accessToken) {
      await navigator.clipboard.writeText(session.accessToken);
      setCopied(true);
      setTimeout(() => setCopied(false), 2000);
    }
  };

  if (status === "loading") {
    return (
      <div className="flex items-center justify-center min-h-screen">
        <div className="text-muted-foreground">Loading...</div>
      </div>
    );
  }

  return (
    <div className="flex items-center justify-center min-h-screen">
      <Card className="w-[400px]">
        <CardHeader>
          <CardTitle>Authentication</CardTitle>
          <CardDescription>
            {session
              ? "You are successfully authenticated!"
              : "Sign in with GitHub to use your access token as an API key"}
          </CardDescription>
        </CardHeader>
        <CardContent className="space-y-4">
          {!session ? (
            <Button
              onClick={() => {
                console.log("Signing in with GitHub...");
                signIn("github", { callbackUrl: "/auth/signin" }).catch(
                  error => {
                    console.error("Sign in error:", error);
                  }
                );
              }}
              className="w-full"
              variant="default"
            >
              <Github className="mr-2 h-4 w-4" />
              Sign in with GitHub
            </Button>
          ) : (
            <div className="space-y-4">
              <div className="text-sm text-muted-foreground">
                Signed in as {session.user?.email}
              </div>

              {session.accessToken && (
                <div className="space-y-2">
                  <div className="text-sm font-medium">
                    GitHub Access Token:
                  </div>
                  <div className="flex gap-2">
                    <code className="flex-1 p-2 bg-muted rounded text-xs break-all">
                      {session.accessToken}
                    </code>
                    <Button
                      size="sm"
                      variant="outline"
                      onClick={handleCopyToken}
                    >
                      {copied ? (
                        <Check className="h-4 w-4" />
                      ) : (
                        <Copy className="h-4 w-4" />
                      )}
                    </Button>
                  </div>
                  <div className="text-xs text-muted-foreground">
                    This GitHub token will be used as your API key for
                    authenticated OGX requests.
                  </div>
                </div>
              )}

              <div className="flex gap-2">
                <Button onClick={() => router.push("/")} className="flex-1">
                  <Home className="mr-2 h-4 w-4" />
                  Go to Dashboard
                </Button>
                <Button
                  onClick={() => signOut()}
                  variant="outline"
                  className="flex-1"
                >
                  Sign out
                </Button>
              </div>
            </div>
          )}
        </CardContent>
      </Card>
    </div>
  );
}
